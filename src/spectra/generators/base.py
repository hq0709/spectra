"""Generation adapters (R004 smoke, Gate-2 throughput, Gate-4 generation).

An adapter receives exactly what the round-trip check certified: the templated text of `Interface.inputs`
(mode S) or `Interface.mode_r_inputs` (mode R), plus the prior film rendered to 8 bits (spectra.images) and a
seed. Image systems return the generated follow-up at their native size (the evaluator reads every film at
the common 512²); state predictors return their greedy response and the parsed state (None when it does not
parse: the failure rule then counts the readout as incorrect).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

READ_RES = 512
GRADE_KEYS = {"edema": "edema", "effusion_left": "effusion_L", "effusion_right": "effusion_R"}
DEVICE_KEYS = {"endotracheal_tube": "ETT", "central_venous_catheter": "CVC", "picc": "PICC", "enteric_tube": "enteric",
               "chest_tube": "chest_tube"}
SIDES = {"left": "L", "right": "R", "none": "none"}


@dataclass
class Output:
    system: str
    kind: str                          # image | state
    image: Optional[object] = None     # PIL image, grayscale, at the system's native size (read at READ_RES²)
    text: Optional[str] = None         # raw response of a state predictor
    state: Optional[dict] = None       # {"grades": {channel: 0-4}, "devices": {device: present}, "CVC_side": L|R|none}
    seconds: float = 0.0
    peak_gpu_gb: float = 0.0
    meta: dict = field(default_factory=dict)


class NoImage(RuntimeError):
    """A generation that ended without an output image (the E1 failure rule then scores its contrasts incorrect)."""


class Generator:
    """One evaluated configuration. Subclasses implement `load` (weights onto the device) and `_run`."""

    kind = "image"

    def __init__(self, name: str, spec: dict, device: str = "cuda"):
        self.name, self.spec, self.device = name, spec, device
        self._loaded = False

    def load(self) -> None:
        raise NotImplementedError

    def _run(self, prior, text: str, seed: int):
        raise NotImplementedError

    def generate(self, prior, text: str, seed: int) -> Output:
        if prior.mode not in ("L", "RGB"):
            raise ValueError(f"prior film in mode {prior.mode}: render it with spectra.images.to_uint8 first")
        import torch

        if not self._loaded:
            self.load()
            self._loaded = True
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        res = self._run(prior.convert("RGB"), text, seed)
        torch.cuda.synchronize()
        out = Output(self.name, self.kind, seconds=time.time() - t,
                     peak_gpu_gb=sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2 ** 30)
        if self.kind == "image":
            out.meta["native_size"] = list(res.size)
            out.image = res.convert("L")
        else:
            out.text, out.state = res, parse_state(res)
        return out


def parse_state(text: str) -> Optional[dict]:
    """The state in a predictor's response: the first JSON object, with every answer key of the template
    (resources/templates.yaml, answer_format) of the right type; anything else is unparseable (None)."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    try:
        grades = {ch: int(d[k]) for k, ch in GRADE_KEYS.items()}
        devices = {dev: d[k] for k, dev in DEVICE_KEYS.items()}
        side = SIDES[str(d["central_venous_catheter_side"]).lower()]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(0 <= g <= 4 for g in grades.values()) or not all(isinstance(v, bool) for v in devices.values()):
        return None
    return {"grades": grades, "devices": devices, "CVC_side": side}
