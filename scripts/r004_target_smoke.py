"""R004 for the two gated image models that do not take the certified text interface (gates opened 2026-10-21):

- RadEdit, the E2 positive control (target state given): both arms of the smoke E2 test of r004_smoke.py (ETT
  insertion on a prior ASSUMED absent) as target states, arm A "ETT absent" and arm B "ETT present", each an edit of
  the prior with the ETT region mask (spectra.generators.hf.RadEdit);
- RoentGen-v2, the initialization of the reference family and the external reference pair (not an evaluated
  system): one text-to-image sample with its model card's prompt and default sampler.

Uses the prior film that r004_smoke.py prepared in --out (prior.png); records seconds per generation and peak GPU
memory per model, and merges them into --out/summary.json.

    python scripts/r004_target_smoke.py --out ../runs/R004/cluster_gated
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r004_smoke import SEED, _prior_corr  # noqa: E402

ROENTGEN_PROMPT = "50 year old female. Normal chest radiograph."   # the model card's example


def radedit(out: Path, prior) -> dict:
    import torch

    from spectra.config import systems
    from spectra.generators.hf import RadEdit
    from spectra.images import to_eval

    gen = RadEdit("radedit", systems()["radedit"])
    t = time.time()
    gen.load()
    gen._loaded = True
    rec = {"load_seconds": round(time.time() - t, 1), "outputs": []}
    for arm, present in (("A", False), ("B", True)):
        o = gen.target(prior, "ETT", present, seed=SEED)
        f = out / f"radedit_S{arm}.png"
        o.image.save(f)
        rec["outputs"].append({"mode": "S", "arm": arm, "target_ETT": present, "seconds": round(o.seconds, 2),
                               "peak_gpu_gb": round(o.peak_gpu_gb, 2), "file": str(f), "native_size": o.meta["native_size"],
                               "prior_corr": _prior_corr(to_eval(prior), o.image)})   # on the same 512² canvas
    del gen
    torch.cuda.empty_cache()
    return rec


def roentgen(out: Path) -> dict:
    import torch
    from diffusers import DiffusionPipeline

    from spectra.config import systems

    t = time.time()
    pipe = DiffusionPipeline.from_pretrained(systems()["external_reference_pair"]["init"], torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    rec = {"load_seconds": round(time.time() - t, 1), "outputs": []}
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.inference_mode():
        img = pipe(ROENTGEN_PROMPT, generator=torch.Generator("cuda").manual_seed(SEED)).images[0]
    torch.cuda.synchronize()
    f = out / "roentgen_v2_T2I.png"
    img.save(f)
    rec["outputs"].append({"mode": "text-to-image", "input": ROENTGEN_PROMPT, "seconds": round(time.time() - t, 2),
                           "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "file": str(f),
                           "native_size": list(img.size)})
    return rec


def main(argv=None) -> int:
    import torch
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    prior = Image.open(out / "prior.png")
    summ_f = out / "summary.json"
    summary = json.loads(summ_f.read_text()) if summ_f.exists() else {}
    for name, fn in (("radedit", lambda: radedit(out, prior)), ("roentgen_v2_init", lambda: roentgen(out))):
        rec = {"system": name, "host": platform.node(), "torch": torch.__version__,
               "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        try:
            rec.update(fn(), ok=True)
        except Exception as e:  # recorded, then the next model
            rec.update(ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
        (out / f"{name}.json").write_text(json.dumps(rec, indent=1, default=str))
        summary[name] = {"status": "ok" if rec["ok"] else "failed", "roundtrip_ok": None, "load_seconds": rec.get("load_seconds"),
                         "gpus": rec["gpus"], "seconds": [o["seconds"] for o in rec.get("outputs", [])],
                         "peak_gpu_gb": max([o["peak_gpu_gb"] for o in rec.get("outputs", [])], default=None),
                         "prior_corr": [o["prior_corr"] for o in rec.get("outputs", []) if "prior_corr" in o] or None,
                         "error": (rec.get("error") or "")[:200] or None}
        print(name, summary[name]["status"], flush=True)
    summ_f.write_text(json.dumps(summary, indent=1))
    return 0 if all(summary[n]["status"] == "ok" for n in ("radedit", "roentgen_v2_init")) else 1


if __name__ == "__main__":
    sys.exit(main())
