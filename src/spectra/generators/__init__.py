"""Generation adapters of the registered systems (resources/systems.yaml); see base.py for the contract."""
from __future__ import annotations

from importlib import import_module

from .base import READ_RES, Generator, Output, parse_state

ADAPTERS = {
    "qwen_image_edit_2511": ("hf", "QwenImageEdit"),
    "flux2_klein_4b": ("hf", "Flux2Klein"),
    "lingshu_32b": ("hf", "StatePredictor"),
    "medgemma_1_5_4b": ("hf", "StatePredictor"),
    "progemu": ("third_party", "ProgEmu"),
    "unimedvl": ("third_party", "UniMedVL"),
    "prism": ("third_party", "Prism"),
}


def make(system: str, device: str = "cuda") -> Generator:
    from ..config import systems

    if system not in ADAPTERS:
        raise KeyError(f"no generation adapter for {system!r}")
    mod, cls = ADAPTERS[system]
    return getattr(import_module(f".{mod}", __name__), cls)(system, systems()[system], device)
