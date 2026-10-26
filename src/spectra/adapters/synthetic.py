"""Synthetic source for unit tests: films, graded findings and device streams with known answers."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..types import CER, Device, DeviceState, Film
from .base import SourceAdapter

CHANNELS = ("edema", "effusion_L", "effusion_R")


class SyntheticSource(SourceAdapter):
    """n_patients ICU stays, each with n_films frontal films 6-30 h apart and random ordinal grades."""

    name = "synthetic"
    time_resolution = "minute"
    has_grades = True

    def __init__(self, n_patients: int = 40, n_films: int = 5, seed: int = 0, tie_rate: float = 0.3):
        rng = np.random.default_rng(seed)
        self._films: List[Film] = []
        self._grades: Dict[str, Dict[str, Optional[int]]] = {}
        for p in range(n_patients):
            pid = f"P{p:04d}"
            t = float(rng.uniform(0, 1000))
            g = {c: int(rng.integers(0, 5)) for c in CHANNELS}
            for k in range(n_films):
                sid = f"{pid}-S{k}"
                self._films.append(Film(pid, sid, sid + "-img", t, int(t // 24), int((t % 24) * 60), f"stay{p}", "AP",
                                        "test", "synthetic"))
                self._grades[sid] = dict(g)
                t += float(rng.uniform(6, 30))
                for c in CHANNELS:
                    if rng.random() > tie_rate:
                        g[c] = int(np.clip(g[c] + rng.choice([-1, 1]), 0, 4))
            # one channel never changes for one patient in five: its contrasts are all ties (undefined)
            if p % 5 == 0:
                for k in range(n_films):
                    self._grades[f"{pid}-S{k}"]["effusion_R"] = 2

    def films(self) -> List[Film]:
        return list(self._films)

    def grades(self) -> Mapping[str, Mapping[str, Optional[int]]]:
        return self._grades


def ett_placement_stream(t_place: float = 3.0, with_settings: bool = True) -> Tuple[CER, ...]:
    """A window where an endotracheal tube is placed: intubation, ventilation start, settings, an NMB,
    plus unrelated background actions."""
    xs = [
        CER("action", "diuretic", 0.5, value="Furosemide (Lasix)", dose_value=40.0, dose_unit="mg", priority="diuretic_vasoactive"),
        CER("transition", "intubation", t_place, device=Device("ETT", "place", "nasal/oral", "none", "Intubation"), priority="device_procedure"),
        CER("action", "invasive_ventilation", t_place + 0.05, ongoing_at_delta=True, priority="device_procedure"),
        CER("action", "neuromuscular_blocker", t_place + 0.1, value="Rocuronium", dose_value=50.0, dose_unit="mg", priority="other_medication"),
        CER("action", "antibiotic", 6.0, value="Vancomycin", dose_value=1.0, dose_unit="dose", priority="antibiotic"),
        CER("action", "fluid", 8.0, value="LR", dose_value=500.0, dose_unit="ml", priority="fluid"),
    ]
    if with_settings:
        xs += [CER("action", "ventilator_setting", t_place + 0.2, value="PEEP set=5", priority="ventilator_setting"),
               CER("action", "ventilator_setting", t_place + 0.2, value="Ventilator Mode=CMV/ASSIST", priority="ventilator_setting")]
    return tuple(sorted(xs, key=lambda c: c.sort_key()))


def ett_removal_stream(t_remove: float = 5.0) -> Tuple[CER, ...]:
    xs = [
        CER("state", "invasive_ventilation", None, priority="device_procedure"),
        CER("state", "ventilator_setting", None, value="PEEP set=8", priority="ventilator_setting"),
        CER("action", "ventilator_setting", 1.0, value="PEEP set=5", priority="ventilator_setting"),
        CER("transition", "extubation", t_remove, device=Device("ETT", "remove", "nasal/oral", "none", "Extubation"), priority="device_procedure"),
        CER("action", "invasive_ventilation_stop", t_remove, priority="device_procedure"),
        CER("action", "noninvasive_ventilation", t_remove + 0.5, priority="device_procedure"),
        CER("action", "vasoactive", 2.0, value="Norepinephrine", dose_value=2.0, dose_unit="mg", priority="diuretic_vasoactive"),
    ]
    return tuple(sorted(xs, key=lambda c: c.sort_key()))


def cvc_placement_stream(t_place: float = 2.0, side: str = "R", site: str = "IJ") -> Tuple[CER, ...]:
    xs = [
        CER("transition", "line_placement", t_place, device=Device("CVC", "place", site, side, "Multi Lumen"), priority="device_procedure"),
        CER("action", "vasoactive", 2.5, value="Norepinephrine", dose_value=1.0, dose_unit="mg", priority="diuretic_vasoactive"),
        CER("action", "transfusion", 4.0, value="Packed Red Blood Cells", dose_value=350.0, dose_unit="ml", priority="transfusion"),
    ]
    return tuple(sorted(xs, key=lambda c: c.sort_key()))


def cvc_removal_stream(t_remove: float = 4.0, side: str = "L", site: str = "subclavian") -> Tuple[CER, ...]:
    xs = [
        CER("state", "line_present", None, device=Device("CVC", "none", site, side, "Multi Lumen"), priority="device_procedure"),
        CER("transition", "line_removal", t_remove, device=Device("CVC", "remove", site, side, "Multi Lumen"), priority="device_procedure", candidate=True),
        CER("action", "antibiotic", 1.0, value="Cefepime", dose_value=1.0, dose_unit="dose", priority="antibiotic"),
    ]
    return tuple(sorted(xs, key=lambda c: c.sort_key()))


def prior_film(pid: str = "P0001", t_h: float = 100.0) -> Film:
    return Film(pid, f"{pid}-S0", f"{pid}-S0-img", t_h, int(t_h // 24), int((t_h % 24) * 60), "stay1", "AP", "test", "synthetic")


ABSENT = DeviceState(False)
PRESENT = DeviceState(True)
