"""Patient-disjoint splits: salted hash of the patient identifier, unless the source ships official
patient-level splits; forced-test patients (MIMIC: p19 RoentGen-v2 holdout and the official
validate/test patients) are always test."""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Mapping, Optional


def salted_fraction(patient_id: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}|{patient_id}".encode()).hexdigest()
    return int(h[:15], 16) / float(16 ** 15)


def assign(patient_ids: Iterable[str], salt: str, fractions: Mapping[str, float],
           forced_test: Optional[set] = None, official: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """fractions: ordered split -> share among non-forced patients, e.g. {"dev_train": .6, "dev_val": .1, "test": .3}."""
    forced_test = forced_test or set()
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    out = {}
    for pid in patient_ids:
        if official is not None:
            out[pid] = official[pid]
            continue
        if pid in forced_test:
            out[pid] = "test"
            continue
        u, acc = salted_fraction(pid, salt), 0.0
        for name, frac in fractions.items():
            acc += frac
            if u < acc:
                out[pid] = name
                break
        else:
            out[pid] = list(fractions)[-1]
    return out


def mimic_forced_test(subject_ids: Iterable[str], official_valtest: Iterable[str]) -> set:
    """p19 subjects (RoentGen-v2 holdout; MIMIC-CXR folder p19 = subject_id starting with 19) and the
    MIMIC-CXR official validate/test subjects."""
    forced = {s for s in subject_ids if str(s).startswith("19")}
    return forced | {str(s) for s in official_valtest}
