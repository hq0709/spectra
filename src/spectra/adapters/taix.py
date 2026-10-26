"""TAIX-Ray adapter (authoritative E1 rules; promoted from refine-logs/feasibility/taix_adapter.py).

- splits: TAIX-Ray's official patient-level splits (train / val / test); no hash split for this source.
- timing: Age (days at acquisition) only; StudyDate is not used (its offset from Age varies within
  31,085 of 47,724 patients in the release).
- one film per patient-day (same-day order is unknown): keep the smallest UID of that day.
- query set of a prior: the next <= 3 consecutive patient-days, all within 1 <= dAge <= 30 days, stopping
  at the first film outside the horizon; a contrast needs two films of the set.
- references: radiologist grades; heart size 0-3, other items 0-4; a negative code (not assessable) -> unknown.
- primary channels: PulmonaryCongestion (edema channel), PleuralEffusion_Left, PleuralEffusion_Right.
- prior selection: among priors with >= 1 admitted primary contrast, <= 3 per patient, by sha256 order.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Optional

import pandas as pd

from ..e1 import admitted_contrasts, query_sets, sample_priors
from ..references.resolver import orderings_from_grades
from ..types import Film
from .base import SourceAdapter

CHANNEL_COLUMNS = {"edema": "PulmonaryCongestion", "effusion_L": "PleuralEffusion_Left",
                   "effusion_R": "PleuralEffusion_Right"}
SECONDARY_COLUMNS = {"heart_size": "HeartSize", "opacity_L": "PulmonaryOpacities_Left", "opacity_R": "PulmonaryOpacities_Right",
                     "atelectasis_L": "Atelectasis_Left", "atelectasis_R": "Atelectasis_Right"}
SEED = "spectra-taix-v1"
H_MIN_D, H_MAX_D, MAX_LATER, MAX_PRIORS = 1, 30, 3, 3


class TaixAdapter(SourceAdapter):
    name = "taix"
    time_resolution = "day"
    has_grades = True

    def __init__(self, metadata_parquet: str):
        self.df = pd.read_parquet(metadata_parquet)
        self._films: Optional[List[Film]] = None

    def films(self) -> List[Film]:
        if self._films is None:
            d = self.df.sort_values(["PatientID", "Age", "UID"]).drop_duplicates(["PatientID", "Age"], keep="first")
            self._films = [Film(r.PatientID, r.UID, r.UID, float(r.Age) * 24.0, int(r.Age), None, None, "AP", r.Split, "taix")
                           for r in d.itertuples(index=False)]
        return self._films

    def grades(self) -> Mapping[str, Mapping[str, Optional[int]]]:
        cols = {**CHANNEL_COLUMNS, **SECONDARY_COLUMNS}
        out = {}
        for r in self.df[["UID"] + list(cols.values())].itertuples(index=False):
            vals = r._asdict()
            out[vals["UID"]] = {c: (None if pd.isna(vals[col]) or int(vals[col]) < 0 else int(vals[col])) for c, col in cols.items()}
        return out

    def e1_sets(self, split: str = "test", channels=None):
        """Query sets and admitted contrasts of one split, with the authoritative prior sampling. channels: the
        channels scored on this source (default: all primary channels); priors are sampled among those with an
        admitted contrast in these channels (2026-10-21: edema is unscored on TAIX-Ray after E-b and E-c)."""
        channels = list(channels or CHANNEL_COLUMNS)
        films = [f for f in self.films() if f.split == split]
        grades = self.grades()
        by_pat = defaultdict(list)
        for f in films:
            by_pat[f.patient_id].append(f)
        index_of = {}
        for pid, fs in by_pat.items():
            for i, f in enumerate(sorted(fs, key=lambda f: (f.t_h, f.study_id))):
                index_of[f.study_id] = i
        sets = query_sets(films, H_MIN_D * 24.0, H_MAX_D * 24.0, MAX_LATER, same_stay=False)
        sets = [qs for qs in sets if len(qs.later) >= 2]
        orderings = {}
        for pid, fs in by_pat.items():
            orderings.update(orderings_from_grades(fs, grades, channels))
        with_c = [(qs, admitted_contrasts(qs, orderings, channels)) for qs in sets]
        chosen = sample_priors(with_c, MAX_PRIORS, SEED, key=lambda qs: index_of[qs.prior.study_id])
        return sets, with_c, chosen

    def funnel(self, split: str = "test", channels=None) -> Dict[str, object]:
        films = [f for f in self.films() if f.split == split]
        raw = self.df[self.df["Split"] == split]
        sets, with_c, chosen = self.e1_sets(split, channels)
        n_days = defaultdict(int)
        for f in films:
            n_days[f.patient_id] += 1
        pairs = [o for qs, cons in with_c for o in cons]
        by_channel = defaultdict(set)
        n_contr = defaultdict(int)
        for qs, cons in chosen:
            for c in cons:
                by_channel[c.channel].add(qs.prior.patient_id)
                n_contr[c.channel] += 1
        return {
            "films": int(len(raw)), "patient_days": len(films), "patients": int(raw["PatientID"].nunique()),
            "same_day_films_dropped": int(len(raw) - len(films)),
            "patients_ge3_patient_days": sum(1 for v in n_days.values() if v >= 3),
            "patients_with_eligible_prior": len({qs.prior.patient_id for qs in sets}),
            "eligible_priors": len(sets),
            "selected_priors": len(chosen),
            "patients_with_admitted_contrast": len({qs.prior.patient_id for qs, _ in chosen}),
            "admitted_contrasts": dict(n_contr),
            "patients_with_admitted_contrast_by_channel": {c: len(s) for c, s in by_channel.items()},
            "admitted_pairs_all_priors": len(pairs),
        }
