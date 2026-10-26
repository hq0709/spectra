"""MIMIC adapter: MIMIC-CXR-JPG 2.1.0 films + MIMIC-CXR 2.1.0 reports + MIMIC-IV 3.1 ICU tables.
Runs on the MIMIC server only; nothing patient-level it produces leaves that machine.

Film rule F1 (one image per study): ViewPosition in {AP, PA}; prefer the view matching the setting (AP for
ICU portable studies, PA otherwise); then the larger pixel count; then the smallest dicom_id. Acquisition
time = StudyDate + StudyTime (intervals and time of day unmodified; same date shift as MIMIC-IV).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional

import pandas as pd

from ..cer import CERCompiler, IcuTables, MimicOntology
from ..splits import assign, mimic_forced_test
from ..types import Film
from .base import SourceAdapter

EPOCH = pd.Timestamp("1970-01-01")


def study_time(study_date, study_time) -> pd.Timestamp:
    d = str(int(study_date))
    t = float(study_time)
    hh, rem = divmod(int(t), 10000)
    mm, ss = divmod(rem, 100)
    frac = t - int(t)
    return pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:8]}") + pd.Timedelta(hours=hh, minutes=mm, seconds=ss + frac)


def apply_rule_f1(meta: pd.DataFrame, icu_study_ids: Optional[set] = None) -> pd.DataFrame:
    """One frontal image per study by rule F1. `meta` has the MIMIC-CXR-JPG metadata columns."""
    m = meta[meta["ViewPosition"].isin(["AP", "PA"])].copy()
    icu = icu_study_ids or set()
    m["_setting_match"] = [(v == "AP") if s in icu else (v == "PA") for s, v in zip(m["study_id"], m["ViewPosition"])]
    m["_pixels"] = m["Rows"].astype(float) * m["Columns"].astype(float)
    m = m.sort_values(["study_id", "_setting_match", "_pixels", "dicom_id"], ascending=[True, False, False, True])
    return m.drop_duplicates("study_id", keep="first").drop(columns=["_setting_match", "_pixels"])


class MimicAdapter(SourceAdapter):
    name = "mimic"
    time_resolution = "minute"
    has_reports = True
    has_actions = True
    has_observations = True

    def __init__(self, cxr_jpg_dir: str, cxr_reports_dir: str, mimiciv_dir: str, salt: str = "spectra-mimic-v1",
                 fractions: Optional[Mapping[str, float]] = None, icu_tables: Optional[IcuTables] = None):
        self.cxr_jpg_dir = Path(cxr_jpg_dir)
        self.cxr_reports_dir = Path(cxr_reports_dir)
        self.mimiciv_dir = Path(mimiciv_dir)
        self.salt = salt
        self.fractions = dict(fractions or {"dev_train": 0.6, "dev_val": 0.1, "test": 0.3})
        self._icu = icu_tables
        self._films: Optional[List[Film]] = None
        self.ontology = MimicOntology()
        self.compiler = CERCompiler(self.ontology)

    # (i) films ---------------------------------------------------------------------------
    def _read(self, name: str) -> pd.DataFrame:
        for cand in (self.cxr_jpg_dir / f"{name}.csv.gz", self.cxr_jpg_dir / f"{name}.csv"):
            if cand.exists():
                return pd.read_csv(cand)
        raise FileNotFoundError(name)

    def icustays(self) -> pd.DataFrame:
        p = self.mimiciv_dir / "icu" / "icustays.csv.gz"
        df = pd.read_csv(p, parse_dates=["intime", "outtime"])
        return df

    def films(self) -> List[Film]:
        if self._films is not None:
            return self._films
        meta = self._read("mimic-cxr-2.0.0-metadata")
        meta["t"] = [study_time(d, t) for d, t in zip(meta["StudyDate"], meta["StudyTime"])]
        stays = self.icustays()
        st = meta[["subject_id", "study_id", "t"]].drop_duplicates("study_id").merge(
            stays[["subject_id", "stay_id", "intime", "outtime"]], on="subject_id", how="left")
        st = st[(st["t"] >= st["intime"]) & (st["t"] <= st["outtime"])]
        stay_of = dict(zip(st["study_id"], st["stay_id"]))
        films = apply_rule_f1(meta, set(stay_of))
        split_tab = self._read("mimic-cxr-2.0.0-split")
        valtest = set(split_tab.loc[split_tab["split"].isin(["validate", "test"]), "subject_id"].astype(str))
        subjects = sorted(films["subject_id"].astype(str).unique())
        splits = assign(subjects, self.salt, self.fractions, forced_test=mimic_forced_test(subjects, valtest))
        out = []
        for r in films.itertuples(index=False):
            t = r.t
            sid = stay_of.get(r.study_id)
            out.append(Film(str(r.subject_id), str(r.study_id), str(r.dicom_id), (t - EPOCH).total_seconds() / 3600.0,
                            (t.normalize() - EPOCH).days, t.hour * 60 + t.minute, None if sid is None else str(int(sid)),
                            r.ViewPosition, splits[str(r.subject_id)], "mimic"))
        self._films = out
        return out

    # (ii) reports ------------------------------------------------------------------------
    def report_path(self, subject_id: str, study_id: str) -> Path:
        s = str(subject_id)
        return self.cxr_reports_dir / "files" / f"p{s[:2]}" / f"p{s}" / f"s{study_id}.txt"

    def reports(self) -> Mapping[str, str]:
        out = {}
        for f in self.films():
            p = self.report_path(f.patient_id, f.study_id)
            if p.exists():
                out[f.study_id] = p.read_text()
        return out

    # (iii)/(iv) action log ----------------------------------------------------------------
    def icu(self, track: str = "A", stay_ids=None, loader: str = "duckdb") -> IcuTables:
        """Tables for the cohort's stays and the requested track only (bounded reads)."""
        key = (track, None if stay_ids is None else tuple(sorted(int(s) for s in stay_ids)))
        if not hasattr(self, "_icu_cache"):
            self._icu_cache = {}
        if key not in self._icu_cache:
            if self._icu is not None:
                self._icu_cache[key] = self._icu
            else:
                stays = stay_ids if stay_ids is not None else sorted({int(f.stay_id) for f in self.films() if f.stay_id})
                if loader == "duckdb":
                    self._icu_cache[key] = IcuTables.from_duckdb(self.mimiciv_dir / "icu", self.mimiciv_dir / "hosp",
                                                                 stays, self.ontology, track_b=track == "B")
                else:
                    self._icu_cache[key] = IcuTables.from_dir(self.mimiciv_dir / "icu", self.mimiciv_dir / "hosp",
                                                              stays, self.ontology, track_b=track == "B")
        return self._icu_cache[key]

    def window_stream(self, prior: Film, target: Film, track: str = "A", cutoff: str = "occurrence", stay_ids=None):
        if prior.stay_id is None or prior.stay_id != target.stay_id:
            raise ValueError("MIMIC streams are compiled within one ICU stay")
        t0 = EPOCH + pd.Timedelta(hours=prior.t_h)
        tk = EPOCH + pd.Timedelta(hours=target.t_h)
        return self.compiler.compile(self.icu(track, stay_ids), int(prior.stay_id), t0, tk, track=track, cutoff=cutoff)
