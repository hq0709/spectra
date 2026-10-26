"""The only path for outbound files from the MIMIC server: aggregate, non-patient-level tables.

Automatic checks (they cannot prove aggregation; the credentialed analyst's inspection is required and is
recorded in the manifest):
1. every column is on the aggregate allowlist (no identifier, time, text or evidence columns);
2. text values come from the closed vocabulary of endpoints, systems, channels, devices, edit classes,
   sources, splits, tracks, Delta bins, hypotheses and declarations, or (rule / funnel columns only) are
   short machine rule names; nothing else can carry report text;
3. the categorical columns form a unique key, so the table cannot list one row per patient, query or test;
4. at least one count column is present and every count is a non-negative integer;
5. no value looks like a MIMIC identifier.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import List

import pandas as pd

from .config import systems
from .types import DEVICE_TYPES, EDIT_CLASSES, TRACKS

COUNT_COLUMNS = {"n", "n_patients", "n_units", "patients", "priors", "queries", "units", "contrasts", "tests", "replicates"}
NUMERIC_COLUMNS = {"estimate", "se", "sd", "p_value", "p_holm", "lambda", "gpu_hours", "seconds_per_generation",
                   "value", "rung"} | COUNT_COLUMNS
TEXT_COLUMNS = {"endpoint", "system", "source", "device", "edit_class", "channel", "stratum", "funnel", "hypothesis",
                "declaration", "metric", "split", "track", "delta_bin", "rule"}
ALLOWED_COLUMNS = NUMERIC_COLUMNS | TEXT_COLUMNS
RULE_NAME = re.compile(r"^[a-z0-9_:+\-]{1,60}$")
ID_VALUE = [re.compile(r"^\d{8}$"), re.compile(r"^[0-9a-f]{8}(-[0-9a-f]{8}){4}$"), re.compile(r"^[ps]\d{8}$")]


def _vocabulary() -> set:
    v = set(systems().keys()) | set(systems().get("controls", [])) | {"copy_prior", "population_rate", "oracle", "record_volume"}
    v |= set(DEVICE_TYPES) | set(EDIT_CLASSES) | set(TRACKS) | {"cross_source", "logged", "primary", "secondary", "insertion_spec"}
    v |= {"edema", "effusion", "effusion_L", "effusion_R", "heart_size", "opacity_L", "opacity_R", "atelectasis_L", "atelectasis_R"}
    v |= {"mimic", "taix", "rexgradient", "padchest", "midrc", "brax", "chexpert_plus", "nih", "synthetic"}
    v |= {"dev_train", "dev_val", "test", "train", "val"}
    v |= {f"H{i}" for i in range(1, 10)} | {"JDA_primary", "JDA", "FA", "EA", "FA_minus_EA", "C", "AE", "AUROC", "AUROC_gain"}
    v |= {"exceeds prior+delta by more than the margin", "does not exceed prior+delta by the margin",
          "S2 beats copy-prior on conventional AUROC", "substantial specification failure", "meets the criterion",
          "not declared", "not declarable (uncertainty not estimable)"}
    v |= {"[1h,12h)", "[12h,48h)", "[2d,7d)", "[7d,30d)", "[30d,180d]"}
    v |= {f"S{i}" for i in range(1, 4)} | {"R3_system"}
    return v


class ExportRefused(ValueError):
    pass


def check(df: pd.DataFrame) -> List[str]:
    problems = []
    vocab = _vocabulary()
    for c in df.columns:
        if c not in ALLOWED_COLUMNS:
            problems.append(f"column not on the aggregate allowlist: {c}")
    text_cols = [c for c in df.columns if c in TEXT_COLUMNS]
    for c in text_cols:
        for v in df[c].dropna().astype(str):
            if c in ("rule", "funnel", "stratum", "metric"):
                ok = v in vocab or bool(RULE_NAME.match(v)) or all(RULE_NAME.match(p) or p in vocab for p in v.split("|"))
            else:
                ok = v in vocab
            if not ok:
                problems.append(f"text value outside the closed vocabulary in column {c}: {v[:40]!r}")
                break
    for c in df.columns:
        if c in NUMERIC_COLUMNS and not pd.api.types.is_numeric_dtype(df[c]):
            problems.append(f"non-numeric values in numeric column {c}")
        for v in df[c].dropna().astype(str):
            if any(p.match(v.strip()) for p in ID_VALUE) and c not in COUNT_COLUMNS:
                problems.append(f"identifier-like value in column {c}")
                break
    counts = [c for c in df.columns if c in COUNT_COLUMNS]
    if not counts:
        problems.append("no count column (n, n_patients, units, ...): aggregation cannot be checked")
    for c in counts:
        s = df[c].dropna()
        if pd.api.types.is_numeric_dtype(s) and ((s < 0).any() or (s != s.round()).any()):
            problems.append(f"count column {c} holds non-integer or negative values")
    if text_cols and df.duplicated(subset=text_cols).any():
        problems.append("duplicate categorical keys: the table looks like per-unit rows, not aggregates")
    return problems


def export(df: pd.DataFrame, out_path: str, analyst: str, purpose: str, analyst_confirmed: bool = False) -> Path:
    problems = check(df)
    if problems:
        raise ExportRefused("; ".join(problems))
    if not analyst_confirmed:
        raise ExportRefused("the credentialed analyst must confirm row granularity and content (analyst_confirmed=True)")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    manifest = {"file": out.name, "rows": int(len(df)), "columns": list(map(str, df.columns)), "purpose": purpose,
                "automatic_checks": ["column allowlist", "closed text vocabulary", "unique categorical keys",
                                     "count column present", "no identifier-like values"],
                "automatic_checks_prove_aggregation": False,
                "analyst_inspection": {"analyst": analyst, "confirmed": True},
                "written": dt.datetime.now().isoformat(timespec="seconds")}
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=1))
    return out
