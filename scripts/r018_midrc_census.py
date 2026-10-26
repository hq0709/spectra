"""R018 (NICE): MIDRC focused-study census from the public MIDRC Guppy API (data.midrc.org, no login).

Counts cases with a dated intubation / invasive ventilation record and CR/DX studies before and after the procedure
day, at day resolution (EXPERIMENT_PLAN.md, MIDRC D1 and D4: studies on the procedure day are excluded). Case-level
metadata is read in memory only; the output holds aggregate counts, no case identifiers.

    python scripts/r018_midrc_census.py --out ../runs/R018/midrc_census.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

URL = "https://data.midrc.org/guppy/graphql"
# breathing_support_type values (2026-10-21) by category; everything else (CPAP, nasal cannula, high flow, mask,
# "Assistance with Respiratory Ventilation" (ICD-10-PCS 5A09, non-invasive), NIV, ECMO, the generic "Artificial
# respiration" and "Assisted ventilation in prone position") is not counted
CATEGORIES = {
    "intubation": lambda t: t.startswith("Insertion of Endotracheal Airway into Trachea"),
    "invasive_ventilation": lambda t: (t.startswith("Insertion of Endotracheal Airway into Trachea") or t == "Invasive ventilation"
                                       or (t.startswith("Respiratory Ventilation,") and "Nonmechanical" not in t)),  # 5A19x5Z
}


def query(q: str, variables=None, tries: int = 4):
    body = json.dumps({"query": q, "variables": variables or {}}).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=180))
        except (urllib.error.URLError, TimeoutError) as e:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def pages(index: str, fields: str, flt, page: int = 1000):
    out, off = [], 0
    while True:
        r = query(f"query ($f: JSON) {{ {index}(filter: $f, accessibility: all, first: {page}, offset: {off}) {{ {fields} }} }}",
                  {"f": flt})
        if r.get("errors") or not isinstance((r.get("data") or {}).get(index), list):
            raise RuntimeError(f"Guppy {index} page at offset {off}: {str(r.get('errors') or r)[:300]}")
        rows = r["data"][index]
        out += rows
        if len(rows) < page:
            return out
        off += page


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    flt = {"nested": {"path": "procedures", "AND": [{"IN": {"procedure_name": ["Breathing Support"]}}]}}
    cases = pages("case", "submitter_id procedures { procedure_name breathing_support_type days_to_procedure_start "
                          "days_to_procedure_end }", flt)
    types = Counter(p["breathing_support_type"] for c in cases for p in (c["procedures"] or []))
    days_of = defaultdict(dict)   # category -> case -> first dated start
    for c in cases:
        for p in (c["procedures"] or []):
            t, d = p["breathing_support_type"] or "", p["days_to_procedure_start"]
            for cat, rule in CATEGORIES.items():
                if d is not None and rule(t):
                    days_of[cat][c["submitter_id"]] = min(d, days_of[cat].get(c["submitter_id"], d))
    ids = sorted(set().union(*[set(v) for v in days_of.values()]))
    study_days = defaultdict(list)
    for s in range(0, len(ids), 200):
        rows = pages("imaging_study", "case_ids days_to_study",
                     {"AND": [{"IN": {"study_modality": ["CR", "DX"]}}, {"IN": {"case_ids": ids[s:s + 200]}}]})
        for r in rows:
            cid = r["case_ids"][0] if isinstance(r["case_ids"], list) else r["case_ids"]
            if r["days_to_study"] is not None:
                study_days[cid].append(int(r["days_to_study"]))
    by_cat = {}
    for cat, proc_day in days_of.items():
        n = Counter()
        for cid, d0 in proc_day.items():
            days = study_days.get(cid, [])
            b, a = [d for d in days if d < d0], [d for d in days if d > d0]
            b7, a7 = [d for d in b if d >= d0 - 7], [d for d in a if d <= d0 + 7]
            n["dated_cases"] += 1
            n["ge2_dated_cr_dx"] += len(days) >= 2
            n["before"] += bool(b)
            n["after"] += bool(a)
            n["before_and_after"] += bool(b) and bool(a)
            n["before_and_ge2_after_days"] += bool(b) and len(set(a)) >= 2
            n["before_and_after_within_7d"] += bool(b7) and bool(a7)
        by_cat[cat] = dict(n)
    out = {"source": "MIDRC public Guppy API (data.midrc.org/guppy/graphql, accessibility all, no login)",
           "queried": time.strftime("%Y-%m-%d"),
           "cases_with_breathing_support": len(cases),
           "breathing_support_types": dict(types.most_common()),
           "anchor": "earliest qualifying dated start per case and category",
           "categories": {"intubation": "Insertion of Endotracheal Airway into Trachea, via natural or artificial opening "
                                        "(with or without endoscopy)",
                          "invasive_ventilation": "intubation, 'Invasive ventilation', or ICD-10-PCS Respiratory Ventilation "
                                                  "<24 h / 24-96 h / >96 h (5A19x5Z)"},
           "counts_by_category": by_cat,
           "note": "day resolution; studies on the procedure day excluded; aggregate counts only"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
