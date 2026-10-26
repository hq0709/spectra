"""R010 (MIMIC server): film table, ICU stays and splits (M1-M3) and the E1 eligibility funnel up to the rules
that need no report extraction (EXPERIMENT_PLAN.md, MIMIC E1 funnel rows 0-4, by split), plus the E2 row-1 count of
ICU frontal priors in the test split. Rows 5-6 (admitted orderings) follow the extraction (R011) and the resolver.

Writes counts only (aggregate JSON, no identifiers); the film table stays on the MIMIC server under --work.

    python scripts/r010_mimic_census.py --config config/mimic.yaml --out ../runs/R010/census.json --work ../work/mimic
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def e1_funnel(films, min_h: float, max_h: float, max_later: int) -> dict:
    """Rows 1-4 of the MIMIC E1 funnel, overall and by split, from Film records (one frontal film per study)."""
    from spectra.e1 import query_sets

    def rows(fs):
        icu = [f for f in fs if f.stay_id is not None]
        by_stay = defaultdict(list)
        for f in icu:
            by_stay[f.stay_id].append(f)
        ge3 = [s for s, v in by_stay.items() if len(v) >= 3]
        ge3_pat = {by_stay[s][0].patient_id for s in ge3}
        sets = [qs for qs in query_sets(icu, min_h, max_h, max_later, same_stay=True) if len(qs.later) >= 2]
        return {"1_frontal_f1": {"patients": len({f.patient_id for f in fs}), "studies": len(fs)},
                "2_in_icu_stay": {"patients": len({f.patient_id for f in icu}), "studies": len(icu),
                                  "stays": len(by_stay)},
                "3_ge3_frontal_in_one_stay": {"patients": len(ge3_pat), "stays": len(ge3),
                                              "studies": sum(len(by_stay[s]) for s in ge3)},
                "4_prior_with_ge2_next_films": {"patients": len({qs.prior.patient_id for qs in sets}),
                                                "stays": len({qs.prior.stay_id for qs in sets}),
                                                "eligible_priors": len(sets),
                                                "queries": sum(len(qs.later) for qs in sets)}}

    out = {"all": rows(films)}
    for split in sorted({f.split for f in films}):
        out[split] = rows([f for f in films if f.split == split])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="aggregate JSON (counts only)")
    ap.add_argument("--work", required=True, help="MIMIC-server directory for the patient-level film table")
    args = ap.parse_args(argv)

    import pandas as pd

    from spectra.adapters.mimic import MimicAdapter
    from spectra.config import constants, source_config

    cfg = source_config(args.config)
    E = constants()["e1"]
    hz = E["horizon"]["mimic"]
    t0 = time.time()
    a = MimicAdapter(cfg["cxr_jpg_dir"], cfg["cxr_reports_dir"], cfg["mimiciv_dir"], salt=cfg["split_salt"],
                     fractions=cfg["split_fractions"])
    meta = a._read("mimic-cxr-2.0.0-metadata")
    films = a.films()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dataclasses.asdict(f) for f in films]).to_parquet(work / "films.parquet")
    split_patients = Counter()
    for pid, sp in {f.patient_id: f.split for f in films}.items():
        split_patients[sp] += 1
    res = {"config": {"split_salt": cfg["split_salt"], "split_fractions": cfg["split_fractions"], "horizon": hz,
                      "max_later_films": E["max_later_films"]},
           "0_release": {"patients": int(meta["subject_id"].nunique()), "studies": int(meta["study_id"].nunique()),
                         "images": int(len(meta)),
                         "frontal_images": int(meta["ViewPosition"].isin(["AP", "PA"]).sum())},
           "patients_by_split": dict(split_patients),
           "forced_test": {"p19_patients": len({f.patient_id for f in films if f.patient_id.startswith("19")})},
           "e1": e1_funnel(films, hz["min_h"], hz["max_h"], E["max_later_films"]),
           "e2_row1_icu_frontal_priors_test": sum(1 for f in films if f.stay_id is not None and f.split == "test"),
           "seconds": None}
    res["seconds"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["e1"]["all"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
