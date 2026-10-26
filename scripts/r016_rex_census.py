"""R016, metadata part: ReXGradient-160K census of the test pool (val + public test patients) from the released
metadata alone (no images; DUA 3.A.5), written as aggregate counts to --out (JSON).

1. Films (X2): one frontal film per study (view position AP / PA / POSTERO_ANTERIOR on any image of the study);
   studies whose views are all unknown are counted apart; patient-days with two frontal studies are excluded.
2. Upper bound of the E1 test pool (X6): an E1 set needs a prior and two later frontal films at distinct days
   1-180 d after it among its next max_later_films films (constants.yaml e1), so a patient enters only with
   >= 3 eligible films.
3. Comparison dates (X5): how many calendar dates cited in the reports' Comparison sections (m/d/yyyy, m/d/yy,
   m-d-yy(yy), "Month d(th), yyyy", "d(th) Month yyyy"; month-only dates are not days) equal a StudyDate of the same patient,
   which is what the resolver's explicit-date rule (R1) needs. A second diagnostic asks whether one shift per patient
   maps both cited dates of the second and third reports of a three-study patient onto that patient's released
   earlier studies; it assumes the cited exams are released ones (a report may cite an exam outside the release), so
   a no_match does not show that the patient's dates lack a consistent shift.

    python scripts/r016_rex_census.py --metadata <snapshot>/metadata --out ../runs/R016/rex_census.json
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FRONTAL = {"AP", "PA", "POSTERO_ANTERIOR"}
NON_FRONTAL = {"LATERAL", "LL", "LAT", "RAO", "LAO", "RPO", "LPO"}
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MON = (r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?")                     # whole month words only
NUMERIC = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2})\b")         # m/d/yyyy, m/d/yy, m-d-yy(yy)
MONTH_DAY_YEAR = re.compile(_MON + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I)   # July 6, 1993; July 17th 2004
DAY_MONTH_YEAR = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + _MON + r",?\s+(\d{4})\b", re.I)


def _year(y: str) -> int:
    return int(y) if len(y) == 4 else (2000 + int(y) if int(y) <= 30 else 1900 + int(y))


def cited_dates(text: str):
    """Valid calendar dates written in a Comparison section, in order of appearance."""
    found = []
    for m in NUMERIC.finditer(text or ""):
        found.append((m.start(), _year(m.group(3)), int(m.group(1)), int(m.group(2))))
    for m in MONTH_DAY_YEAR.finditer(text or ""):
        found.append((m.start(), int(m.group(3)), MONTHS[m.group(1).lower()[:3]], int(m.group(2))))
    for m in DAY_MONTH_YEAR.finditer(text or ""):
        found.append((m.start(), int(m.group(3)), MONTHS[m.group(2).lower()[:3]], int(m.group(1))))
    out = []
    for _, y, mo, d in sorted(found):
        try:
            out.append(dt.date(y, mo, d))
        except ValueError:
            pass
    return out


def main(argv=None) -> int:
    from spectra.config import constants

    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    C = constants()["e1"]
    lo, hi, k = C["horizon"]["rexgradient"]["min_days"], C["horizon"]["rexgradient"]["max_days"], C["max_later_films"]
    studies = []
    for split in ("valid", "test"):
        for e in json.loads((Path(args.metadata) / f"{split}_metadata_view_position.json").read_text()).values():
            views = {str(v).upper() for v in (e.get("ImageViewPosition") or [])}
            kind = "frontal" if views & FRONTAL else "non_frontal" if views and views <= NON_FRONTAL else "unknown"
            studies.append({"split": split, "pid": e["PatientID"], "kind": kind, "comp": e.get("Comparison") or "",
                            "day": dt.datetime.strptime(str(e["StudyDate"]), "%Y%m%d").date()})
    res = {"source": "ReXGradient-160K metadata (val + public test)", "window_days": [lo, hi],
           "studies": len(studies), "patients": len({s["pid"] for s in studies}),
           "studies_by_view": dict(collections.Counter(s["kind"] for s in studies))}
    by_pid = collections.defaultdict(list)
    for s in studies:
        by_pid[s["pid"]].append(s)
    res["patients_by_studies"] = dict(sorted(collections.Counter(len(v) for v in by_pid.values()).items()))
    # films: frontal studies, one per patient-day (a day with two frontal studies is excluded)
    films, same_day = {}, 0
    for pid, ss in by_pid.items():
        days = collections.Counter(s["day"] for s in ss if s["kind"] == "frontal")
        same_day += sum(n for n in days.values() if n > 1)
        films[pid] = sorted(d for d, n in days.items() if n == 1)
    res["frontal_films"] = sum(len(v) for v in films.values())
    res["frontal_studies_on_shared_days_excluded"] = same_day

    def e1_eligible(days):  # a prior with two of its next k films at distinct days within [lo, hi] days of it
        return any(sum(lo <= (b - a).days <= hi for b in days[i + 1:i + 1 + k]) >= 2 for i, a in enumerate(days))

    res["patients_with_ge3_frontal_films"] = sum(len(v) >= 3 for v in films.values())
    res["patients_e1_upper_bound"] = sum(e1_eligible(v) for v in films.values())
    gaps = [(b - a).days for v in films.values() for a, b in zip(v, v[1:])]
    res["consecutive_film_pairs"] = len(gaps)
    res["consecutive_pairs_in_window"] = sum(lo <= g <= hi for g in gaps)
    # comparison dates: StudyDate frame or not
    cites = equal = 0
    for ss in by_pid.values():
        days = {s["day"] for s in ss}
        for s in ss:
            for c in cited_dates(s["comp"]):
                cites += 1
                equal += c in days
    shift = collections.Counter()
    for ss in by_pid.values():
        ss = sorted(ss, key=lambda s: s["day"])
        if len(ss) != 3 or len({s["day"] for s in ss}) != 3:
            continue
        c2, c3 = cited_dates(ss[1]["comp"]), cited_dates(ss[2]["comp"])
        if len(c2) != 1 or len(c3) != 1:
            continue
        s = c2[0] - ss[0]["day"]                      # if report 2 cites a released exam, it is study 1
        hit = c3[0] - s
        shift["match" if hit in (ss[0]["day"], ss[1]["day"]) else "no_match"] += 1
    res["comparison_dates"] = {"cited": cites, "equal_to_a_studydate_of_the_patient": equal,
                               "shift_fit_assuming_cited_targets_are_released": dict(shift)}
    res["confirmatory_threshold_patients"] = 2500
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
