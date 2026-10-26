"""Comparison-link resolver R1-R5 and ordering tuples (FINAL_PROPOSAL.md, Core Mechanism section 6).

R1 explicit reference (relative day / clock time / date) must match exactly one earlier frontal chest film
   of the patient (calendar-day offset exact; clock time within 30 min of StudyTime); else unresolved.
R2 default reference (none / redacted): the immediately preceding frontal chest film.
R3 consistency (default links only): every change statement of report(y) must agree with the states
   stated in report(y'); else unresolved.
R4 multiple priors or another modality: unresolved.
R5 no other frontal film of the patient within 15 min of the resolved film; else unresolved.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..types import Film, Ordering

CLOCK_TOL_MIN = 30
R5_TOL_MIN = 15
PRESENT_ABS = {"questionable", "mild", "moderate", "severe", "present_unspecified"}
DIRECTION = {"new": +1, "worsened": +1, "improved": -1, "resolved": -1, "unchanged": 0}
GRADE = {"absent": 0, "questionable": 1, "mild": 2, "moderate": 3, "severe": 4}


@dataclass(frozen=True)
class Resolution:
    film: Optional[Film]
    provenance: Optional[str]  # comparison_explicit | comparison_default
    rule: str                  # resolved | R1 | R3 | R4 | R5 | no_prior


def _clock(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _day_of(date_str: str, epoch: dt.date = dt.date(1970, 1, 1)) -> int:
    return (dt.date.fromisoformat(date_str) - epoch).days


def resolve(y: Film, films: Sequence[Film], ref: Mapping) -> Resolution:
    earlier = sorted((f for f in films if f.patient_id == y.patient_id and f.t_h < y.t_h and f.study_id != y.study_id),
                     key=lambda f: f.t_h)
    if not earlier:
        return Resolution(None, None, "no_prior")
    rtype = ref.get("type", "none")
    if rtype in ("multiple", "other_modality"):
        return Resolution(None, None, "R4")
    if rtype in ("relative_day", "clock_time", "date"):
        needed = {"relative_day": "day_offset", "clock_time": "clock_time", "date": "date"}[rtype]
        if ref.get(needed) in (None, ""):
            return Resolution(None, None, "R1")  # an explicit reference without its value cannot match a film
        cands = list(earlier)
        if ref.get("day_offset") is not None:
            cands = [f for f in cands if f.day == y.day + int(ref["day_offset"])]
        if ref.get("date"):
            cands = [f for f in cands if f.day == _day_of(ref["date"])]
        if ref.get("clock_time"):
            c = _clock(ref["clock_time"])
            cands = [f for f in cands if f.clock_min is not None and abs(f.clock_min - c) <= CLOCK_TOL_MIN]
        if len(cands) != 1:
            return Resolution(None, None, "R1")
        film, prov = cands[0], "comparison_explicit"
    elif rtype in ("none", "redacted"):
        film, prov = earlier[-1], "comparison_default"
    else:
        return Resolution(None, None, "R4")
    if any(f.study_id != film.study_id and abs(f.t_h - film.t_h) * 60 <= R5_TOL_MIN for f in films
           if f.patient_id == y.patient_id):
        return Resolution(None, None, "R5")
    return Resolution(film, prov, "resolved")


def _mentioned(node: Mapping) -> bool:
    return node.get("change", "not_mentioned") not in (None, "not_mentioned") or \
        node.get("absolute", "not_mentioned") not in (None, "not_mentioned")


def _attributed(findings: Mapping) -> Dict[str, Dict[str, Optional[str]]]:
    """Change, absolute state and evidence per scored channel after the unsided-effusion rule, shared by the
    orderings and by R3. An unsided statement goes to a side only when the report names exactly one effusion
    side (in any statement, change or absolute); bilateral statements are extracted as both sides by the
    labeler; a sided change that disagrees with the unsided one leaves that side's change unresolved."""
    node = lambda k: findings.get(k, {}) or {}
    out = {k: {"change": node(k).get("change"), "absolute": node(k).get("absolute"), "evidence": node(k).get("evidence", "")}
           for k in ("edema", "effusion_L", "effusion_R")}
    U = node("effusion_unsided")
    named = [k for k in ("effusion_L", "effusion_R") if _mentioned(node(k))]
    if len(named) == 1:
        cur = out[named[0]]
        uc, ua = U.get("change"), U.get("absolute")
        if uc not in (None, "not_mentioned"):
            if cur["change"] in (None, "not_mentioned", "uncertain"):
                cur["change"], cur["evidence"] = uc, U.get("evidence", "")
            elif cur["change"] != uc:
                cur["change"], cur["evidence"] = None, ""   # the sided and unsided statements disagree
        if ua not in (None, "not_mentioned") and cur["absolute"] in (None, "not_mentioned"):
            cur["absolute"] = ua
    return out


def consistent(report_y: Mapping, report_prev: Mapping) -> bool:
    """R3: change statements of report(y), after side attribution, against the states stated in report(y')."""
    fy, fp = report_y.get("findings", {}), report_prev.get("findings", {})
    ay, ap = _attributed(fy), _attributed(fp)
    pairs = [(ay[ch]["change"], ap[ch]["absolute"]) for ch in ay]
    pairs.append(((fy.get("effusion_unsided", {}) or {}).get("change"),
                  (fp.get("effusion_unsided", {}) or {}).get("absolute", "not_mentioned")))
    for ch_lab, prev_abs in pairs:
        prev_abs = prev_abs or "not_mentioned"
        if ch_lab == "new" and prev_abs in PRESENT_ABS:
            return False
        if ch_lab in ("resolved", "improved") and prev_abs == "absent":
            return False
    dy, dp = report_y.get("devices", {}), report_prev.get("devices", {})
    for d, node in dy.items():
        prev = dp.get(d, {}).get("state", "not_mentioned")
        if node.get("state") == "placed_interval" and prev in ("present", "placed_interval", "repositioned"):
            return False
        if node.get("state") == "removed_interval" and prev in ("absent", "removed_interval"):
            return False
    return True


def _sided(findings: Mapping) -> Dict[str, Tuple[Optional[str], str]]:
    """Effusion change per side channel with its evidence (see _attributed)."""
    a = _attributed(findings)
    return {k: (a[k]["change"], a[k]["evidence"]) for k in ("effusion_L", "effusion_R")}


def orderings_from_link(prev: Film, y: Film, report_y: Mapping, provenance: str) -> List[Ordering]:
    f = report_y.get("findings", {})
    labels = {"edema": (f.get("edema", {}).get("change"), f.get("edema", {}).get("evidence", ""))}
    labels.update(_sided(f))
    return [Ordering(prev.study_id, y.study_id, ch, DIRECTION[lab], provenance, ev)
            for ch, (lab, ev) in labels.items() if lab in DIRECTION]


def resolve_links(films: Sequence[Film], extractions: Mapping[str, Mapping]) -> Tuple[List[Ordering], Dict[str, int]]:
    """Resolve every report's comparison reference (R1, R2, R4, R5), apply R3 to default links, and return
    the ordering tuples with a per-rule count (attrition)."""
    by_patient: Dict[str, List[Film]] = {}
    for f in films:
        by_patient.setdefault(f.patient_id, []).append(f)
    out: List[Ordering] = []
    counts: Dict[str, int] = {}
    for pid, fs in by_patient.items():
        for y in fs:
            ext = extractions.get(y.study_id)
            if ext is None:
                continue
            r = resolve(y, fs, ext.get("comparison_reference", {"type": "none"}))
            rule = r.rule
            if r.film is not None and r.provenance == "comparison_default":
                prev_ext = extractions.get(r.film.study_id)
                if prev_ext is None or not consistent(ext, prev_ext):
                    rule = "R3"
            counts[rule] = counts.get(rule, 0) + 1
            if rule == "resolved":
                out += orderings_from_link(r.film, y, ext, r.provenance)
    return out, counts


def orderings_from_grades(films: Sequence[Film], grades: Mapping[str, Mapping[str, Optional[int]]],
                          channels: Sequence[str]) -> Dict[Tuple[str, str, str], Ordering]:
    """Pairs whose two validated absolute grades differ (TAIX-Ray radiologist grades; validated report
    grades on MIMIC). Not-assessable grades (None) give no ordering."""
    out = {}
    fs = sorted(films, key=lambda f: f.t_h)
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            a, b = fs[i], fs[j]
            if a.patient_id != b.patient_id:
                continue
            for c in channels:
                ga, gb = grades.get(a.study_id, {}).get(c), grades.get(b.study_id, {}).get(c)
                if ga is None or gb is None:
                    continue
                d = (gb > ga) - (gb < ga)
                out[(a.study_id, b.study_id, c)] = Ordering(a.study_id, b.study_id, c, d, "absolute_grades")
    return out


def merge_orderings(link_orderings: Sequence[Ordering],
                    grade_orderings: Mapping[Tuple[str, str, str], Ordering]) -> Dict[Tuple[str, str, str], Ordering]:
    """Direct resolved links plus differing absolute grades; a comparison tuple that conflicts with the
    absolute grades of the same pair leaves the pair unresolved."""
    out: Dict[Tuple[str, str, str], Ordering] = {}
    conflicted = set()
    for k, o in grade_orderings.items():
        out[k] = o
    for o in link_orderings:
        k = (o.earlier_study, o.later_study, o.channel)
        g = grade_orderings.get(k)
        if g is not None and g.direction != o.direction:
            conflicted.add(k)
            continue
        if k not in out:
            out[k] = o
    for k in conflicted:
        out.pop(k, None)
    return out
