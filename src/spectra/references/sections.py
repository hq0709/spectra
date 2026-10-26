"""Report sections (M4): the extractor reads the comparison, findings and impression sections of a report.

Follows the MIT-LCP MIMIC-CXR section parser: headings are upper-case words ending in a colon at the start of a
line; heading variants map to one name; a report without findings or impression contributes its last paragraph
as the impression. Whitespace inside a section collapses to single spaces. The extraction text is rebuilt from the
kept sections in a fixed order, and evidence spans are checked against that text.
"""
from __future__ import annotations

import re
from typing import Dict, Mapping

HEADING = re.compile(r"^[ \t]*([A-Z][A-Z /&(),-]{2,60}):", re.M)
NAMES = {
    "comparison": ("COMPARISON", "COMPARISONS", "COMPARISON STUDY", "COMPARISON STUDIES", "COMPARISON EXAM",
                   "PRIOR STUDY", "PRIOR STUDIES", "PRIOR"),
    "findings": ("FINDINGS", "FINDING", "REPORT", "FINDINGS AND IMPRESSION", "FINDINGS/IMPRESSION"),
    "impression": ("IMPRESSION", "IMPRESSIONS", "CONCLUSION", "CONCLUSIONS", "SUMMARY", "OPINION"),
}
KEPT = ("comparison", "findings", "impression")
_ALIAS = {v: k for k, vs in NAMES.items() for v in vs}


def _clean(s: str) -> str:
    """One line per section: MIMIC-CXR reports wrap lines at a fixed width mid-sentence, and an evidence span
    copied by the extractor must not depend on those breaks."""
    return re.sub(r"\s+", " ", s).strip()


def sections(report: str) -> Dict[str, str]:
    """Kept sections of one report text (empty sections omitted); unknown headings close the previous section."""
    marks = [(m.start(), m.end(), m.group(1).strip()) for m in HEADING.finditer(report)]
    out: Dict[str, str] = {}
    for i, (s, e, name) in enumerate(marks):
        key = _ALIAS.get(re.sub(r"\s+", " ", name))
        if key is None:
            continue
        body = _clean(report[e:marks[i + 1][0] if i + 1 < len(marks) else len(report)])
        if body:
            out[key] = (out[key] + "\n" + body) if key in out else body
    if "findings" not in out and "impression" not in out:
        paras = [p for p in re.split(r"\n\s*\n", report.strip()) if p.strip()]
        if paras:
            last = paras[-1]
            m = list(HEADING.finditer(last))
            out["impression"] = _clean(last[m[-1].end():] if m else last)
    return out


def extraction_text(secs: Mapping[str, str]) -> str:
    """COMPARISON / FINDINGS / IMPRESSION blocks in this order; the text the extractor and its evidence check see."""
    return "\n".join(f"{k.upper()}: {secs[k]}" for k in KEPT if secs.get(k))
