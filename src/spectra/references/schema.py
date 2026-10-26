"""Extraction schema for the local report labeler (Qwen3-32B, temperature 0, JSON-schema-guided decoding).

Per channel the extractor returns: change label, absolute state, the stated comparison reference and a
verbatim evidence span. Outputs whose span is not a substring of the report are rejected.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Dict, List, Optional

import jsonschema  # required: schema validation is never skipped

CHANNELS = ["edema", "effusion_L", "effusion_R", "effusion_unsided"]
DEVICES = ["ETT", "CVC", "PICC", "enteric", "chest_tube", "tracheostomy"]
CHANGE = ["new", "worsened", "improved", "unchanged", "resolved", "not_mentioned", "uncertain"]
ABSOLUTE = ["absent", "questionable", "mild", "moderate", "severe", "present_unspecified", "not_mentioned"]
REFERENCE_TYPES = ["relative_day", "clock_time", "date", "redacted", "multiple", "other_modality", "none"]
DEVICE_STATE = ["present", "absent", "placed_interval", "removed_interval", "repositioned", "not_mentioned"]

_REF = {
    "type": "object",
    "properties": {
        "type": {"enum": REFERENCE_TYPES},
        "day_offset": {"type": ["integer", "null"]},   # relative day, e.g. -1 for "yesterday"
        "clock_time": {"type": ["string", "null"], "pattern": "^([01][0-9]|2[0-3]):[0-5][0-9]$"},
        "date": {"type": ["string", "null"]},
        "evidence": {"type": "string"},
    },
    "required": ["type", "evidence"],
}

SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "comparison_reference": _REF,
        "findings": {
            "type": "object",
            "properties": {
                c: {
                    "type": "object",
                    "properties": {
                        "change": {"enum": CHANGE},
                        "absolute": {"enum": ABSOLUTE},
                        "evidence": {"type": "string"},
                    },
                    "required": ["change", "absolute", "evidence"],
                } for c in CHANNELS
            },
            "required": CHANNELS,
        },
        "devices": {
            "type": "object",
            "properties": {
                d: {
                    "type": "object",
                    "properties": {
                        "state": {"enum": DEVICE_STATE},
                        "side": {"enum": ["left", "right", "none"]},
                        "evidence": {"type": "string"},
                    },
                    "required": ["state", "side", "evidence"],
                } for d in DEVICES
            },
            "required": DEVICES,
        },
        "device_enumeration": {"type": "boolean"},  # the report explicitly lists the lines and tubes present
    },
    "required": ["comparison_reference", "findings", "devices", "device_enumeration"],
}

# v2 (2026-10-21, after the Open-i development smoke): absent vs not_mentioned, bilateral statements on both
# sides, contiguous evidence spans, placeholder redactions. v3: a date reference is a complete ISO date (the resolver
# matches films by calendar day). v4: absent only for an explicit negation that names the finding or device (v3 read
# "The lungs are clear." as edema absent in 475 of 493 accepted Open-i reports and set unnamed devices absent).
# The last development change on Open-i; fixed before any gold-set validation.
PROMPT_VERSION = "v4"
PROMPT = """You extract structured labels from one chest radiograph report.
Return JSON that matches the schema exactly.
Evidence: copy one contiguous span of the report exactly as written (same words in the same order, no ellipses, no
rewording); use an empty evidence string when the value is not_mentioned.
comparison_reference: the prior study this report compares with, as stated: relative_day with day_offset (e.g.
"yesterday" -> -1), clock_time with HH:MM, date with a complete date written as YYYY-MM-DD (the evidence keeps the
report's own spelling), redacted when the date or time is replaced by a placeholder (e.g. ___ or XXXX), multiple when
several priors are cited, other_modality for CT or other imaging, none when no comparison is stated.
findings: edema = pulmonary edema or vascular congestion; pleural effusion by side. A statement that covers both
sides ("bilateral effusions", "no pleural effusion", "no effusions") is extracted for both effusion_L and effusion_R;
effusion_unsided is only for an effusion whose side is not stated ("small pleural effusion").
change: the change stated relative to the comparison study; not_mentioned when no change is stated.
absolute: the state in this film. absent only when the report explicitly says the finding is not present and names
it or a synonym ("no pleural effusion", "no edema", "no vascular congestion", "normal pulmonary vascularity");
statements that do not name the finding ("the lungs are clear", "no acute cardiopulmonary process") leave it
not_mentioned.
devices: present when the report describes the device in this film; absent only when it explicitly says the device is
not present; placed_interval, removed_interval or repositioned for an interval change stated relative to the
comparison; not_mentioned for a device the report does not name. side: side of insertion or position when stated,
else none.
device_enumeration: true only if the report explicitly enumerates the support devices present.

Report:
{report}
"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def evidence_ok(extraction: Dict, report: str) -> List[str]:
    """Paths whose evidence is missing for an asserted value or is not a verbatim substring of the report; the
    comparison ignores letter case and runs of whitespace only (a span re-capitalised at a sentence start is kept,
    a paraphrase or an ellipsis-joined span is not)."""
    bad = []
    rep = _norm(report)

    def chk(path, node, asserted):
        ev = node.get("evidence", "")
        if asserted and not ev:
            bad.append(path + ":missing")
        elif ev and _norm(ev) not in rep:
            bad.append(path + ":not_substring")

    ref = extraction.get("comparison_reference", {})
    chk("comparison_reference", ref, ref.get("type", "none") != "none")
    for c, node in extraction.get("findings", {}).items():
        chk(f"findings.{c}", node, node.get("change") != "not_mentioned" or node.get("absolute") != "not_mentioned")
    for d, node in extraction.get("devices", {}).items():
        chk(f"devices.{d}", node, node.get("state") != "not_mentioned")
    return bad


def reference_fields_ok(extraction: Dict) -> Optional[str]:
    ref = extraction.get("comparison_reference", {})
    need = {"relative_day": "day_offset", "clock_time": "clock_time", "date": "date"}.get(ref.get("type"))
    if need and ref.get(need) in (None, ""):
        return f"reference_field_missing:{need}"
    if need == "date":
        try:
            ok = re.fullmatch(r"\d{4}-\d{2}-\d{2}", ref["date"]) and dt.date.fromisoformat(ref["date"])
        except ValueError:
            ok = False
        if not ok:
            return "reference_field_invalid:date"     # the resolver matches films by calendar day (ISO date)
    return None


def validate_extraction(extraction: Dict, report: str) -> Optional[str]:
    """None if acceptable; otherwise the rejection reason (schema, reference fields or evidence)."""
    try:
        jsonschema.validate(extraction, SCHEMA)
    except jsonschema.ValidationError as e:
        return f"schema:{str(e).splitlines()[0][:120]}"
    why = reference_fields_ok(extraction)
    if why:
        return why
    bad = evidence_ok(extraction, report)
    return None if not bad else "evidence:" + ",".join(bad)
