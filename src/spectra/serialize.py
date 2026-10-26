"""One compiler, one capacity rule: every system's input is compiled from the same CER stream.

Mode R (recorded care, E1): Delta and recorded actions up to capacity C.
Mode S (specification, E2): Delta, then the background B' (selected once per test by fixed type priority,
up to C - max(|M_A|, |M_B|) events and used in both arms), then the complete device bundle M, with fixed
delimiters. Track Delta carries "actions unknown", which is distinct from "no recorded actions".

Styles: ``verbal`` (prompted systems and state predictors), ``event_text`` (EHRXDiff-local; bundle events
prefixed "SPECIFICATION:"), ``compact`` (CLIP-encoder editors, 77-token form) and ``typed`` (reference
family tokens).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import constants, mimic_ontology_config
from .types import CER, Device, E2Test

DEVICE_NAMES = {
    "ETT": "endotracheal tube",
    "CVC": "central venous catheter",
    "PICC": "peripherally inserted central catheter",
    "enteric": "enteric tube",
    "chest_tube": "chest tube",
}
NAME_TO_DEVICE = {v: k for k, v in DEVICE_NAMES.items()}
SITE_NAMES = {"IJ": "internal jugular", "subclavian": "subclavian", "femoral": "femoral",
              "brachial/basilic/cephalic": "arm vein", "nasal/oral": "nasal or oral", "pleural": "pleural",
              "other": "other site", "none": ""}
NAME_TO_SITE = {v: k for k, v in SITE_NAMES.items() if v}
SIDE_NAMES = {"L": "left", "R": "right", "none": ""}
NAME_TO_SIDE = {"left": "L", "right": "R"}
COMPACT_DEVICE = {"ETT": "ETT", "CVC": "CVC", "PICC": "PICC", "enteric": "NGT", "chest_tube": "CT"}
COMPACT_TO_DEVICE = {v: k for k, v in COMPACT_DEVICE.items()}
COMPACT_SITE = {"IJ": "IJ", "subclavian": "SC", "femoral": "FEM", "brachial/basilic/cephalic": "ARM",
                "nasal/oral": "", "pleural": "", "other": "", "none": ""}
COMPACT_TO_SITE = {v: k for k, v in COMPACT_SITE.items() if v}

RECORDED_HEADER = "Recorded actions (may be incomplete):"
UNKNOWN_ACTIONS = "Recorded actions: unknown."
NO_ACTIONS = "none"
SPEC_TAIL = {"ETT": " and its dependent treatments", "enteric": " and its dependent treatments"}


def spec_header(device: str) -> str:
    return f"Device specification (complete for the {DEVICE_NAMES[device]}{SPEC_TAIL.get(device, '')}):"


def _time(c: CER) -> str:
    return "before" if c.is_state else f"+{c.t_start_h:.1f} h"


def _loc(dev: Device) -> str:
    return " ".join(x for x in (SIDE_NAMES[dev.side], SITE_NAMES[dev.site]) if x)


def render_event(c: CER) -> str:
    """Human-readable event with a machine-parsable device phrase."""
    t = _time(c)
    if c.device.type != "none":
        name = DEVICE_NAMES[c.device.type]
        if c.kind == "state":
            verb = "in place"
        else:
            verb = "placement" if c.device.transition == "place" else "removal"
        loc = _loc(c.device)
        return f"[{t}] {name} {verb}" + (f" ({loc})" if loc else "")
    at = c.action_type
    if at == "invasive_ventilation":
        return f"[{t}] invasive ventilation " + ("ongoing" if c.is_state else "start")
    if at == "invasive_ventilation_stop":
        return f"[{t}] invasive ventilation stop"
    if at == "noninvasive_ventilation":
        return f"[{t}] non-invasive ventilation " + ("ongoing" if c.is_state else "start")
    if at == "noninvasive_ventilation_stop":
        return f"[{t}] non-invasive ventilation stop"
    if at == "ventilator_setting":
        return f"[{t}] ventilator setting {c.value}"
    if c.kind == "observation":
        return f"[{t}] {_observation_text(c)}"
    label = c.value or at.replace("_", " ")
    dose = "" if c.dose_value is None else f" {c.dose_value:g} {c.dose_unit or ''}".rstrip()
    return f"[{t}] {label}{dose}{_course(c)}"


def _course(c: CER) -> str:
    """How an interval action runs on: through Delta, or stopped inside the window."""
    if c.ongoing_at_delta:
        return " (ongoing)"
    if c.t_end_h is not None and (c.is_state or c.t_end_h > c.t_start_h):
        return f" (stopped +{c.t_end_h:.1f} h)"
    return ""


def _observation_text(c: CER) -> str:
    """Track B observation: the measured variable or test, then its value (explicitly unknown when absent)."""
    kind, _, name = c.action_type.partition(":")[2].rpartition(":")
    name = name.replace("_", " ")
    value = c.value if c.value not in (None, "") else "value unknown"
    if kind == "lab":
        return f"lab {name}: {value}"
    if kind == "micro":
        return f"{name} specimen collected" + (f" ({c.value})" if c.value else "")
    if kind == "micro_result":
        return f"{name} result: {value}"
    return f"{name}: {value}"


EVENT_RE = re.compile(
    r"\[(?P<t>before|\+(?P<h>\d+(?:\.\d+)?) h)\] (?P<name>" + "|".join(map(re.escape, DEVICE_NAMES.values())) +
    r") (?P<verb>placement|removal|in place)(?: \((?P<loc>[^)]*)\))?")


def parse_device_events(segment: str) -> List[CER]:
    """Inverse of render_event for device phrases (the only content the round-trip check must recover)."""
    out = []
    for m in EVENT_RE.finditer(segment):
        dev = NAME_TO_DEVICE[m.group("name")]
        side, site = "none", "none"
        if m.group("loc"):
            words = m.group("loc").split(" ", 1)
            if words[0] in NAME_TO_SIDE:
                side = NAME_TO_SIDE[words[0]]
                rest = words[1] if len(words) > 1 else ""
            else:
                rest = m.group("loc")
            site = NAME_TO_SITE.get(rest, "none")
        verb = m.group("verb")
        t = None if m.group("t") == "before" else float(m.group("h"))
        if verb == "in place":
            out.append(CER("state", "parsed", None, device=Device(dev, "none", site, side)))
        else:
            out.append(CER("transition", "parsed", t, device=Device(dev, "place" if verb == "placement" else "remove", site, side)))
    return out


# ---------------------------------------------------------------------------------------------
# capacity rule
# ---------------------------------------------------------------------------------------------
def _priority_index() -> Dict[str, int]:
    order = mimic_ontology_config()["priority_order"]
    return {p: i for i, p in enumerate(order)}


def select_background(background: Sequence[CER], budget: int) -> Tuple[CER, ...]:
    """B' subset of B by fixed type priority (device procedures > ventilator settings > diuretics and
    vasoactives > transfusions > fluids > antibiotics > other medications), then time; returned in
    chronological order."""
    if budget <= 0:
        return ()
    idx = _priority_index()
    ranked = sorted(background, key=lambda c: (idx.get(c.priority, len(idx)), c.sort_key()))
    keep = ranked[:budget]
    return tuple(sorted(keep, key=lambda c: c.sort_key()))


def shared_background(test: E2Test, capacity: Optional[int] = None) -> Tuple[CER, ...]:
    capacity = constants()["capacity"]["events_llm"] if capacity is None else capacity
    budget = capacity - max(len(test.bundle_a), len(test.bundle_b))
    return select_background(test.background, budget)


def token_budget_background(test: E2Test, style: str, tokenizer, max_tokens: int,
                            capacity: Optional[int] = None, wrap=None) -> Optional[Tuple[CER, ...]]:
    """Capacity rule for token-limited encoders (CLIP, 77 tokens): Delta and M are always carried; B' is the
    largest priority prefix of the event-capacity background for which BOTH arms encode within max_tokens
    (the prompt template, `wrap`, included). Returns None when even Delta + M does not fit (the
    configuration is rejected for E2)."""
    wrap = wrap or (lambda t: t)
    full = shared_background(test, capacity)
    idx = _priority_index()
    ranked = sorted(full, key=lambda c: (idx.get(c.priority, len(idx)), c.sort_key()))

    def fits(k: int) -> bool:
        b = tuple(sorted(ranked[:k], key=lambda c: c.sort_key()))
        return all(len(tokenizer.encode(wrap(serialize_mode_s(test, arm, style, capacity, b).text))) <= max_tokens
                   for arm in ("A", "B"))

    if not fits(0):
        return None
    lo, hi = 0, len(ranked)
    while lo < hi:  # largest k that fits (the compact form only grows with k)
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return tuple(sorted(ranked[:lo], key=lambda c: c.sort_key()))


# ---------------------------------------------------------------------------------------------
# serialized form with segment spans
# ---------------------------------------------------------------------------------------------
@dataclass
class Serialized:
    text: str
    segments: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # character spans
    tokens: Optional[List[tuple]] = None  # typed style only


def _delta_text(delta_h: float) -> str:
    return f"Interval: {delta_h:.1f} hours after the prior radiograph."


def _join(events: Sequence[CER]) -> str:
    return "; ".join(render_event(c) for c in events) if events else NO_ACTIONS


def serialize_mode_r(stream: Sequence[CER], delta_h: float, actions_known: bool = True,
                     capacity: Optional[int] = None, style: str = "verbal") -> Serialized:
    capacity = constants()["capacity"]["events_llm"] if capacity is None else capacity
    head = _delta_text(delta_h)
    if not actions_known:
        body = UNKNOWN_ACTIONS
    else:
        events = select_background(stream, capacity)
        if style == "event_text":
            body = RECORDED_HEADER + "\n" + ("\n".join(render_event(c) for c in events) if events else NO_ACTIONS)
        else:
            body = f"{RECORDED_HEADER} {_join(events)}."
    text = head + "\n" + body
    return Serialized(text, {"delta": (0, len(head)), "background": (len(head) + 1, len(text))})


def serialize_mode_s(test: E2Test, arm: str, style: str = "verbal", capacity: Optional[int] = None,
                     background: Optional[Tuple[CER, ...]] = None) -> Serialized:
    """Serialization order Delta, B', M. B' is passed in (or recomputed identically) so both arms share it."""
    if arm not in ("A", "B"):
        raise ValueError("arm must be 'A' or 'B'")
    bundle = test.bundle_a if arm == "A" else test.bundle_b
    bprime = shared_background(test, capacity) if background is None else background
    if style == "typed":
        return _typed(test, bundle, bprime)
    if style == "compact":
        return _compact(test, bundle, bprime)
    head = _delta_text(test.delta_h)
    if not test.actions_known:
        bg = UNKNOWN_ACTIONS
    elif style == "event_text":
        bg = RECORDED_HEADER + "\n" + ("\n".join(render_event(c) for c in bprime) if bprime else NO_ACTIONS)
    else:
        bg = f"{RECORDED_HEADER} {_join(bprime)}."
    devices = [test.device] + ([test.second_device] if test.second_device else [])
    header = " ".join(spec_header(d) for d in devices) if len(devices) == 1 else \
        f"Device specification (complete for the {DEVICE_NAMES[devices[0]]} and the {DEVICE_NAMES[devices[1]]}):"
    if style == "event_text":
        spec = header + "\n" + ("\n".join("SPECIFICATION: " + render_event(c) for c in bundle) if bundle else "SPECIFICATION: none")
    else:
        spec = f"{header} {_join(bundle)}."
    text = head + "\n" + bg + "\n" + spec
    s_bg = len(head) + 1
    s_spec = s_bg + len(bg) + 1
    return Serialized(text, {"delta": (0, len(head)), "background": (s_bg, s_bg + len(bg)), "bundle": (s_spec, len(text))})


def _typed_token(c: CER, part: str) -> tuple:
    t_bin = "state" if c.is_state else ("<1h" if c.t_start_h < 1 else "<12h" if c.t_start_h < 12 else "<48h" if c.t_start_h < 48 else ">=48h")
    dose_bin = "none" if c.dose_value is None else ("low" if c.dose_value < 10 else "mid" if c.dose_value < 100 else "high")
    return (part, c.action_type, t_bin, c.device.type, c.device.transition, c.device.site, c.device.side, dose_bin)


def _typed(test: E2Test, bundle, bprime) -> Serialized:
    toks = [("delta", round(test.delta_h, 1))]
    if not test.actions_known:
        toks.append(("background", "actions_unknown"))
    else:
        toks += [_typed_token(c, "background") for c in bprime]
    toks += [_typed_token(c, "bundle") for c in bundle]
    return Serialized("", {}, toks)


def _compact_event(c: CER) -> str:
    if c.device.type != "none":
        code = COMPACT_DEVICE[c.device.type]
        mark = "in" if c.kind == "state" else ("+" if c.device.transition == "place" else "-")
        loc = " ".join(x for x in (c.device.side if c.device.side != "none" else "", COMPACT_SITE[c.device.site]) if x)
        when = "" if c.is_state else f"@{c.t_start_h:.0f}h"
        return f"{code}{mark}{(' ' + loc) if loc else ''}{when}"
    short = {"invasive_ventilation": "vent+", "invasive_ventilation_stop": "vent-", "noninvasive_ventilation": "niv",
             "ventilator_setting": "vset", "diuretic": "diuretic", "vasoactive": "pressor",
             "neuromuscular_blocker": "nmb", "transfusion": "transfusion", "fluid": "fluid", "antibiotic": "abx",
             "enteral_feed": "feed"}
    return short.get(c.action_type, c.action_type.split("_")[0][:10])


def _compact(test: E2Test, bundle, bprime) -> Serialized:
    head = f"dt {test.delta_h:.0f}h."
    bg_items = [] if not test.actions_known else sorted(set(_compact_event(c) for c in bprime))
    bg = "bg unknown." if not test.actions_known else ("bg " + ", ".join(bg_items) + "." if bg_items else "bg none.")
    dev_events = [c for c in bundle if c.device.type != "none"]
    dep = sorted(set(_compact_event(c) for c in bundle if c.device.type == "none"))
    spec_items = [_compact_event(c) for c in sorted(dev_events, key=lambda c: c.sort_key())] + dep
    spec = "spec " + (", ".join(spec_items) if spec_items else "none") + "."
    text = head + "\n" + bg + "\n" + spec
    s_bg = len(head) + 1
    s_spec = s_bg + len(bg) + 1
    return Serialized(text, {"delta": (0, len(head)), "background": (s_bg, s_bg + len(bg)), "bundle": (s_spec, len(text))})


COMPACT_RE = re.compile(
    r"\b(?P<code>ETT|CVC|PICC|NGT|CT)\s*(?P<mark>in\b|\+|-)\s*(?:(?P<side>[LR])\b\s*)?(?:(?P<site>IJ|SC|FEM|ARM)\b\s*)?(?:@\s*(?P<h>\d+)\s*h)?",
    re.IGNORECASE)


def parse_compact_bundle(segment: str) -> List[CER]:
    """Case- and spacing-tolerant (CLIP decoding lower-cases and re-spaces the text)."""
    out = []
    for m in COMPACT_RE.finditer(segment):
        dev = COMPACT_TO_DEVICE[m.group("code").upper()]
        side = (m.group("side") or "none").upper() if m.group("side") else "none"
        site = COMPACT_TO_SITE.get((m.group("site") or "").upper(), "none")
        if m.group("mark").lower() == "in":
            out.append(CER("state", "parsed", None, device=Device(dev, "none", site, side)))
        else:
            t = float(m.group("h")) if m.group("h") else 0.0
            out.append(CER("transition", "parsed", t, device=Device(dev, "place" if m.group("mark") == "+" else "remove", site, side)))
    return out


def bundle_segment(decoded: str, style: str, device: str, second_device: Optional[str] = None) -> Optional[str]:
    """Locate the bundle segment in decoded text; None when the delimiter did not survive tokenization."""
    if style == "compact":
        ms = list(re.finditer(r"\bspec\b", decoded, re.IGNORECASE))
        return None if not ms else decoded[ms[-1].end():]  # the bundle is the last segment
    if second_device:
        marker = f"Device specification (complete for the {DEVICE_NAMES[device]} and the {DEVICE_NAMES[second_device]}):"
    else:
        marker = spec_header(device)
    i = decoded.find(marker)
    return None if i < 0 else decoded[i + len(marker):]



# ---------------------------------------------------------------------------------------------
# typed tokens: independent decoding (the round-trip check reads the tokens the system receives)
# ---------------------------------------------------------------------------------------------
def decode_typed(tokens: Sequence[tuple]) -> Tuple[Optional[float], List[tuple], List[CER]]:
    """Inverse of the typed serializer: (Delta, background tokens, bundle device CERs in token order)."""
    delta, background, bundle = None, [], []
    for i, tok in enumerate(tokens):
        if not isinstance(tok, tuple) or not tok:
            continue
        if tok[0] == "delta" and len(tok) == 2:
            delta = float(tok[1])
        elif tok[0] in ("background", "recorded"):
            background.append(tok)
        elif tok[0] == "bundle" and len(tok) == 8:
            _, at, t_bin, dev, tr, site, side, _dose = tok
            if dev in ("none", None) or dev not in DEVICE_NAMES:
                continue
            if tr == "none":
                bundle.append(CER("state", "decoded", None, device=Device(dev, "none", site, side)))
            elif tr in ("place", "remove"):
                bundle.append(CER("transition", "decoded", float(i), device=Device(dev, tr, site, side)))
    return delta, background, bundle


# ---------------------------------------------------------------------------------------------
# mode R for every style (E1 queries); the capacity rule caps recorded actions at C events
# ---------------------------------------------------------------------------------------------
def serialize_mode_r_style(stream: Sequence[CER], delta_h: float, style: str, actions_known: bool = True,
                           capacity: Optional[int] = None, tokenizer=None, max_tokens: Optional[int] = None,
                           wrap=None) -> Serialized:
    capacity = constants()["capacity"]["events_llm"] if capacity is None else capacity
    wrap = wrap or (lambda t: t)
    if style in ("verbal", "event_text"):
        return serialize_mode_r(stream, delta_h, actions_known, capacity, style)
    events = select_background(stream, capacity) if actions_known else ()
    if style == "typed":
        toks = [("delta", round(delta_h, 1))]
        toks += [_typed_token(c, "recorded") for c in events] if actions_known else [("recorded", "actions_unknown")]
        return Serialized("", {}, toks)
    if style == "compact":
        idx = _priority_index()
        ranked = sorted(events, key=lambda c: (idx.get(c.priority, len(idx)), c.sort_key()))

        def text(k):
            head = f"dt {delta_h:.0f}h."
            if not actions_known:
                return head + "\nrec unknown."
            items = sorted(set(_compact_event(c) for c in ranked[:k]))
            return head + "\n" + ("rec " + ", ".join(items) + "." if items else "rec none.")

        k = len(ranked)
        if tokenizer is not None and max_tokens is not None:
            while k > 0 and len(tokenizer.encode(wrap(text(k)))) > max_tokens:
                k -= 1
        t = text(k)
        return Serialized(t, {"delta": (0, t.index("\n")), "background": (t.index("\n") + 1, len(t))})
    raise ValueError(f"unknown style {style!r}")
