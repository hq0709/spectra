"""E2: device specification tests and joint device accuracy (FINAL_PROPOSAL.md, Core Mechanism 3-5).

A test tau = (x0, d, s0, B, M_A, M_B, Delta). The two arms share the prior, the interval and the
background B and differ only in the mutable bundle M of the scored device type d. The expected state
g_d(s0, M) is a deterministic state machine over the ordered transitions of d in M.

Bundle membership is read from the dependency table (resources/dependency_table.yaml): the device's own
transitions and states, its dependent actions (an interval action moves together with its stop event),
the settings charted while the device is in use, and the dependents that start after a placement or a
removal. Everything else is background. Track B observations that presuppose the device are withheld
from both arms.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import constants, dependency_table
from .types import (CER, PRIMARY_DEVICES, PRIMARY_EDIT_CLASSES, Device, DeviceState, E2Test, Film)

SIDED_SITES = ("IJ", "subclavian", "brachial/basilic/cephalic")
OTHER_SIDE = {"L": "R", "R": "L"}
SECONDARY_LOGGED = "logged_transition"


# ---------------------------------------------------------------------------------------------
# state machine
# ---------------------------------------------------------------------------------------------
def expected_state(s0: DeviceState, bundle: Sequence[CER], device: str) -> Optional[DeviceState]:
    """g_d(s0, M): place -> present(side); remove -> absent; empty -> s0 persists.
    Returns None when a transition is inconsistent with the current state (the test is ineligible)."""
    state = s0
    for c in sorted((c for c in bundle if c.kind == "transition" and c.device.type == device),
                    key=lambda c: (c.t_start_h, c.action_type)):
        if c.device.transition == "place":
            if state.present:
                return None
            state = DeviceState(True, c.device.side)
        elif c.device.transition == "remove":
            if not state.present:
                return None
            state = DeviceState(False, "none")
    return state


# ---------------------------------------------------------------------------------------------
# dependency-table helpers
# ---------------------------------------------------------------------------------------------
def _dep_types(device: str, deps: dict) -> Dict[str, set]:
    spec = deps.get(device, {})
    pb, rb = spec.get("placement_bundle", {}) or {}, spec.get("removal_bundle", {}) or {}
    pairs = deps.get("interval_pairs", {}) or {}
    stops = {pairs[a] for a in (pb.get("dependent_actions", []) or []) if a in pairs}
    return {
        "place_actions": set(pb.get("dependent_actions", []) or []),
        "after_place": set(pb.get("dependent_after_placement", []) or []),
        "remove_actions": set(rb.get("dependent_actions", []) or []) | stops,
        "after_remove": set(rb.get("dependent_after_removal", []) or []),
        "pairs": dict(pairs),
    }


def _in_use_intervals(stream: Sequence[CER], device: str, deps: dict, window_end_h: float) -> List[Tuple[float, float]]:
    """Intervals when the device's dependent process runs (ETT: invasive ventilation), from start/stop events."""
    d = _dep_types(device, deps)
    starts = d["place_actions"]
    if not starts:
        return []
    pairs = d["pairs"]
    events = []
    for c in stream:
        if c.action_type in starts:
            events.append((0.0 if c.is_state else c.t_start_h, +1))
        elif c.action_type in {pairs.get(a) for a in starts}:
            events.append((c.t_start_h, -1))
    events.sort()
    out, start = [], None
    for t, kind in events:
        if kind == +1 and start is None:
            start = t
        elif kind == -1 and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, window_end_h))
    return out


def _within(t: Optional[float], intervals: Sequence[Tuple[float, float]]) -> bool:
    t = 0.0 if t is None else t
    return any(a <= t <= b for a, b in intervals)


@dataclass(frozen=True)
class RecordEvidence:
    """What the records say about the device at t0: 'present' (active rows), 'absent_unknown' (no rows:
    absence is not documented, so it is unknown), or 'ambiguous' (more than one instance)."""

    status: str
    instances: Tuple[CER, ...] = ()

    @property
    def present(self) -> bool:
        return self.status == "present"


def record_evidence_at_t0(stream: Sequence[CER], device: str, deps: Optional[dict] = None) -> RecordEvidence:
    deps = deps or dependency_table()
    states = tuple(c for c in stream if c.is_state and c.device.type == device)
    if len(states) > 1:
        return RecordEvidence("ambiguous", states)
    if states:
        return RecordEvidence("present", states)
    d = _dep_types(device, deps)
    if any(c.is_state and c.action_type in d["place_actions"] for c in stream):  # e.g. ventilation ongoing -> ETT
        return RecordEvidence("present")
    return RecordEvidence("absent_unknown")


def record_state_at_t0(stream: Sequence[CER], device: str) -> Optional[DeviceState]:
    """Backward-compatible view: present / absent-unknown as DeviceState, None if ambiguous."""
    ev = record_evidence_at_t0(stream, device)
    if ev.status == "ambiguous":
        return None
    if ev.present:
        side = ev.instances[0].device.side if ev.instances else "none"
        return DeviceState(True, side)
    return DeviceState(False)


@dataclass(frozen=True)
class Partition:
    background: Tuple[CER, ...]
    common: Tuple[CER, ...]        # target-device CERs shared by both arms (inside M)
    placement: Tuple[CER, ...]     # placement transition(s) and their dependency bundle
    removal: Tuple[CER, ...]       # removal transition(s) and their dependency bundle
    withheld: Tuple[CER, ...]      # Track B dependents of the scored device (in neither arm)
    ambiguous: Tuple[str, ...] = ()  # dependency associations the window cannot resolve (tests are rejected)


def partition(stream: Sequence[CER], device: str, window_end_h: float, deps: Optional[dict] = None) -> Partition:
    """Split a compiled window into background B and the target-device material that forms M."""
    deps = deps or dependency_table()
    d = _dep_types(device, deps)
    in_use = _in_use_intervals(stream, device, deps, window_end_h)
    t_places = [c.t_start_h for c in stream if c.kind == "transition" and c.device.type == device and c.device.transition == "place"]
    t_removes = [c.t_start_h for c in stream if c.kind == "transition" and c.device.type == device and c.device.transition == "remove"]
    first_place = min(t_places) if t_places else None
    first_remove = min(t_removes) if t_removes else None
    pairs = d["pairs"]
    stop_of = {v: k for k, v in pairs.items()}

    background, common, placement, removal, withheld = [], [], [], [], []
    part_of_start: Dict[str, List[Tuple[float, str]]] = defaultdict(list)  # interval action -> [(t, part)]
    device_at_t0 = record_evidence_at_t0(stream, device, deps).status != "absent_unknown"

    def put(c: CER, part: str):
        {"common": common, "placement": placement, "removal": removal, "background": background}[part].append(c)
        if c.action_type in pairs:
            part_of_start[c.action_type].append((-1.0 if c.is_state else c.t_start_h, part))

    stops = []
    for c in stream:
        if c.track == "B" and c.origin == f"dependent:{device}":
            withheld.append(c)
            continue
        if c.device.type == device:
            if c.kind == "state":
                put(c, "common")
            elif c.device.transition == "place":
                put(c, "placement")
            else:
                put(c, "removal")
            continue
        at = c.action_type
        if at in stop_of:
            stops.append(c)  # decided after the starts are placed
            continue
        if at in d["place_actions"]:
            if c.is_state:
                put(c, "common")
            elif first_place is not None and c.t_start_h >= first_place:
                put(c, "placement")
            else:
                put(c, "background")
            continue
        if at in d["after_place"]:
            t = c.t_start_h
            # devices with a dependent process (ETT: ventilation) bind settings charted while it runs;
            # devices without one (enteric tube: feeds) bind every dependent after the placement
            in_use_ok = _within(t, in_use) if d["place_actions"] else True
            if first_place is not None and t is not None and t >= first_place and in_use_ok:
                put(c, "placement")
            elif first_remove is not None and t is not None and t > first_remove:
                put(c, "removal")
            elif (_within(t, in_use) if d["place_actions"] else False) or (c.is_state and device_at_t0):
                put(c, "common")
            else:
                put(c, "background")
            continue
        if at in d["after_remove"]:
            if first_remove is not None and not c.is_state and c.t_start_h >= first_remove:
                put(c, "removal")
            else:
                put(c, "background")
            continue
        put(c, "background")

    # a stop event follows its interval: the part of the latest start at or before it. The stop the removal
    # requires (ventilation stop at extubation) ends an interval running in M; it joins the removal bundle
    # when the window holds exactly one removal and one such stop, whichever is charted first
    # (ventilation is often stopped minutes before the extubation is documented).
    def start_part(c):
        prev = [(t, p) for t, p in part_of_start[stop_of[c.action_type]] if t <= c.t_start_h]
        return max(prev)[1] if prev else "background"

    required = [c for c in stops if (stop_of[c.action_type] in d["remove_actions"] or c.action_type in d["remove_actions"])
                and start_part(c) == "common"]
    ambiguous = ()
    if required and t_removes and (len(required) > 1 or len(t_removes) > 1):
        ambiguous = ("removal_dependency_association",)
    for c in stops:
        if len(required) == 1 and c is required[0] and len(t_removes) == 1:
            removal.append(c)
            continue
        {"common": common, "placement": placement, "removal": removal, "background": background}[start_part(c)].append(c)

    mk = lambda xs, part: tuple(x.with_part(part) for x in sorted(xs, key=lambda c: c.sort_key()))
    return Partition(mk(background, "background"), mk(common, "bundle"), mk(placement, "bundle"),
                     mk(removal, "bundle"), tuple(withheld), ambiguous)


def _ongoing(common: Sequence[CER], removal: Sequence[CER], device: str, deps: dict) -> Tuple[CER, ...]:
    """The edited arm of a removal deletion: what the deleted removal ended persists through Delta — the
    device's own state, the intervals whose stop was deleted with it, and the latest value of each setting
    the dependency table carries forward. Superseded settings and other records are kept as recorded."""
    d = _dep_types(device, deps)
    stopped = {a for a, stop in d["pairs"].items() if any(c.action_type == stop for c in removal)}
    carry = set((deps.get(device, {}).get("removal_bundle", {}) or {}).get("carry_forward_on_deletion", []) or [])
    latest = {}  # (type, setting name before "=") -> index of its latest value in time order
    for j in sorted(range(len(common)), key=lambda j: common[j].sort_key()):
        if common[j].action_type in carry:
            latest[(common[j].action_type, str(common[j].value).split("=")[0])] = j
    extend = set(latest.values())
    extend |= {i for i, c in enumerate(common) if (c.is_state and c.device.type == device)
               or (c.is_state and c.action_type in stopped)}
    return tuple(replace(c, ongoing_at_delta=True, t_end_h=None, origin="edit") if i in extend else c
                 for i, c in enumerate(common))


# ---------------------------------------------------------------------------------------------
# eligibility and test construction
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Rejection:
    patient_id: str
    prior_study: str
    device: str
    edit_class: str
    rule: str


def _test_id(*parts) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _transitions(xs: Sequence[CER], device: str, kind: str) -> List[CER]:
    return [c for c in xs if c.kind == "transition" and c.device.type == device and c.device.transition == kind]


def _excluded(stream: Sequence[CER], device: str, deps: dict,
              report_flags: Optional[Mapping[str, DeviceState]] = None) -> Optional[str]:
    """Visibility and confounding exclusions: tracheostomy for ETT; ports for CVC; femoral CVC."""
    spec = deps.get(device, {})
    for a in spec.get("exclusion_actions", []) or []:
        if any(c.action_type == a for c in stream):
            return f"exclusion:{a}"
        if report_flags and report_flags.get(a) is not None and report_flags[a].present:
            return f"exclusion:{a}"
    bad = set(spec.get("exclude_subtypes", []) or [])
    if bad and any(c.device.type == device and c.device.subtype in bad for c in stream):
        return "exclusion:subtype"
    if report_flags and report_flags.get("port") is not None and report_flags["port"].present and device == "CVC":
        return "exclusion:port"
    return None


def _common_checks(stream: Sequence[CER], device: str, s0: Optional[DeviceState], deps: dict,
                   report_flags=None, need_absent: bool = False) -> Optional[str]:
    """Checks every constructor applies: verified s0, record/reference consistency, single instance,
    exclusions. Missing records are not evidence of absence; only active rows contradict an absent s0."""
    if s0 is None:
        return "initial_state_unverified"
    ev = record_evidence_at_t0(stream, device, deps)
    if ev.status == "ambiguous":
        return "more_than_one_instance"
    if ev.present and not s0.present:
        return "record_report_conflict_at_prior"
    if need_absent and s0.present:
        return "initial_state_not_absent"
    why = _excluded(stream, device, deps, report_flags)
    if why:
        return why
    return None


def _site_ok(c: CER, device: str, deps: dict, primary: bool) -> Optional[str]:
    if device != "CVC":
        return None
    if c.device.site == "femoral":
        return "visibility:femoral"
    if primary and c.device.site not in deps["CVC"]["scored_sites"]:
        return "site_unknown_or_unscored"
    return None


def logged_tests(patient_id: str, prior: Film, stream: Sequence[CER], delta_h: float, device: str,
                 s0: Optional[DeviceState], follow_up: Optional[DeviceState], deps: Optional[dict] = None,
                 source: str = "", target_study: str = "", report_flags=None) -> Tuple[List[E2Test], List[Rejection]]:
    """Placement-deletion and removal-deletion tests from one logged window.

    s0 is the verified initial state (tier 1 / validated tier 2 report evidence); follow_up is the device
    state stated by the follow-up report (corroboration). Records alone never verify a state."""
    deps = deps or dependency_table()
    rej = lambda cls, rule: Rejection(patient_id, prior.study_id, device, cls, rule)
    why = _common_checks(stream, device, s0, deps, report_flags)
    if why:
        return [], [rej("logged", why)]
    part = partition(stream, device, delta_h, deps)
    places = _transitions(part.placement, device, "place")
    removes = _transitions(part.removal, device, "remove")
    if len(places) + len(removes) == 0:
        return [], [rej("logged", "no_transition")]
    if part.ambiguous:
        return [], [rej("logged", part.ambiguous[0])]
    if len(places) + len(removes) > 1:
        return [], [rej("logged", "more_than_one_transition")]
    states = [c for c in part.common if c.kind == "state" and c.device.type == device]
    if places:
        cls, p = "placement_deletion", places[0]
        if s0.present:
            return [], [rej(cls, "placement_with_device_present")]
        why = _site_ok(p, device, deps, primary=True)
        if why:
            return [], [rej(cls, why)]
        if device == "ETT" and not any(c.action_type == "invasive_ventilation" for c in part.placement):
            return [], [rej(cls, "incomplete_bundle")]
        if follow_up is None or not follow_up.present:
            return [], [rej(cls, "follow_up_not_corroborated")]
        bundle_a, bundle_b, t_tr = part.common + part.placement, part.common, p.t_start_h
    else:
        cls, r = "removal_deletion", removes[0]
        if not s0.present:
            return [], [rej(cls, "removal_with_device_absent")]
        if device == "CVC":
            if not states:
                return [], [rej(cls, "no_identified_instance")]
            why = _site_ok(states[0], device, deps, primary=True)
            if why:
                return [], [rej(cls, why)]
        if device == "ETT" and not any(c.action_type == "invasive_ventilation_stop" for c in part.removal):
            return [], [rej(cls, "incomplete_bundle")]
        if follow_up is None or follow_up.present:
            return [], [rej(cls, "follow_up_not_corroborated")]
        bundle_a, bundle_b, t_tr = part.common + part.removal, _ongoing(part.common, part.removal, device, deps), r.t_start_h
    g_a = expected_state(s0, bundle_a, device)
    g_b = expected_state(s0, bundle_b, device)
    if g_a is None or g_b is None or g_a.present == g_b.present:
        return [], [rej(cls, "inconsistent_specification")]
    tid = _test_id(source, patient_id, prior.study_id, target_study, round(delta_h, 3), device, cls)
    return [E2Test(tid, patient_id, prior, device, cls, s0, part.background, bundle_a, bundle_b, delta_h,
                   "presence", g_a.present, g_b.present, logged_arm="A", t_transition_h=t_tr, source=source,
                   meta={"target_study": target_study})], []


def mirror_test(patient_id: str, prior: Film, stream: Sequence[CER], delta_h: float, device: str,
                s0: Optional[DeviceState], follow_up_side: Optional[str], deps: Optional[dict] = None,
                source: str = "", target_study: str = "", report_flags=None) -> Tuple[Optional[E2Test], Optional[Rejection]]:
    """Mirror: a sided placement with known insertion-access side (records and report agree)."""
    deps = deps or dependency_table()
    rej = lambda rule: Rejection(patient_id, prior.study_id, device, "mirror", rule)
    why = _common_checks(stream, device, s0, deps, report_flags, need_absent=True)
    if why:
        return None, rej(why)
    if record_evidence_at_t0(stream, device, deps).present:
        return None, rej("instance_present_at_prior")
    part = partition(stream, device, delta_h, deps)
    places = _transitions(part.placement, device, "place")
    removes = _transitions(part.removal, device, "remove")
    if len(places) != 1 or removes:
        return None, rej("not_single_placement")
    p = places[0]
    if p.device.site == "femoral":
        return None, rej("visibility:femoral")
    if p.device.site not in SIDED_SITES or p.device.side not in ("L", "R"):
        return None, rej("side_unknown")
    if follow_up_side != p.device.side:
        return None, rej("record_report_side_disagree")
    flipped = replace(p, device=replace(p.device, side=OTHER_SIDE[p.device.side]))
    bundle_a = part.common + part.placement
    bundle_b = part.common + tuple(flipped if c is p else c for c in part.placement)
    g_a, g_b = expected_state(s0, bundle_a, device), expected_state(s0, bundle_b, device)
    tid = _test_id(source, patient_id, prior.study_id, target_study, round(delta_h, 3), device, "mirror")
    return E2Test(tid, patient_id, prior, device, "mirror", s0, part.background, bundle_a, bundle_b, delta_h,
                  "presence_side", g_a.in_space("presence_side"), g_b.in_space("presence_side"), logged_arm="A",
                  t_transition_h=p.t_start_h, source=source, meta={"target_study": target_study}), None


def insertion_bundle(device: str, t_h: float, side: str = "R") -> Tuple[CER, ...]:
    """Minimal coherent bundle for an inserted device (ETT: intubation + invasive-ventilation start)."""
    if device == "ETT":
        return (CER("transition", "intubation", t_h, device=Device("ETT", "place", "nasal/oral", "none", "Intubation"),
                    priority="device_procedure", part="bundle", origin="edit"),
                CER("action", "invasive_ventilation", t_h, ongoing_at_delta=True, priority="device_procedure",
                    part="bundle", origin="edit"))
    if device == "CVC":
        return (CER("transition", "line_placement", t_h, device=Device("CVC", "place", "IJ", side, "Multi Lumen"),
                    priority="device_procedure", part="bundle", origin="edit"),)
    if device == "PICC":
        return (CER("transition", "line_placement", t_h, device=Device("PICC", "place", "brachial/basilic/cephalic", side, "PICC Line"),
                    priority="device_procedure", part="bundle", origin="edit"),)
    if device == "enteric":
        return (CER("transition", "enteric_tube_placement", t_h, device=Device("enteric", "place", "nasal/oral", "none"),
                    priority="device_procedure", part="bundle", origin="edit"),)
    if device == "chest_tube":
        return (CER("transition", "chest_tube_placement", t_h, device=Device("chest_tube", "place", "pleural", side),
                    priority="device_procedure", part="bundle", origin="edit"),)
    raise ValueError(device)


def removal_bundle(device: str, t_h: float) -> Tuple[CER, ...]:
    names = {"ETT": "extubation", "CVC": "line_removal", "PICC": "line_removal", "enteric": "enteric_tube_removal",
             "chest_tube": "chest_tube_removal"}
    site = {"ETT": "nasal/oral", "CVC": "IJ", "PICC": "brachial/basilic/cephalic", "enteric": "nasal/oral",
            "chest_tube": "pleural"}[device]
    out = [CER("transition", names[device], t_h, device=Device(device, "remove", site, "none"),
               priority="device_procedure", part="bundle", origin="edit")]
    if device == "ETT":
        out.append(CER("action", "invasive_ventilation_stop", t_h, priority="device_procedure", part="bundle", origin="edit"))
    return tuple(out)


def _balanced_side(*parts) -> str:
    return "L" if int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest(), 16) % 2 else "R"


def _device_or_dependents_recorded(stream: Sequence[CER], device: str, deps: dict) -> bool:
    """Records of the device or of anything that presupposes it (ventilation, ventilator settings and
    paralysis for the ETT; enteral feeding for the enteric tube): an absent arm cannot keep them."""
    d = _dep_types(device, deps)
    dep = d["place_actions"] | d["remove_actions"] | d["after_place"]
    return any(c.device.type == device or c.action_type in dep for c in stream)


def insertion_test(patient_id: str, prior: Film, stream: Sequence[CER], delta_h: float, device: str,
                   s0: Optional[DeviceState], follow_up: Optional[DeviceState], deps: Optional[dict] = None,
                   offset_h: Optional[float] = None, source: str = "", target_study: str = "",
                   report_flags=None) -> Tuple[Optional[E2Test], Optional[Rejection]]:
    deps = deps or dependency_table()
    offset_h = constants()["e2"]["insertion_offset_h"] if offset_h is None else offset_h
    rej = lambda rule: Rejection(patient_id, prior.study_id, device, "insertion", rule)
    why = _common_checks(stream, device, s0, deps, report_flags, need_absent=True)
    if why:
        return None, rej(why)
    if follow_up is None or follow_up.present:
        return None, rej("follow_up_not_verified_absent")
    if _device_or_dependents_recorded(stream, device, deps):
        return None, rej("target_or_dependent_records_present")
    part = partition(stream, device, delta_h, deps)
    ins = insertion_bundle(device, offset_h, _balanced_side(patient_id, prior.study_id, device))
    g_a, g_b = expected_state(s0, part.common, device), expected_state(s0, part.common + ins, device)
    tid = _test_id(source, patient_id, prior.study_id, target_study, round(delta_h, 3), device, "insertion")
    return E2Test(tid, patient_id, prior, device, "insertion", s0, part.background, part.common, part.common + ins,
                  delta_h, "presence", g_a.present, g_b.present, logged_arm="A", source=source,
                  meta={"target_study": target_study}), None


def type_control_test(patient_id: str, prior: Film, stream: Sequence[CER], delta_h: float, d1: str, d2: str,
                      s0_1: Optional[DeviceState], s0_2: Optional[DeviceState], deps: Optional[dict] = None,
                      offset_h: Optional[float] = None, source: str = "", target_study: str = "",
                      report_flags=None) -> Tuple[Optional[E2Test], Optional[Rejection]]:
    """Fixed-length device-type control: two one-row device types, both verified absent at the prior."""
    deps = deps or dependency_table()
    offset_h = constants()["e2"]["insertion_offset_h"] if offset_h is None else offset_h
    rej = lambda rule: Rejection(patient_id, prior.study_id, f"{d1}+{d2}", "type_control", rule)
    one_row = set(deps["one_row_types"])
    if d1 not in one_row or d2 not in one_row or d1 == d2:
        return None, rej("not_one_row_types")
    for d, s in ((d1, s0_1), (d2, s0_2)):
        why = _common_checks(stream, d, s, deps, report_flags, need_absent=True)
        if why:
            return None, rej(f"{d}:{why}")
        if _device_or_dependents_recorded(stream, d, deps):
            return None, rej(f"{d}:target_or_dependent_records_present")
    withheld = {f"dependent:{d1}", f"dependent:{d2}"}
    background = tuple(c.with_part("background") for c in stream if not (c.track == "B" and c.origin in withheld))
    side = _balanced_side(patient_id, prior.study_id, d1, d2)
    m_a, m_b = insertion_bundle(d1, offset_h, side), insertion_bundle(d2, offset_h, side)
    if len(m_a) != len(m_b):
        return None, rej("bundles_not_fixed_length")
    tid = _test_id(source, patient_id, prior.study_id, target_study, round(delta_h, 3), d1, d2, "type_control")
    return E2Test(tid, patient_id, prior, d1, "type_control", s0_1, background, m_a, m_b, delta_h, "type_pair",
                  (True, False), (False, True), logged_arm=None, second_device=d2, source=source,
                  meta={"target_study": target_study}), None


def cross_source_tests(patient_id: str, prior: Film, device: str, s0: Optional[DeviceState],
                       report_flags: Optional[Mapping[str, DeviceState]] = None, deps: Optional[dict] = None,
                       source: str = "") -> Tuple[List[E2Test], List[Rejection]]:
    """Cross-source specification protocol: Delta = 24 h, event at t0 + 1 h, background = actions unknown.
    Insertion on verified-absent priors; removal specification on verified-present priors. report_flags
    carries the report-stated states the exclusions need (tracheostomy, port, another instance)."""
    deps = deps or dependency_table()
    k = constants()["e2"]["cross_source_protocol"]
    delta, t_ev = float(k["delta_h"]), float(k["event_offset_h"])
    rej = lambda rule: Rejection(patient_id, prior.study_id, device, "cross_source", rule)
    if s0 is None:
        return [], [rej("initial_state_unverified")]
    why = _excluded((), device, deps, report_flags)
    if why:
        return [], [rej(why)]
    if report_flags and report_flags.get(f"other_{device}") is not None and report_flags[f"other_{device}"].present:
        return [], [rej("more_than_one_instance")]
    if not s0.present:
        ins = insertion_bundle(device, t_ev, _balanced_side(patient_id, prior.study_id, device))
        tid = _test_id(source, patient_id, prior.study_id, device, "cross_insertion")
        return [E2Test(tid, patient_id, prior, device, "insertion", s0, (), (), ins, delta, "presence", False, True,
                       actions_known=False, source=source, meta={"protocol": "cross_source"})], []
    rem = removal_bundle(device, t_ev)
    tid = _test_id(source, patient_id, prior.study_id, device, "cross_removal")
    return [E2Test(tid, patient_id, prior, device, "removal_specification", s0, (), (), rem, delta, "presence",
                   True, False, actions_known=False, source=source, meta={"protocol": "cross_source"})], []


# ---------------------------------------------------------------------------------------------
# cohort selection (Algorithm 1 steps 7a / 7b)
# ---------------------------------------------------------------------------------------------
def _when(t: E2Test) -> float:
    return t.prior.t_h + (t.t_transition_h or 0.0)


def primary_cohort(tests: Iterable[E2Test], split: str = "test") -> List[E2Test]:
    """Every eligible test patient's earliest logged transition per primary device type; no caps.
    Ties (several windows containing the same earliest transition) go to the shortest horizon, i.e. the
    first follow-up after the transition, then to the test id."""
    best: Dict[Tuple[str, str], E2Test] = {}
    for t in tests:
        if t.prior.split != split or t.device not in PRIMARY_DEVICES or t.edit_class not in PRIMARY_EDIT_CLASSES:
            continue
        key = (t.patient_id, t.device)
        rank = (round(_when(t), 6), round(t.delta_h, 6), t.test_id)
        cur = best.get(key)
        if cur is None or rank < (round(_when(cur), 6), round(cur.delta_h, 6), cur.test_id):
            best[key] = t
    return sorted(best.values(), key=lambda t: (t.device, t.patient_id))


def allocate_secondary(candidates: Iterable[E2Test], target_per_type: Dict[str, int], seed: int = 0,
                       priority: Optional[Sequence[str]] = None, cap_fraction: Optional[float] = None,
                       exclude_patients: Optional[Dict[str, set]] = None, split: str = "test") -> List[E2Test]:
    """At most one secondary test per patient and device type. Secondary-device logged transitions come
    first; then mirror > type control > insertion, each capped at cap_fraction of the type's secondary
    target. A type control counts toward its first device's target and uses the patient for both devices."""
    k = constants()["e2"]
    priority = [SECONDARY_LOGGED] + list(priority or k["secondary_priority"])
    cap_fraction = k["secondary_class_cap_fraction"] if cap_fraction is None else cap_fraction
    used = defaultdict(set)
    for dev, pats in (exclude_patients or {}).items():
        used[dev] |= set(pats)
    rng = np.random.default_rng(seed)
    pool = [t for t in candidates if t.prior.split == split and not
            (t.edit_class in PRIMARY_EDIT_CLASSES and t.device in PRIMARY_DEVICES)]
    pool = [pool[i] for i in rng.permutation(len(pool))]
    cls_of = lambda t: SECONDARY_LOGGED if t.edit_class in PRIMARY_EDIT_CLASSES else t.edit_class
    chosen: List[E2Test] = []
    count = defaultdict(int)
    per_class = defaultdict(int)
    for cls in priority:
        for t in pool:
            if cls_of(t) != cls:
                continue
            dev = t.device
            target = int(target_per_type.get(dev, 0))
            if target <= 0 or count[dev] >= target:
                continue
            if cls != SECONDARY_LOGGED and per_class[(dev, cls)] >= int(np.floor(cap_fraction * target)):
                continue
            devices = [dev] + ([t.second_device] if t.second_device else [])
            if any(t.patient_id in used[d] for d in devices):
                continue
            chosen.append(t)
            count[dev] += 1
            per_class[(dev, cls)] += 1
            for d in devices:
                used[d].add(t.patient_id)
    return chosen


# ---------------------------------------------------------------------------------------------
# scoring: joint device accuracy, factual-arm and edited-arm accuracy
# ---------------------------------------------------------------------------------------------
class SystemBlocked(RuntimeError):
    """The system failed the mandatory round-trip check; its E2 scores may not be computed."""


def _correct(readout, expected) -> bool:
    """Failure rule: an unassessable readout (None) or an unassessed side counts as incorrect."""
    if readout is None:
        return False
    if isinstance(readout, tuple) and any(x is None or x == "unassessed" for x in readout):
        return False
    return readout == expected


def joint_accuracy(tests: Sequence[E2Test], readouts: Dict[Tuple[str, str, int], object], seeds: Sequence[int]) -> dict:
    """Patient-macro JDA, FA and EA. readouts[(test_id, arm, seed)] is the state-space readout of that output."""
    per_patient_j = defaultdict(list)
    per_patient_fa = defaultdict(list)
    per_patient_ea = defaultdict(list)
    for t in tests:
        js, fa, ea = [], [], []
        for s in seeds:
            ok_a = _correct(readouts.get((t.test_id, "A", s)), t.expected_a)
            ok_b = _correct(readouts.get((t.test_id, "B", s)), t.expected_b)
            js.append(float(ok_a and ok_b))
            if t.logged_arm == "A":
                fa.append(float(ok_a))
                ea.append(float(ok_b))
        per_patient_j[t.patient_id].append(np.mean(js))
        if fa:
            per_patient_fa[t.patient_id].append(np.mean(fa))
            per_patient_ea[t.patient_id].append(np.mean(ea))
    macro = lambda d: float(np.mean([np.mean(v) for v in d.values()])) if d else float("nan")
    return {"JDA": macro(per_patient_j), "FA": macro(per_patient_fa), "EA": macro(per_patient_ea),
            "n_patients": len(per_patient_j), "n_tests": len(tests)}


def jda_primary(tests: Sequence[E2Test], readouts, seeds, weights: Optional[Dict[str, float]] = None) -> dict:
    """JDA_primary: ETT and CVC weighted 1/2 each; within a device, classes at their cohort frequency."""
    weights = weights or constants()["e2"]["device_weights"]
    out, total = {}, 0.0
    for dev, w in weights.items():
        sub = [t for t in tests if t.device == dev and t.edit_class in PRIMARY_EDIT_CLASSES]
        r = joint_accuracy(sub, readouts, seeds)
        out[dev] = r
        total += w * r["JDA"]
    out["JDA_primary"] = total
    return out


def score_system(system: str, tests: Sequence[E2Test], readouts, seeds, roundtrip_results: Mapping[str, Sequence]) -> dict:
    """E2 scoring entry point: refuses a system without a passing round-trip certificate on every test."""
    results = roundtrip_results.get(system)
    if not results:
        raise SystemBlocked(f"{system}: no round-trip results; E2 scoring refused")
    failed = [r for r in results if not r.ok]
    covered = {r.test_id for r in results}
    missing = [t.test_id for t in tests if t.test_id not in covered]
    if failed or missing:
        raise SystemBlocked(f"{system}: round-trip failed on {len(failed)} tests, missing on {len(missing)}")
    return {"system": system, **jda_primary(tests, readouts, seeds),
            "by_class": {c: joint_accuracy([t for t in tests if t.edit_class == c], readouts, seeds)
                         for c in sorted({t.edit_class for t in tests})}}
