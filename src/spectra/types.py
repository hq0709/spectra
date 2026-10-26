"""Records shared by every SpecTra module.

Times inside a query window are hours after the prior film (t0). A canonical event
record (CER) with ``t_start_h is None`` is a state that was already active at t0 and is
carried forward; every other CER starts inside the window (t0, tk].
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

DEVICE_TYPES = ("ETT", "CVC", "PICC", "enteric", "chest_tube", "none")
TRANSITIONS = ("place", "remove", "none")
SITES = ("IJ", "subclavian", "femoral", "brachial/basilic/cephalic", "nasal/oral", "pleural", "other", "none")
SIDES = ("L", "R", "none")
KINDS = ("state", "transition", "action", "observation")
PARTS = ("recorded", "background", "bundle")
TRACKS = ("delta", "A", "B")
EDIT_CLASSES = (
    "placement_deletion",
    "removal_deletion",
    "mirror",
    "insertion",
    "type_control",
    "removal_specification",
)
PRIMARY_EDIT_CLASSES = ("placement_deletion", "removal_deletion")
PRIMARY_DEVICES = ("ETT", "CVC")


@dataclass(frozen=True)
class Device:
    type: str = "none"
    transition: str = "none"
    site: str = "none"
    side: str = "none"
    subtype: str = ""  # source label, e.g. "Multi Lumen"; used by the single-instance rule

    def __post_init__(self):
        if self.type not in DEVICE_TYPES:
            raise ValueError(f"unknown device type {self.type!r}")
        if self.transition not in TRANSITIONS:
            raise ValueError(f"unknown transition {self.transition!r}")
        if self.site not in SITES:
            raise ValueError(f"unknown site {self.site!r}")
        if self.side not in SIDES:
            raise ValueError(f"unknown side {self.side!r}")


NO_DEVICE = Device()


@dataclass(frozen=True)
class CER:
    """Canonical event record."""

    kind: str
    action_type: str
    t_start_h: Optional[float]
    t_end_h: Optional[float] = None
    ongoing_at_delta: bool = False
    device: Device = NO_DEVICE
    dose_value: Optional[float] = None
    dose_unit: Optional[str] = None
    value: Optional[str] = None  # charted setting value, e.g. "CMV/ASSIST" or "5"
    source_itemids: Tuple[int, ...] = ()
    priority: str = "other_medication"
    track: str = "A"
    part: str = "recorded"
    candidate: bool = False  # transition inferred from an interval endpoint; needs report corroboration
    origin: str = "record"  # record | edit (added by a specification edit)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown CER kind {self.kind!r}")
        if self.part not in PARTS:
            raise ValueError(f"unknown CER part {self.part!r}")
        if self.track not in ("A", "B"):
            raise ValueError(f"unknown CER track {self.track!r}")

    @property
    def is_state(self) -> bool:
        return self.t_start_h is None

    def with_part(self, part: str) -> "CER":
        return replace(self, part=part)

    def sort_key(self):
        start = -1.0 if self.t_start_h is None else self.t_start_h
        return (start, self.kind != "state", self.action_type, self.source_itemids, self.value or "")


@dataclass(frozen=True)
class Film:
    """One frontal image per study (rule F1)."""

    patient_id: str
    study_id: str
    image_id: str
    t_h: float  # hours since a per-patient anchor (MIMIC: shifted clock; TAIX-Ray: Age in days x 24)
    day: int  # calendar day index on the same clock (used by comparison-reference matching)
    clock_min: Optional[int] = None  # minutes after midnight when the source keeps time of day
    stay_id: Optional[str] = None
    view: str = ""
    split: str = ""
    source: str = ""


@dataclass(frozen=True)
class DeviceState:
    present: bool
    side: str = "none"  # L | R | none (unknown or unsided)

    def in_space(self, sigma: str):
        if sigma == "presence":
            return self.present
        if sigma == "presence_side":
            return (self.present, self.side if self.present else "none")
        raise ValueError(f"unknown state space {sigma!r}")


@dataclass(frozen=True)
class Ordering:
    """Reference ordering between two films for one channel (references section of the proposal)."""

    earlier_study: str
    later_study: str
    channel: str  # edema | effusion_L | effusion_R
    direction: int  # -1 improved, 0 unchanged, +1 worsened (sign of later - earlier)
    provenance: str  # comparison_explicit | comparison_default | absolute_grades
    evidence: str = ""

    def __post_init__(self):
        if self.direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0 or +1")


@dataclass(frozen=True)
class E1Query:
    query_id: str
    patient_id: str
    prior: Film
    target: Film
    delta_h: float
    track: str
    stream: Tuple[CER, ...] = ()
    actions_known: bool = True


@dataclass(frozen=True)
class E1Contrast:
    prior_study: str
    query_j: str
    query_k: str
    channel: str
    rho: int  # admitted ordering of k relative to j, +1 or -1


@dataclass(frozen=True)
class E2Test:
    """Device specification test tau = (x0, d, s0, B, M_A, M_B, Delta)."""

    test_id: str
    patient_id: str
    prior: Film
    device: str
    edit_class: str
    s0: DeviceState
    background: Tuple[CER, ...]
    bundle_a: Tuple[CER, ...]
    bundle_b: Tuple[CER, ...]
    delta_h: float
    sigma: str  # presence | presence_side | type_pair
    expected_a: object
    expected_b: object
    logged_arm: Optional[str] = None  # "A" when arm A is the recorded transition
    second_device: Optional[str] = None  # type-control partner device
    actions_known: bool = True  # False for the cross-source protocol (background = actions unknown)
    t_transition_h: Optional[float] = None  # logged transition time (cohort ordering)
    source: str = ""
    meta: dict = field(default_factory=dict, compare=False, hash=False)
