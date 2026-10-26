"""Device initial states and transitions from report evidence (FINAL_PROPOSAL.md, Core Mechanism section 6).

Tier 1: an explicit statement in the prior's report, or an explicit comparison statement in the next
resolved report ("interval placement" -> absent at the prior; "removed" -> present at the prior).
Tier 2 (admitted per state type only after its audit passes): an explicit device enumeration in the report
that omits the target, plus no active target rows in the records. Records-only evidence never suffices.
"""
from __future__ import annotations

from typing import Mapping, Optional, Tuple

from ..types import DeviceState

SIDE = {"left": "L", "right": "R", "none": "none"}


def initial_state(device: str, prior_report: Optional[Mapping], next_report: Optional[Mapping],
                  records_active: bool, tier2_admitted: bool = False,
                  next_links_prior: bool = False) -> Tuple[Optional[DeviceState], Optional[str]]:
    """Returns (state, tier) or (None, None) when the initial state is not verified. The next report's
    comparison statement is used only when its resolved comparison targets this prior (next_links_prior)."""
    if prior_report is not None:
        node = prior_report.get("devices", {}).get(device, {})
        st = node.get("state", "not_mentioned")
        if st in ("present", "placed_interval", "repositioned"):
            return DeviceState(True, SIDE.get(node.get("side", "none"), "none")), "tier1"
        if st in ("absent", "removed_interval"):
            return DeviceState(False), "tier1"
    if next_report is not None and next_links_prior:
        node = next_report.get("devices", {}).get(device, {})
        st = node.get("state", "not_mentioned")
        if st == "placed_interval":
            return DeviceState(False), "tier1_next"
        if st == "removed_interval":
            return DeviceState(True, SIDE.get(node.get("side", "none"), "none")), "tier1_next"
    if tier2_admitted and prior_report is not None and prior_report.get("device_enumeration") \
            and prior_report.get("devices", {}).get(device, {}).get("state", "not_mentioned") == "not_mentioned" \
            and not records_active:
        return DeviceState(False), "tier2"
    return None, None


def follow_up_state(device: str, follow_report: Optional[Mapping]) -> Optional[DeviceState]:
    """Device state stated by the follow-up report (corroboration of a logged transition)."""
    if follow_report is None:
        return None
    node = follow_report.get("devices", {}).get(device, {})
    st = node.get("state", "not_mentioned")
    if st in ("present", "placed_interval", "repositioned"):
        return DeviceState(True, SIDE.get(node.get("side", "none"), "none"))
    if st in ("absent", "removed_interval"):
        return DeviceState(False)
    return None
