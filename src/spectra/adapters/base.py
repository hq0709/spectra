"""Source contract (FINAL_PROPOSAL.md, Core Mechanism section 10).

A source enters SpecTra through an adapter that provides (i) patient identifiers and frontal films with
acquisition times at minute or day resolution (order-only sources enter a secondary order-only track
without Delta); (ii) per-film reports or graded findings from which references are extracted and
validated; and optionally (iii) a typed action log with occurrence and availability times plus a device
dictionary (placement and removal codes, dependency table, site and side fields) and (iv) intervening
observations. E1 needs (i)-(ii); logged-transition E2 needs (i)-(iii); the cross-source specification
protocol needs (i) and device-state evidence from (ii). An unavailable log is never encoded as an empty one.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Optional

from ..types import Film


class SourceAdapter(ABC):
    name: str = "source"
    time_resolution: str = "minute"   # minute | day | order
    has_reports: bool = False
    has_grades: bool = False
    has_actions: bool = False         # (iii) typed action log
    has_observations: bool = False    # (iv) intervening observations

    @abstractmethod
    def films(self) -> List[Film]:
        """(i) one frontal film per study (rule F1) or per patient-day, with split assigned."""

    def reports(self) -> Mapping[str, str]:
        """(ii) study id -> report text."""
        raise NotImplementedError(f"{self.name} has no reports")

    def grades(self) -> Mapping[str, Mapping[str, Optional[int]]]:
        """(ii) study id -> channel -> ordinal grade (None = not assessable)."""
        raise NotImplementedError(f"{self.name} has no graded findings")

    def window_stream(self, prior: Film, target: Film, track: str = "A"):
        """(iii)/(iv) compiled CER stream of the window (prior.t, target.t]."""
        raise NotImplementedError(f"{self.name} has no action log")

    def describe(self) -> Dict[str, object]:
        return {"source": self.name, "time_resolution": self.time_resolution, "reports": self.has_reports,
                "grades": self.has_grades, "actions": self.has_actions, "observations": self.has_observations}
