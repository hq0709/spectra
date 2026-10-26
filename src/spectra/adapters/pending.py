"""Adapters whose data access is pending (Gate 0, R005). Each declares what it will provide under the
source contract; the implementations land at Gate 1 when the data use agreement is in place."""
from __future__ import annotations

from .base import SourceAdapter


class _Pending(SourceAdapter):
    access = ""

    def films(self):
        raise NotImplementedError(f"{self.name}: data access pending ({self.access}); implemented at Gate 1")


class ReXGradientAdapter(_Pending):
    name, time_resolution, has_reports = "rexgradient", "day", True
    access = "Hugging Face gate rajpurkarlab/ReXGradient-160K (non-commercial DUA)"


class PadChestAdapter(_Pending):
    name, time_resolution, has_reports = "padchest", "day", True
    access = "BIMCV PadChest request form (PADCHEST Research Use Agreement)"


class MIDRCAdapter(_Pending):
    name, time_resolution, has_reports, has_actions = "midrc", "day", True, True
    access = "MIDRC registration + DUA"


class BRAXAdapter(_Pending):
    name, time_resolution = "brax", "day"
    access = "PhysioNet credentialed (BRAX)"


class CheXpertPlusAdapter(_Pending):
    name, time_resolution, has_reports = "chexpert_plus", "order", True
    access = "Stanford AIMI research use agreement"


class NIHAdapter(_Pending):
    name, time_resolution = "nih", "order"
    access = "open (NIH ChestX-ray14); staged locally at Gate 1"
