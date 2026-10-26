"""Attrition per rule and per output (patients, priors, queries, contrasts / tests)."""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, List

import pandas as pd


@dataclass
class Funnel:
    name: str
    steps: "OrderedDict[str, dict]" = field(default_factory=OrderedDict)
    rejections: Counter = field(default_factory=Counter)

    def step(self, rule: str, patients: int, priors: int = None, queries: int = None, units: int = None):
        self.steps[rule] = {"patients": patients, "priors": priors, "queries": queries, "units": units}

    def reject(self, rule: str, n: int = 1):
        self.rejections[rule] += n

    def add_rejections(self, rejs: Iterable):
        for r in rejs:
            self.rejections[f"{r.edit_class}:{r.rule}"] += 1

    def table(self) -> pd.DataFrame:
        rows = [{"funnel": self.name, "rule": k, **v} for k, v in self.steps.items()]
        rows += [{"funnel": self.name, "rule": f"rejected:{k}", "units": n} for k, n in sorted(self.rejections.items())]
        return pd.DataFrame(rows)
