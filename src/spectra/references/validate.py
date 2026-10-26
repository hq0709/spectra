"""Reference validation: point estimates with their own denominators and binomial SDs (no intervals)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from ..config import constants


def binomial_sd(k: int, n: int) -> float:
    """SD of the observed rate k / n, sqrt(p (1 - p) / n)."""
    if n == 0:
        return float("nan")
    p = k / n
    return math.sqrt(p * (1 - p) / n)


@dataclass
class Rate:
    k: int
    n: int

    @property
    def p(self) -> float:
        return self.k / self.n if self.n else float("nan")

    @property
    def sd(self) -> float:
        return binomial_sd(self.k, self.n)


def precision_recall(pred: Sequence, gold: Sequence, positive) -> Dict[str, Rate]:
    tp = sum(1 for p, g in zip(pred, gold) if p == positive and g == positive)
    fp = sum(1 for p, g in zip(pred, gold) if p == positive and g != positive)
    fn = sum(1 for p, g in zip(pred, gold) if p != positive and g == positive)
    return {"precision": Rate(tp, tp + fp), "recall": Rate(tp, tp + fn)}


def _rate(k: int, n: int, key: str, threshold: float, at_least: bool) -> Tuple[bool, dict]:
    p = k / n if n else float("nan")
    ok = n > 0 and (p >= threshold if at_least else p <= threshold)
    return ok, {key: p, "sd": binomial_sd(k, n), "n": n}


def reference_type_accepted(correct: int, n: int) -> Tuple[bool, dict]:
    return _rate(correct, n, "precision", constants()["reference_acceptance"]["precision"], True)


def tier2_accepted(correct: int, n: int) -> Tuple[bool, dict]:
    return _rate(correct, n, "precision", constants()["reference_acceptance"]["tier2_precision"], True)


def initial_state_accepted(errors: int, n: int) -> Tuple[bool, dict]:
    return _rate(errors, n, "error", constants()["reference_acceptance"]["initial_state_error"], False)


def link_validity_accepted(valid: int, n: int) -> Tuple[bool, dict]:
    return _rate(valid, n, "validity", constants()["reference_acceptance"]["link_validity"], True)
