"""Confirmatory decision table (FINAL_PROPOSAL.md, Core Mechanism section 12).

Familywise alpha = 0.05, Holm over the elementary hypotheses of the reader-budget rung in force
(Rung 0: H1, H2, H5; Rung 1: H1-H7; Rungs 2-5: H1-H9), fixed before any Gate-4 confirmatory outcome.
A declaration requires a Holm-adjusted p <= alpha; its direction is the sign of the estimate relative to
the null value. Holm is applied to p-values collected from both machines (aggregates only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np
from scipy.stats import norm

from .config import constants

NULLS = {"H1": 0.02, "H2": 0.02, "H3": 0.90, "H4": 0.90, "H5": 0.0, "H6": 0.90, "H7": 0.90, "H8": 0.90, "H9": 0.90}
ONE_SIDED = {"H5"}


@dataclass(frozen=True)
class Elementary:
    hid: str
    estimate: float
    se: float

    def estimable(self) -> bool:
        return self.se is not None and np.isfinite(self.se) and self.se > 0 and np.isfinite(self.estimate)

    def p_value(self) -> float:
        """NaN when the uncertainty cannot be estimated (zero or non-finite SE): never declarable."""
        if not self.estimable():
            return float("nan")
        z = (self.estimate - NULLS[self.hid]) / self.se
        if self.hid in ONE_SIDED:
            return float(1.0 - norm.cdf(z))
        return float(2.0 * (1.0 - norm.cdf(abs(z))))

    def direction(self) -> int:
        d = self.estimate - NULLS[self.hid]
        return (d > 0) - (d < 0)


def holm(pvals: Mapping[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values; a non-estimable (NaN) p counts as 1 in the family."""
    pvals = {k: (1.0 if v is None or not np.isfinite(v) else v) for k, v in pvals.items()}
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for r, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - r) * p))
        out[k] = running
    return out


def decide(results: Mapping[str, Elementary], rung: int, alpha: Optional[float] = None) -> dict:
    k = constants()["inference"]
    alpha = k["alpha"] if alpha is None else alpha
    family = list(k["holm_families"][rung])
    missing = [h for h in family if h not in results]
    if missing:
        raise ValueError(f"rung {rung} family needs {missing}")
    p = {h: results[h].p_value() for h in family}
    adj = holm(p)
    decl = {}
    for h in family:
        e = results[h]
        if not e.estimable():
            decl[h] = "not declarable (uncertainty not estimable)"
            continue
        if adj[h] > alpha:
            decl[h] = "not declared"
            continue
        d = e.direction()
        if h in ("H1", "H2"):
            decl[h] = "exceeds prior+delta by more than the margin" if d > 0 else "does not exceed prior+delta by the margin"
        elif h == "H5":
            decl[h] = "S2 beats copy-prior on conventional AUROC"
        else:
            decl[h] = "substantial specification failure" if d < 0 else "meets the criterion"
    claims = {}
    failure = lambda h: decl.get(h) == "substantial specification failure"
    claims["divergence"] = decl.get("H5") == "S2 beats copy-prior on conventional AUROC" and failure("H4")
    if rung >= 1:
        claims["padchest_failure"] = failure("H6") or failure("H7")
    if rung >= 2:
        claims["recurs_on_both_sources"] = (failure("H6") and failure("H8")) or (failure("H7") and failure("H9"))
    return {"rung": rung, "family": family, "p": p, "holm": adj, "declarations": decl, "claims": claims}
