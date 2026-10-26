"""Human-anchored estimates by prediction-powered inference (PPI / PPI++), FINAL_PROPOSAL.md section 8.

Two-stage audit sampling: stage 1 draws n_h patients by simple random sampling from the N_h test patients
of stratum h (fixed seed, independent of outcomes, the same patients for every audited system); stage 2
draws one unit (test or contrast) and one seed uniformly within each sampled patient.

For an audited patient i: H_i = adjudicated human score of the sampled unit, Z_i = automated score of the
same unit and seed, X_i = the patient's automated patient-macro mean over all units and seeds. X is known
for every pool patient. Estimator (audited patients are a subset of the pool):

    mu_h = lambda * Xbar_N + (1/n) * sum_{i in audit} (H_i - lambda * Z_i)

With f = n/N the variance-minimising lambda for this nested design is

    lambda* = (Cov(H, Z) - f Cov(H, X)) / (Var(Z) + f Var(X) - 2 f Cov(Z, X)),

clipped to [0, 1] (lambda = 1 is classical PPI; a zero denominator means the automated score carries no
information and lambda = 0). Spread: the standard deviation of the estimate over a stratified patient bootstrap,
resampling patients jointly in the pool and the audit sample (each audited patient keeps its single sampled unit)
and re-estimating lambda. Only the point estimate and this SD are reported; no interval is formed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Stratum:
    name: str
    weight: float
    pool: Dict[str, float]                                     # patient -> X_i (patient automated mean)
    audit: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # patient -> (H_i, Z_i)

    def __post_init__(self):
        missing = [p for p in self.audit if p not in self.pool]
        if missing:
            raise ValueError(f"stratum {self.name}: audited patients missing from the pool: {missing[:3]}")


def _rng(seed, *parts) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, (seed,) + parts)).encode()).hexdigest(), 16) % (2 ** 32))


def draw_patients(patients: Sequence[str], n: int, seed, stratum: str) -> List[str]:
    """Stage 1: simple random sample of n patients (fixed seed, independent of outcomes)."""
    pats = sorted(patients)
    rng = _rng(seed, "stage1", stratum)
    idx = rng.choice(len(pats), size=min(n, len(pats)), replace=False)
    return sorted(pats[i] for i in idx)


def draw_unit(patient: str, units: Sequence[str], seeds: Sequence[int], seed, stratum: str) -> Tuple[str, int]:
    """Stage 2: one unit and one seed uniformly within the sampled patient."""
    rng = _rng(seed, "stage2", stratum, patient)
    return sorted(units)[int(rng.integers(len(units)))], list(seeds)[int(rng.integers(len(seeds)))]


def ppi_lambda(h: np.ndarray, z: np.ndarray, x_aud: np.ndarray, x_pool: np.ndarray) -> float:
    """Power-tuned lambda for the nested two-stage design (see module docstring)."""
    n, N = len(h), len(x_pool)
    if n < 2 or N < 1:
        return 1.0
    f = n / N
    cov = lambda a, b: float(np.cov(a, b, ddof=1)[0, 1])
    num = cov(h, z) - f * cov(h, x_aud)
    den = float(np.var(z, ddof=1)) + f * float(np.var(x_pool, ddof=1) if N > 1 else 0.0) - 2 * f * cov(z, x_aud)
    if not np.isfinite(den) or den <= 1e-12:
        return 0.0
    return float(np.clip(num / den, 0.0, 1.0))


def _arrays(s: Stratum, patients: Optional[Sequence[str]] = None):
    pats = list(s.pool) if patients is None else list(patients)
    x_pool = np.array([s.pool[p] for p in pats], float)
    aud = [p for p in pats if p in s.audit]
    h = np.array([s.audit[p][0] for p in aud], float)
    z = np.array([s.audit[p][1] for p in aud], float)
    x_aud = np.array([s.pool[p] for p in aud], float)
    return x_pool, h, z, x_aud


def stratum_estimate(s: Stratum, lam: Optional[float] = None, patients: Optional[Sequence[str]] = None) -> Tuple[float, float]:
    x_pool, h, z, x_aud = _arrays(s, patients)
    if len(h) == 0:
        raise ValueError(f"stratum {s.name} has no audited patient")
    lam = ppi_lambda(h, z, x_aud, x_pool) if lam is None else lam
    return float(lam * x_pool.mean() + np.mean(h - lam * z)), lam


def estimate(strata: Sequence[Stratum], lam: Optional[float] = None) -> dict:
    w = np.array([s.weight for s in strata], float)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("stratum weights must sum to 1")
    per = {s.name: stratum_estimate(s, lam) for s in strata}
    mu = float(sum(s.weight * per[s.name][0] for s in strata))
    return {"estimate": mu, "strata": {k: {"estimate": v[0], "lambda": v[1]} for k, v in per.items()}}


def bootstrap(strata: Sequence[Stratum], replicates: int = 2000, seed=0, lam: Optional[float] = None) -> dict:
    """Point estimate and its SD over a stratified patient bootstrap (jointly over pool and audit sample,
    re-estimating lambda each time). No interval."""
    rng = _rng(seed, "bootstrap")
    point = estimate(strata, lam)["estimate"]
    vals = []
    for _ in range(replicates):
        total, ok = 0.0, True
        for s in strata:
            pats = list(s.pool)
            draw = [pats[i] for i in rng.integers(len(pats), size=len(pats))]
            x_pool = np.array([s.pool[p] for p in draw], float)
            aud = [p for p in draw if p in s.audit]
            if len(aud) < 2:
                ok = False
                break
            h = np.array([s.audit[p][0] for p in aud], float)
            z = np.array([s.audit[p][1] for p in aud], float)
            x_aud = np.array([s.pool[p] for p in aud], float)
            l = ppi_lambda(h, z, x_aud, x_pool) if lam is None else lam
            total += s.weight * (l * x_pool.mean() + np.mean(h - l * z))
        if ok:
            vals.append(total)
    vals = np.array(vals)
    se = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
    return {"estimate": point, "se": se, "replicates_used": int(len(vals))}


def automated_only(pool: Mapping[str, float]) -> float:
    return float(np.mean(list(pool.values())))
