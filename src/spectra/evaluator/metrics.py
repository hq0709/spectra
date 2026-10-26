"""Acceptance metrics of the evaluator (FINAL_PROPOSAL.md section 12, step E-b): point estimates decide; spreads are
standard deviations, never confidence intervals."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def discretize(phi, n_classes: int) -> np.ndarray:
    """A film's grade from its expected grade: rounded half up, clipped to 0..n_classes-1 (constants.yaml)."""
    return np.clip(np.floor(np.asarray(phi, float) + 0.5), 0, n_classes - 1).astype(int)


def weighted_kappa(a, b, n_classes: int, weights: str = "quadratic") -> float:
    """Cohen's weighted kappa between two integer gradings 0..n_classes-1."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    k = n_classes
    o = np.bincount(a * k + b, minlength=k * k).reshape(k, k).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / max(o.sum(), 1.0)
    i, j = np.indices((k, k))
    w = (i - j) ** 2 / (k - 1) ** 2 if weights == "quadratic" else np.abs(i - j) / (k - 1)
    den = (w * e).sum()
    return float("nan") if den == 0 else float(1.0 - (w * o).sum() / den)


def patient_mean_sd(values) -> dict:
    """Mean and standard deviation (ddof = 1) of per-patient values; no confidence intervals (user rule 2026-10-25)."""
    v = np.asarray(list(values), float)
    return {"mean": float(v.mean()) if len(v) else float("nan"),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "N": int(len(v))}


def _point(value: float, patients) -> dict:
    return {"estimate": float(value), "n_patients": int(len(set(np.asarray(patients).tolist()))),
            "n_rows": int(len(patients))}


def ordinal_report(phi, grade, patients, n_classes: int, qwk_min: float) -> dict:
    """E-b / E-c metrics of one ordinal channel on held-out real films; the QWK point estimate decides."""
    from scipy.stats import spearmanr

    from ..e1 import auroc

    phi, grade = np.asarray(phi, float), np.asarray(grade, int)
    d = discretize(phi, n_classes)
    out = {"qwk": _point(weighted_kappa(d, grade, n_classes, "quadratic"), patients),
           "linear_kappa": weighted_kappa(d, grade, n_classes, "linear"),
           "exact_accuracy": float((d == grade).mean()), "mae_expected_grade": float(np.abs(phi - grade).mean()),
           "spearman": float(spearmanr(phi, grade).statistic),
           "auroc_grade_ge1": auroc(phi, (grade >= 1).astype(int)), "auroc_grade_ge2": auroc(phi, (grade >= 2).astype(int)),
           "grade_distribution": np.bincount(grade, minlength=n_classes).tolist(),
           "predicted_distribution": np.bincount(d, minlength=n_classes).tolist(),
           "confusion_pred_by_true": np.bincount(d * n_classes + grade, minlength=n_classes ** 2).reshape(
               n_classes, n_classes).tolist()}
    out["criterion"] = {"qwk_min": qwk_min, "pass": out["qwk"]["estimate"] >= qwk_min}
    return out


def rank_auroc(scores, labels) -> float:
    """Mann-Whitney AUROC with ties counted 1/2 (equal to spectra.e1.auroc), from average ranks: O(n log n)."""
    from scipy.stats import rankdata

    s, y = np.asarray(scores, float), np.asarray(labels, int)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def presence_report(psi, present, patients, auroc_min: Optional[float]) -> dict:
    """E-b metrics of one presence head on held-out real films: AUROC of psi = P(present) (the point estimate decides
    when the head has a section-12 criterion; auxiliary heads have none)."""
    psi, present = np.asarray(psi, float), np.asarray(present, int)
    out = {"auroc": _point(rank_auroc(psi, present), patients),
           "prevalence": float(present.mean()), "n_present": int(present.sum()), "n_absent": int((1 - present).sum())}
    out["criterion"] = None if auroc_min is None else {"auroc_min": auroc_min,
                                                       "pass": out["auroc"]["estimate"] >= auroc_min}
    return out
