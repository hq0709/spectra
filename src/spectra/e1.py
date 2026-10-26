"""E1: within-prior concordance on real logged trajectories (FINAL_PROPOSAL.md, Core Mechanism section 2).

Q(x0) = the next <=3 consecutive frontal films after the prior x0 within horizon H, all queried in mode R.
For channel c and admitted ordering rho_c(j,k) in {+1,-1} between two later films j < k of Q(x0):

    kappa^(s)_c(j,k) = 1[sgn(phi_c(xhat_k) - phi_c(xhat_j)) = rho] + 1/2 * 1[phi_c(xhat_k) = phi_c(xhat_j)]   (1)
    C_{i,c} = mean_s mean_{(j,k)} kappa,   C_c = patient-macro mean over patients with >= 1 contrast    (2)

A channel without an admitted contrast is undefined (never 1/2).
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import constants
from .types import E1Contrast, Film, Ordering


@dataclass(frozen=True)
class QuerySet:
    prior: Film
    later: Tuple[Film, ...]
    group: str  # stay id (MIMIC) or patient id (external sources): the unit of the <=3-priors cap


def query_sets(films: Sequence[Film], min_h: float, max_h: float, max_later: int = 3,
               same_stay: bool = True) -> List[QuerySet]:
    """For every prior, the next <= max_later consecutive films in acquisition order, all with an interval in
    [min_h, max_h] (and in the same stay when required); the set stops at the first film outside the
    horizon, as in the authoritative TAIX-Ray adapter. Films must already be one per study (rule F1) or one
    per patient-day (TAIX-Ray)."""
    by_patient = defaultdict(list)
    for f in films:
        by_patient[f.patient_id].append(f)
    out: List[QuerySet] = []
    for pid, fs in by_patient.items():
        fs = sorted(fs, key=lambda f: (f.t_h, f.study_id))
        for i, x0 in enumerate(fs):
            later = []
            for f in fs[i + 1:i + 1 + max_later]:
                if same_stay and (x0.stay_id is None or f.stay_id != x0.stay_id):
                    break
                if not (min_h <= f.t_h - x0.t_h <= max_h):
                    break
                later.append(f)
            if later:
                out.append(QuerySet(x0, tuple(later), x0.stay_id if same_stay else pid))
    return out


def admitted_contrasts(qs: QuerySet, orderings: Mapping[Tuple[str, str, str], Ordering],
                       channels: Sequence[str]) -> List[E1Contrast]:
    """Pairs (j, k), j earlier than k, of later films of one prior with an admitted non-tied ordering."""
    out = []
    films = sorted(qs.later, key=lambda f: (f.t_h, f.study_id))
    for a in range(len(films)):
        for b in range(a + 1, len(films)):
            fj, fk = films[a], films[b]
            for c in channels:
                o = orderings.get((fj.study_id, fk.study_id, c))
                if o is None:
                    rev = orderings.get((fk.study_id, fj.study_id, c))
                    if rev is not None:
                        o = Ordering(fj.study_id, fk.study_id, c, -rev.direction, rev.provenance, rev.evidence)
                if o is None or o.direction == 0:
                    continue
                out.append(E1Contrast(qs.prior.study_id, query_id(qs.prior, fj), query_id(qs.prior, fk), c, o.direction))
    return out


def query_id(prior: Film, target: Film) -> str:
    return f"{prior.study_id}->{target.study_id}"


def sample_priors(sets_with_contrasts: Sequence[Tuple[QuerySet, List[E1Contrast]]], max_per_group: int,
                  seed, key=None) -> List[Tuple[QuerySet, List[E1Contrast]]]:
    """<= max_per_group priors per stay (per patient externally), drawn with a fixed seed among priors with
    >= 1 admitted contrast: candidates are ordered by sha256(seed|group|key) and the first ones kept
    (the scheme of the authoritative TAIX-Ray adapter; key defaults to the prior's study id)."""
    key = key or (lambda qs: qs.prior.study_id)
    by_group = defaultdict(list)
    for qs, cons in sets_with_contrasts:
        if cons:
            by_group[qs.group].append((qs, cons))
    out = []
    for g, items in sorted(by_group.items(), key=lambda kv: str(kv[0])):
        h = lambda it: hashlib.sha256(f"{seed}|{g}|{key(it[0])}".encode()).hexdigest()
        keep = sorted(items, key=h)[:max_per_group]
        out += sorted(keep, key=lambda it: (it[0].prior.t_h, it[0].prior.study_id))
    return out


# ---------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------
def kappa(phi_j: float, phi_k: float, rho: int) -> float:
    if phi_j is None or phi_k is None:
        raise ValueError("an unreadable output must be resolved by the failure rule before scoring")
    if phi_k == phi_j:
        return 0.5
    return 1.0 if np.sign(phi_k - phi_j) == rho else 0.0


def concordance(contrasts: Sequence[Tuple[str, E1Contrast]], readouts: Mapping[Tuple[str, int], Mapping[str, float]],
                seeds: Sequence[int], pooled: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, dict]:
    """contrasts: (patient_id, contrast). readouts[(query_id, seed)][channel] = phi_c of that output.
    Returns per channel (and per pooled channel) the patient-macro C_c, its patient values and N_c."""
    pooled = constants()["e1"]["pooled_channels"] if pooled is None else pooled
    per = defaultdict(lambda: defaultdict(list))  # channel -> patient -> kappas (one per contrast, seed-averaged)
    for pid, con in contrasts:
        ks = [kappa(readouts[(con.query_j, s)][con.channel], readouts[(con.query_k, s)][con.channel], con.rho)
              for s in seeds]
        per[con.channel][pid].append(float(np.mean(ks)))
        for pname, members in pooled.items():
            if con.channel in members:
                per[pname][pid].append(float(np.mean(ks)))
    out = {}
    for ch, pats in per.items():
        vals = {p: float(np.mean(v)) for p, v in pats.items()}
        out[ch] = {"C": float(np.mean(list(vals.values()))), "N": len(vals), "patients": vals}
    return out


def channel_value(result: Mapping[str, dict], channel: str) -> Optional[float]:
    """C_c, or None when the channel has no admitted contrast (undefined, not 1/2)."""
    r = result.get(channel)
    return None if r is None or r["N"] == 0 else r["C"]


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney AUROC with ties counted 1/2."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def conventional_auroc(query_ids: Sequence[str], readouts: Mapping[Tuple[str, int], Mapping[str, float]],
                       labels: Mapping[str, int], channel: str, seeds: Sequence[int]) -> float:
    """The conventional factual score: AUROC across independently scored follow-up queries, the readout of
    each generated follow-up (seed-averaged) against the reference presence in the real follow-up."""
    xs, ys = [], []
    for q in query_ids:
        if q not in labels:
            continue
        xs.append(float(np.mean([readouts[(q, s)][channel] for s in seeds])))
        ys.append(int(labels[q]))
    return auroc(xs, ys)


def absolute_error(query_ids, readouts, grades: Mapping[str, float], channel: str, seeds,
                   patient_of: Mapping[str, str]) -> float:
    """AE_c, patient-macro: |phi_c(generated sample) - reference grade of the real follow-up| per sample,
    averaged over seeds and queries within a patient, then over patients."""
    per = defaultdict(list)
    for q in query_ids:
        if q not in grades:
            continue
        for s in seeds:
            per[patient_of[q]].append(abs(float(readouts[(q, s)][channel]) - float(grades[q])))
    return float(np.mean([np.mean(v) for v in per.values()])) if per else float("nan")
