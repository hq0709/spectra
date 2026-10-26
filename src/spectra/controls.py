"""Controls (FINAL_PROPOSAL.md, Core Mechanism section 11) and the exact control properties of P2.

Controls verify properties; they are not a leaderboard. Any failed exact property stops the run.
- P2a copy-prior: C_c = 1/2 exactly.
- P2b a pathwise query-blind generator with shared, reset seeds: C_c = 1/2 exactly.
- P2c any system blind to the device bundle M with shared, reset seeds: JDA = 0 on every E2 endpoint.
- P2d a channel with no admitted contrast is undefined, not 1/2.
- P2e record-volume control: JDA = 0 on removal deletion and on mirror.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .types import E2Test


class ExactControlFailure(RuntimeError):
    """An exact control property failed: the implementation or the input contract is wrong. Stop the run."""


def _rng(*parts) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest(), 16) % (2 ** 32))


@dataclass(frozen=True)
class Output:
    """An abstract generated film for unit tests: the evaluator reads `state` deterministically."""

    state: Tuple
    tag: str = ""


class CopyPrior:
    name = "copy_prior"

    def generate(self, prior_state, condition, seed: int):
        return prior_state


class QueryBlind:
    """Depends on (prior, seed) only; sampler re-seeded per query, seeds shared across a prior's queries."""

    name = "query_blind"

    def __init__(self, noise: float = 1.0):
        self.noise = noise

    def generate(self, prior_state, condition, seed: int):
        r = _rng("qb", prior_state, seed)
        return tuple(float(v) + self.noise * float(r.normal()) for v in prior_state)


class BundleBlind:
    """Sees prior, interval and background but not the bundle: identical outputs in both arms."""

    name = "bundle_blind"

    def generate_arm(self, test: E2Test, arm: str, seed: int):
        r = _rng("bb", test.prior.study_id, test.delta_h, len(test.background), seed)
        return bool(r.random() < 0.5)


class RecordVolume:
    """State-level control: 'present iff the arm's stream has more rows'."""

    name = "record_volume"

    def readout(self, test: E2Test, arm: str):
        a, b = len(test.bundle_a), len(test.bundle_b)
        mine, other = (a, b) if arm == "A" else (b, a)
        present = mine > other
        if test.sigma == "presence":
            return present
        if test.sigma == "presence_side":
            return (present, "none") if not present else (present, "unassessed")
        if test.sigma == "type_pair":
            return (present, present)
        raise ValueError(test.sigma)


class ExpectedStateOracle:
    """State-level; returns g for each arm (JDA = 1; checks the scoring path)."""

    name = "oracle"

    def readout(self, test: E2Test, arm: str):
        return test.expected_a if arm == "A" else test.expected_b


class PopulationRate:
    """State-level E1 control: prior grade + development-set mean grade change for the query's Delta bin."""

    name = "population_rate"

    def __init__(self, mean_change_by_bin: Mapping[int, Mapping[str, float]], bins_h: Sequence[float]):
        self.mean_change_by_bin = mean_change_by_bin
        self.bins_h = list(bins_h)

    def bin_of(self, delta_h: float) -> int:
        """Bins [b0, b1), ..., [b_{k-1}, b_k]: the final bin includes its upper endpoint (180 d)."""
        if delta_h < self.bins_h[0] or delta_h > self.bins_h[-1]:
            raise ValueError(f"interval {delta_h} h outside the configured Delta bins")
        if delta_h == self.bins_h[-1]:
            return len(self.bins_h) - 2
        return int(np.searchsorted(self.bins_h, delta_h, side="right") - 1)

    def predict(self, prior_grades: Mapping[str, float], delta_h: float) -> Dict[str, float]:
        ch = self.mean_change_by_bin.get(self.bin_of(delta_h), {})
        return {c: float(g) + float(ch.get(c, 0.0)) for c, g in prior_grades.items()}


def assert_half(value: Optional[float], what: str, tol: float = 0.0):
    if value is None or abs(value - 0.5) > tol:
        raise ExactControlFailure(f"{what}: expected C = 1/2 exactly, got {value!r}")


def assert_zero(value: float, what: str):
    if value != 0.0:
        raise ExactControlFailure(f"{what}: expected JDA = 0 exactly, got {value!r}")


def assert_undefined(value, what: str):
    if value is not None:
        raise ExactControlFailure(f"{what}: a channel without admitted contrasts must be undefined, got {value!r}")
