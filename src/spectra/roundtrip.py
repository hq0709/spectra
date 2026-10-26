"""Mandatory round-trip check (every E2 test, every system), after the system's actual tokenization:
(i) parse(arm A) and parse(arm B) return g^A and g^B; (ii) the background segment is token-identical in
both arms; (iii) the arms differ only inside the bundle segment. Any failure blocks E2 scoring of that
system (FINAL_PROPOSAL.md, Core Mechanism section 4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .config import constants
from .e2 import expected_state
from .serialize import (Serialized, bundle_segment, decode_typed, parse_compact_bundle, parse_device_events,
                        serialize_mode_r_style, serialize_mode_s, shared_background, token_budget_background)
from .types import E2Test


@dataclass
class Interface:
    """How one evaluated configuration receives its inputs."""

    system: str
    style: str  # verbal | event_text | compact | typed
    tokenizer: Optional[object] = None
    capacity_events: Optional[int] = None
    template: str = "{input}"  # the system's prompt template (config.template); the check runs on the result

    def __post_init__(self):
        if self.capacity_events is None:
            self.capacity_events = constants()["capacity"]["events_llm"]
        if self.template.count("{input}") != 1:
            raise ValueError("a template has exactly one {input} slot")

    def wrap_text(self, text: str) -> str:
        pre, post = self.template.split("{input}")
        return pre + text + post

    def _wrap(self, s: Serialized) -> Serialized:
        """Templated input: the template text around the serialized input, segment spans shifted."""
        if self.style == "typed" or self.template == "{input}":
            return s
        pre = self.template.split("{input}")[0]
        return Serialized(self.wrap_text(s.text), {k: (a + len(pre), b + len(pre)) for k, (a, b) in s.segments.items()},
                          s.tokens)

    def background(self, test: E2Test):
        """The shared B' of a test for this configuration (the same selection feeds generation and the
        round-trip check). None when Delta + M cannot be carried: the configuration is rejected for E2."""
        if self.style != "compact" and max(len(test.bundle_a), len(test.bundle_b)) > self.capacity_events:
            return None  # an event-limited interface cannot carry the mandatory bundle
        lim = getattr(self.tokenizer, "max_len", None) if self.tokenizer is not None else None
        if self.style == "compact" and lim is not None:
            return token_budget_background(test, self.style, self.tokenizer, lim, self.capacity_events, self.wrap_text)
        return shared_background(test, self.capacity_events)

    def mode_r_inputs(self, stream, delta_h: float, actions_known: bool = True):
        """Mode-R input of one E1 query, exactly as the system receives it."""
        lim = getattr(self.tokenizer, "max_len", None) if self.tokenizer is not None else None
        return self._wrap(serialize_mode_r_style(stream, delta_h, self.style, actions_known, self.capacity_events,
                                                 self.tokenizer, lim, self.wrap_text))

    def inputs(self, test: E2Test, arm: str):
        """Mode-S input of one arm, exactly as the system receives it."""
        b = self.background(test)
        if b is None:
            raise ValueError(f"{self.system} cannot carry Delta + M for test {test.test_id}")
        return self._wrap(serialize_mode_s(test, arm, self.style, self.capacity_events, b))


@dataclass
class RoundTripResult:
    system: str
    test_id: str
    ok: bool
    reasons: List[str] = field(default_factory=list)


def _space(test: E2Test, parsed, arm: str):
    if test.sigma == "type_pair":
        s1 = expected_state(test.s0, parsed, test.device)
        s2 = expected_state(test.s0, parsed, test.second_device)
        if s1 is None or s2 is None:
            return None
        return (s1.present, s2.present)
    g = expected_state(test.s0, parsed, test.device)
    return None if g is None else g.in_space(test.sigma)


def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _common_suffix(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        n += 1
    return n


def check(test: E2Test, iface: Interface) -> RoundTripResult:
    """Runs on the exact inputs the system receives (Interface.inputs), after its actual tokenization."""
    res = RoundTripResult(iface.system, test.test_id, True)
    fail = lambda why: (res.reasons.append(why), setattr(res, "ok", False))
    if iface.background(test) is None:
        fail("cannot_carry_delta_and_bundle")
        return res
    sa, sb = iface.inputs(test, "A"), iface.inputs(test, "B")

    if iface.style == "typed":
        da, bga, bun_a = decode_typed(sa.tokens or [])
        db, bgb, bun_b = decode_typed(sb.tokens or [])
        if da is None or db is None or abs(da - test.delta_h) > 0.05 or abs(db - test.delta_h) > 0.05:
            fail("delta_not_recovered")
        if len(sa.tokens or []) - 1 > iface.capacity_events or len(sb.tokens or []) - 1 > iface.capacity_events:
            fail("capacity_exceeded")
        if bga != bgb:
            fail("background_not_identical")
        outside = lambda toks: [t for t in toks if not (isinstance(t, tuple) and t and t[0] == "bundle")]
        if outside(sa.tokens or []) != outside(sb.tokens or []):
            fail("tokens_outside_bundle_differ")
        for arm, parsed, exp in (("A", bun_a, test.expected_a), ("B", bun_b, test.expected_b)):
            if _space(test, parsed, arm) != exp:
                fail(f"state_not_recovered_{arm}")
        return res

    tok = iface.tokenizer
    if tok is None:
        raise ValueError(f"{iface.system}: a text interface needs a tokenizer")
    ids_a, ids_b = tok.encode(sa.text), tok.encode(sb.text)
    lim = getattr(tok, "max_len", None)
    if lim is not None:
        if len(ids_a) > lim or len(ids_b) > lim:
            fail("truncated")
        ids_a, ids_b = ids_a[:lim], ids_b[:lim]
    for arm, ids, exp in (("A", ids_a, test.expected_a), ("B", ids_b, test.expected_b)):
        dec = tok.decode(ids)
        seg = bundle_segment(dec, iface.style, test.device, test.second_device)
        if seg is None:
            fail(f"bundle_delimiter_lost_{arm}")
            continue
        parsed = parse_compact_bundle(seg) if iface.style == "compact" else parse_device_events(seg)
        if _space(test, parsed, arm) != exp:
            fail(f"state_not_recovered_{arm}")
    # (ii) background token-identical: the shared prefix (Delta + B') encodes identically in both arms
    bg_end = sa.segments["background"][1]
    if sa.text[:bg_end] != sb.text[:sb.segments["background"][1]]:
        fail("background_text_differs")
    else:
        prefix = sa.text[:bg_end]
        pre_ids = tok.encode(prefix, add_special=False)
        need = getattr(tok, "bos_len", 0) + len(pre_ids)
        common = _common_prefix(ids_a, ids_b)
        if common < need:
            # tolerate only a merge of the background's last token with the delimiter, never a content change
            boundary_merge = (common == need - 1 and tok.decode(ids_a[:need]).startswith(prefix)
                              and tok.decode(ids_b[:need]).startswith(prefix))
            if not boundary_merge:
                fail("background_tokens_differ")
    # (iii) the arms differ only inside the bundle: nothing follows the bundle except identical text
    if sa.text[sa.segments["bundle"][1]:] != sb.text[sb.segments["bundle"][1]:]:
        fail("text_outside_bundle_differs")
    return res


def check_all(tests: Sequence[E2Test], interfaces: Sequence[Interface]) -> Dict[str, List[RoundTripResult]]:
    """Round-trip every test for every interface; a system with any failure is blocked from E2 scoring."""
    out: Dict[str, List[RoundTripResult]] = {}
    for iface in interfaces:
        out[iface.system] = [check(t, iface) for t in tests]
    return out


def blocked_systems(results: Dict[str, List[RoundTripResult]]) -> Dict[str, List[str]]:
    return {s: sorted({why for r in rs for why in r.reasons}) for s, rs in results.items() if any(not r.ok for r in rs)}
