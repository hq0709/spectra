"""R003: round-trip and background-equality check per interface tokenizer on synthetic E2 tests."""
import os
from dataclasses import replace

import pytest

from spectra.adapters.synthetic import ABSENT, PRESENT, cvc_placement_stream, ett_placement_stream, prior_film
from spectra.config import systems, template
from spectra.e2 import logged_tests
from spectra.roundtrip import Interface, blocked_systems, check, check_all
from spectra.serialize import (UNKNOWN_ACTIONS, Serialized, decode_typed, serialize_mode_r_style, serialize_mode_s,
                               shared_background)
from spectra.tokenizers import ClipLikeTokenizer, RegexTokenizer
from spectra.types import CER, DeviceState

from conftest import DEMO_ICU, needs_demo
from test_r002_exact_controls import _e2_tests


def _interfaces():
    return [
        Interface("reference_family", "typed"),
        Interface("ehrxdiff_local", "event_text", RegexTokenizer()),
        Interface("prompted_verbal", "verbal", RegexTokenizer()),
        Interface("state_predictor", "verbal", RegexTokenizer(max_len=4096)),
        Interface("prism_compact", "compact", ClipLikeTokenizer(77)),
    ]


def test_every_interface_round_trips_every_test():
    tests = _e2_tests()
    res = check_all(tests, _interfaces())
    assert blocked_systems(res) == {}, blocked_systems(res)


def test_truncation_blocks_a_system():
    tests = [t for t in _e2_tests() if t.device == "ETT"]
    iface = Interface("tiny_context", "verbal", RegexTokenizer(max_len=40))
    res = check_all(tests, [iface])
    reasons = blocked_systems(res)["tiny_context"]
    assert "truncated" in reasons


class _Faulty(Interface):
    """An interface whose inputs() is wrong in one specific way; the checker must catch it."""

    def __init__(self, *a, fault=None, **kw):
        super().__init__(*a, **kw)
        self.fault = fault

    def inputs(self, test, arm):
        if self.fault == "arm_a_twice":
            return super().inputs(test, "A")
        s = super().inputs(test, arm)
        if self.fault == "typed_device_flip" and arm == "A":   # corrupt the device token the system receives
            toks = [t if not (t[0] == "bundle" and t[3] != "none") else t[:4] + ("remove" if t[4] == "place" else "place",) + t[5:]
                    for t in s.tokens]
            return Serialized("", {}, toks)
        if arm != "B":
            return s
        if self.fault == "background_edit":          # an arm-specific event inside the background segment
            if self.style == "compact":                # "bg ..." -> "bg vset, ...", still within 77 tokens
                a, b = s.segments["background"]
                ins = "vset, "
                return Serialized(s.text[:a + 3] + ins + s.text[a + 3:],
                                  {"delta": s.segments["delta"], "background": (a, b + len(ins)),
                                   "bundle": tuple(x + len(ins) for x in s.segments["bundle"])})
            if self.style == "typed":
                return Serialized("", {}, [s.tokens[0], ("background", "fluid", "<12h", "none", "none", "none", "none", "mid")]
                                  + s.tokens[1:])
            a, b = s.segments["background"]
            text = s.text[:b - 1] + "; [+3.2 h] PEEP set=5." + s.text[b:]
            shift = len(text) - len(s.text)
            return Serialized(text, {"delta": s.segments["delta"], "background": (a, b + shift),
                                     "bundle": (s.segments["bundle"][0] + shift, s.segments["bundle"][1] + shift)})
        if self.fault == "trailing_text":             # text after the bundle that differs between arms
            return Serialized(s.text + " Note.", dict(s.segments))
        return s


@pytest.mark.parametrize("style,tok", [("verbal", RegexTokenizer()), ("event_text", RegexTokenizer()),
                                       ("compact", ClipLikeTokenizer(77)), ("typed", None)])
def test_checker_reads_the_inputs_the_system_receives(style, tok):
    tests = _e2_tests()
    assert blocked_systems(check_all(tests, [Interface("ok", style, tok)])) == {}
    r = check_all(tests, [_Faulty("bad", style, tok, fault="arm_a_twice")])
    assert any("state_not_recovered_B" in why for why in blocked_systems(r)["bad"])
    t = [x for x in tests if x.edit_class == "placement_deletion" and x.device == "ETT"][0]
    bad = _Faulty("bad", style, tok, fault="background_edit")
    if style == "compact":
        assert len(tok.encode(bad.inputs(t, "B").text)) <= 77
    why = check(t, bad).reasons
    assert {"background_text_differs", "background_not_identical"} & set(why), why
    if style != "typed":
        assert "text_outside_bundle_differs" in check(t, _Faulty("bad", style, tok, fault="trailing_text")).reasons
    else:
        assert "state_not_recovered_A" in check(t, _Faulty("bad", style, tok, fault="typed_device_flip")).reasons


def test_typed_decoding_is_independent_of_the_cer_objects():
    t = [x for x in _e2_tests() if x.edit_class == "mirror"][0]
    toks = Interface("ref", "typed").inputs(t, "B").tokens
    delta, bg, bundle = decode_typed(toks)
    assert delta == pytest.approx(t.delta_h, abs=0.05)
    assert [(c.device.type, c.device.transition, c.device.side) for c in bundle] == [("CVC", "place", "R")]   # flipped L -> R
    assert decode_typed([("delta", 12.0), ("bundle", "x")])[2] == []          # malformed tokens recover nothing


def test_oversized_mandatory_bundle_rejects_every_event_limited_interface():
    stream = ett_placement_stream()
    settings = tuple(CER("action", "ventilator_setting", 3.3 + i / 100, value=f"PEEP set={i}", priority="ventilator_setting")
                     for i in range(73))
    tests, _ = logged_tests("P1", prior_film("P1"), stream + settings, 12.0, "ETT", ABSENT, PRESENT)
    t = tests[0]
    assert len(t.bundle_a) > 64
    for iface in (Interface("v", "verbal", RegexTokenizer()), Interface("e", "event_text", RegexTokenizer()),
                  Interface("t", "typed")):
        assert check(t, iface).reasons == ["cannot_carry_delta_and_bundle"]
        with pytest.raises(ValueError):
            iface.inputs(t, "A")


def test_mode_r_serializers_per_style():
    stream = ett_placement_stream() + tuple(CER("action", "fluid", 0.1 + i / 100, value=f"fluid{i}", dose_value=10.0,
                                                dose_unit="ml", priority="fluid") for i in range(100))
    stream = tuple(sorted(stream, key=lambda c: c.sort_key()))
    v = serialize_mode_r_style(stream, 12.0, "verbal")
    assert v.text.count("[+") == 64                                     # capacity C = 64 recorded events
    assert serialize_mode_r_style(stream, 12.0, "verbal", actions_known=False).text.endswith(UNKNOWN_ACTIONS)
    assert serialize_mode_r_style((), 12.0, "verbal").text.endswith("none.")    # no actions != actions unknown
    ty = serialize_mode_r_style(stream, 12.0, "typed")
    assert ty.tokens[0] == ("delta", 12.0) and len(ty.tokens) == 65 and all(t[0] == "recorded" for t in ty.tokens[1:])
    assert serialize_mode_r_style(stream, 12.0, "typed", actions_known=False).tokens[1] == ("recorded", "actions_unknown")
    clip = ClipLikeTokenizer(77)
    wide = tuple(sorted(ett_placement_stream() + _distinct_background(), key=lambda c: c.sort_key()))
    full = serialize_mode_r_style(wide, 12.0, "compact")                    # no token budget
    c = Interface("prism", "compact", clip).mode_r_inputs(wide, 12.0)
    assert len(clip.encode(full.text)) > 77 and len(clip.encode(c.text)) <= 77
    assert c.text.startswith("dt 12h.") and "vent+" in c.text and c.text.count(",") < full.text.count(",")
    assert Interface("prism", "compact", clip).mode_r_inputs(stream, 12.0, actions_known=False).text.endswith("rec unknown.")
    with pytest.raises(ValueError):
        serialize_mode_r_style(stream, 12.0, "unknown_style")


def test_wrong_bundle_is_not_recovered():
    t = [x for x in _e2_tests() if x.edit_class == "removal_deletion" and x.device == "CVC"][0]
    swapped = replace(t, bundle_a=t.bundle_b)  # arm A no longer contains the removal
    r = check(swapped, Interface("prompted_verbal", "verbal", RegexTokenizer()))
    assert not r.ok and "state_not_recovered_A" in r.reasons


def test_capacity_rule_keeps_bundle_and_shares_background():
    stream = ett_placement_stream()
    many = tuple(CER("action", "fluid", 0.1 + i / 100, value=f"fluid{i}", dose_value=10.0, dose_unit="ml", priority="fluid")
                 for i in range(100))
    tests, _ = logged_tests("P1", prior_film("P1"), stream + many, 12.0, "ETT", ABSENT, PRESENT)
    t = tests[0]
    bprime = shared_background(t, 64)
    assert len(bprime) == 64 - max(len(t.bundle_a), len(t.bundle_b))
    # device procedures and settings outrank fluids; the diuretic outranks the fluids too
    assert any(c.action_type == "diuretic" for c in bprime)
    r = check(t, Interface("prompted_verbal", "verbal", RegexTokenizer()))
    assert r.ok, r.reasons


def _registered_text_systems():
    return [(n, s) for n, s in systems().items()
            if isinstance(s, dict) and s.get("tokenizer") and s.get("style") in ("verbal", "event_text", "compact")]


@pytest.mark.skipif(os.environ.get("SPECTRA_HF_TOKENIZERS") is None,
                    reason="set SPECTRA_HF_TOKENIZERS=registry (or system names) to round-trip with the systems' own tokenizers")
@pytest.mark.parametrize("name,spec", _registered_text_systems(), ids=[n for n, _ in _registered_text_systems()])
def test_registered_systems_round_trip_with_their_own_tokenizer(name, spec, request):
    """Every registered text-interface system, with its actual tokenizer and its own interface style."""
    from spectra.tokenizers import HFTokenizer

    sel = os.environ["SPECTRA_HF_TOKENIZERS"]
    if sel != "registry" and name not in sel.split(","):
        pytest.skip("not selected")
    from huggingface_hub import try_to_load_from_cache

    sub = spec.get("tokenizer_subfolder")
    staged = try_to_load_from_cache(spec["tokenizer"], f"{sub}/tokenizer_config.json" if sub else "tokenizer_config.json")
    if not isinstance(staged, str):  # not staged here (gated repository, offline machine): reported, never passed
        pytest.skip(f"{spec['tokenizer']} not staged (scripts/fetch_tokenizers.py)")
    tok = HFTokenizer(spec["tokenizer"], max_len=spec.get("max_tokens"), subfolder=sub,
                      trust_remote_code=bool(spec.get("tokenizer_remote_code")))   # any other failure is a failure
    iface = Interface(name, spec["style"], tok, template=template(name))
    tests = _e2_tests()
    if DEMO_ICU.exists():
        tests = tests + _demo_derived_tests(request.getfixturevalue("demo_tables"))
    res = check_all(tests, [iface])
    assert blocked_systems(res) == {}, (name, blocked_systems(res))
    if spec["style"] == "compact":
        for t in tests:
            assert all(len(tok.encode(iface.inputs(t, arm).text)) <= spec["max_tokens"] for arm in ("A", "B"))


def _distinct_background(n=60):
    """Background actions with distinct compact labels (the compact form merges repeated labels)."""
    return tuple(CER("action", f"med{i}", 0.1 + i / 10, priority="other_medication") for i in range(n))


def test_compact_form_budgets_background_to_77_tokens():
    """A realistic ICU window (many background actions) must still fit CLIP's 77 tokens with M intact."""
    tests, _ = logged_tests("P1", prior_film("P1"), ett_placement_stream() + _distinct_background(), 12.0, "ETT",
                            ABSENT, PRESENT)
    tok = ClipLikeTokenizer(77)
    iface = Interface("prism_compact", "compact", tok)
    event_capped = shared_background(tests[0], 64)
    assert len(tok.encode(serialize_mode_s(tests[0], "A", "compact", 64, event_capped).text)) > 77
    b = iface.background(tests[0])
    assert b is not None and 0 < len(b) < len(event_capped)            # token budgeting trims beyond the event cap
    for arm in ("A", "B"):
        assert len(tok.encode(iface.inputs(tests[0], arm).text)) <= 77
    assert check(tests[0], iface).ok
    # a 20-token encoder cannot carry Delta + M: rejected for E2
    assert check(tests[0], Interface("tiny_clip", "compact", ClipLikeTokenizer(20))).reasons == ["cannot_carry_delta_and_bundle"]


def _demo_derived_tests(demo_tables):
    from spectra.cer import CERCompiler
    from spectra.types import Film

    comp = CERCompiler()
    s1 = comp.compile(demo_tables, 30757476, "2131-03-02 00:20", "2131-03-03 08:00")
    s2 = comp.compile(demo_tables, 32391858, "2201-10-30 17:00", "2201-10-31 08:00")
    t1, _ = logged_tests("10000001", Film("10000001", "s2", "d2", 0.0, 0, 20, "30757476"), s1, 31.67, "CVC",
                         DeviceState(True, "R"), ABSENT)
    t2, _ = logged_tests("10000002", Film("10000002", "s3", "d3", 0.0, 0, 1020, "32391858"), s2, 15.0, "CVC",
                         ABSENT, DeviceState(True, "L"))
    return t1 + t2


@needs_demo
def test_round_trip_on_demo_derived_tests(demo_tables):
    """Real compiled windows (open MIMIC-IV demo): a CVC removal deletion and a CVC placement deletion."""
    tests = _demo_derived_tests(demo_tables)
    assert len(tests) == 2
    res = check_all(tests, _interfaces())
    assert blocked_systems(res) == {}, blocked_systems(res)


def test_prompt_templates_are_registered_and_well_formed():
    from spectra.config import _shared

    reg = systems()
    for name, ts in _shared("templates").items():
        if name == "answer_format":
            continue
        assert name in reg and reg[name].get("style") in ("verbal", "event_text", "compact"), name
        assert 1 <= len(ts) <= 3, name                                  # at most three templates per system
        for tid in ts:
            t = template(name, tid)
            assert t.count("{input}") == 1 and "{answer_format}" not in t
    assert '"endotracheal_tube"' in template("lingshu_32b") and template("unregistered_system") == "{input}"


def test_round_trip_runs_on_the_templated_text():
    tests = _e2_tests()
    t = [x for x in tests if x.edit_class == "placement_deletion" and x.device == "ETT"][0]
    plain = Interface("p", "verbal", RegexTokenizer())
    tpl = Interface("p", "verbal", RegexTokenizer(), template="Edit this radiograph. {input}\nAnswer in JSON.")
    a, b = plain.inputs(t, "A"), tpl.inputs(t, "A")
    assert b.text == "Edit this radiograph. " + a.text + "\nAnswer in JSON."
    for seg in ("delta", "background", "bundle"):
        assert b.text[slice(*b.segments[seg])] == a.text[slice(*a.segments[seg])]
    assert blocked_systems(check_all(tests, [tpl])) == {}
    with pytest.raises(ValueError):
        Interface("p", "verbal", RegexTokenizer(), template="no slot")


def test_compact_budget_includes_the_template():
    stream = ett_placement_stream()
    # distinct medications: the compact form merges repeated labels, so only distinct ones cost tokens
    many = tuple(CER("action", f"med{i}", 0.1 + i / 10, priority="other_medication") for i in range(40))
    t = logged_tests("P1", prior_film("P1"), stream + many, 12.0, "ETT", ABSENT, PRESENT)[0][0]
    tok = ClipLikeTokenizer(77)
    plain = Interface("prism", "compact", tok)
    long = Interface("prism", "compact", tok, template="a frontal chest radiograph of an adult patient in the icu. {input}")
    assert len(long.background(t)) < len(plain.background(t))
    for arm in ("A", "B"):
        assert len(tok.encode(long.inputs(t, arm).text)) <= 77
    assert check(t, long).ok
    huge = Interface("prism", "compact", tok, template=" ".join(["radiograph"] * 40) + " {input}")
    assert check(t, huge).reasons == ["cannot_carry_delta_and_bundle"]
    r = long.mode_r_inputs(stream + many, 12.0)
    assert r.text.startswith("a frontal chest radiograph") and len(tok.encode(r.text)) <= 77
