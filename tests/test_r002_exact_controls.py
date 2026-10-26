"""R002: exact control properties (P2) on synthetic sequences. Any failure here stops the run."""
import numpy as np
import pytest

from spectra.adapters.synthetic import (ABSENT, PRESENT, SyntheticSource, cvc_placement_stream, cvc_removal_stream,
                                        ett_placement_stream, ett_removal_stream, prior_film)
from spectra.controls import (BundleBlind, CopyPrior, ExactControlFailure, ExpectedStateOracle, QueryBlind,
                              RecordVolume, assert_half, assert_undefined, assert_zero)
from spectra.e1 import admitted_contrasts, channel_value, concordance, query_id, query_sets, sample_priors
from spectra.e2 import (cross_source_tests, insertion_test, joint_accuracy, jda_primary, logged_tests, mirror_test,
                        type_control_test)
from spectra.references.resolver import orderings_from_grades
from spectra.types import DeviceState

SEEDS = [1009, 1010, 1011, 1012]
CHANNELS = ["edema", "effusion_L", "effusion_R"]


def _e1_setup(seed=0):
    src = SyntheticSource(n_patients=60, n_films=6, seed=seed)
    films, grades = src.films(), src.grades()
    sets = query_sets(films, 1.0, 168.0, 3, same_stay=True)
    orderings = orderings_from_grades(films, grades, CHANNELS)
    with_c = [(qs, admitted_contrasts(qs, orderings, CHANNELS)) for qs in sets]
    chosen = sample_priors(with_c, 3, "seed")
    contrasts = [(qs.prior.patient_id, c) for qs, cons in chosen for c in cons]
    return src, chosen, contrasts, grades


def _readouts(chosen, gen, grades, seeds):
    """phi_c of each generated follow-up; generated outputs are vectors of channel values."""
    out = {}
    for qs, _ in chosen:
        prior_state = tuple(grades[qs.prior.study_id][c] for c in CHANNELS)
        for f in qs.later:
            for s in seeds:
                cond = (f.t_h - qs.prior.t_h,)
                x = gen.generate(prior_state, cond, s)
                out[(query_id(qs.prior, f), s)] = dict(zip(CHANNELS, x))
    return out


def test_copy_prior_is_exactly_half():
    src, chosen, contrasts, grades = _e1_setup()
    assert contrasts, "synthetic source must yield admitted contrasts"
    res = concordance(contrasts, _readouts(chosen, CopyPrior(), grades, SEEDS), SEEDS)
    for ch in ["edema", "effusion_L", "effusion_R", "effusion"]:
        assert_half(channel_value(res, ch), f"copy-prior {ch}")
        assert all(v == 0.5 for v in res[ch]["patients"].values())


def test_query_blind_generator_with_shared_reset_seeds_is_exactly_half():
    src, chosen, contrasts, grades = _e1_setup(seed=3)
    res = concordance(contrasts, _readouts(chosen, QueryBlind(noise=2.0), grades, SEEDS), SEEDS)
    for ch in ["edema", "effusion_L", "effusion_R", "effusion"]:
        assert_half(channel_value(res, ch), f"query-blind {ch}")


def test_query_using_oracle_beats_half_and_unshared_seeds_break_exactness():
    src, chosen, contrasts, grades = _e1_setup(seed=5)

    readouts = {}
    for qs, _ in chosen:
        for f in qs.later:
            for s in SEEDS:
                readouts[(query_id(qs.prior, f), s)] = {c: float(grades[f.study_id][c]) for c in CHANNELS}
    res = concordance(contrasts, readouts, SEEDS)
    assert channel_value(res, "edema") == 1.0  # the reference readout orders every admitted contrast
    # a query-blind generator whose seeds are NOT shared across a prior's queries is only 1/2 in expectation
    qb = QueryBlind(noise=2.0)
    unshared = {}
    for qs, _ in chosen:
        prior_state = tuple(grades[qs.prior.study_id][c] for c in CHANNELS)
        for f in qs.later:
            for s in SEEDS:
                x = qb.generate(prior_state, None, hash((f.study_id, s)) % 10_000)
                unshared[(query_id(qs.prior, f), s)] = dict(zip(CHANNELS, x))
    v = channel_value(concordance(contrasts, unshared, SEEDS), "edema")
    assert v != 0.5 and abs(v - 0.5) < 0.15


def test_channel_without_contrast_is_undefined():
    src, chosen, contrasts, grades = _e1_setup()
    only_edema = [(p, c) for p, c in contrasts if c.channel == "edema"]
    res = concordance(only_edema, _readouts(chosen, CopyPrior(), grades, SEEDS), SEEDS)
    assert_undefined(channel_value(res, "effusion_R"), "effusion_R")
    assert_undefined(channel_value(res, "effusion"), "pooled effusion")
    with pytest.raises(ExactControlFailure):
        assert_undefined(0.5, "a channel reported as 1/2 instead of undefined")


def _e2_tests():
    p = prior_film
    tests = []
    t, _ = logged_tests("P1", p("P1"), ett_placement_stream(), 12.0, "ETT", ABSENT, PRESENT); tests += t
    t, _ = logged_tests("P2", p("P2"), ett_removal_stream(), 12.0, "ETT", PRESENT, ABSENT); tests += t
    t, _ = logged_tests("P3", p("P3"), cvc_placement_stream(), 12.0, "CVC", ABSENT, DeviceState(True, "R")); tests += t
    t, _ = logged_tests("P4", p("P4"), cvc_removal_stream(), 12.0, "CVC", DeviceState(True, "L"), ABSENT); tests += t
    m, _ = mirror_test("P5", p("P5"), cvc_placement_stream(side="L"), 12.0, "CVC", ABSENT, "L"); tests.append(m)
    bg_only = tuple(c for c in cvc_removal_stream() if c.device.type == "none")  # one antibiotic, no device events
    i, _ = insertion_test("P6", p("P6"), bg_only, 12.0, "ETT", ABSENT, ABSENT); tests.append(i)
    tc, _ = type_control_test("P7", p("P7"), bg_only, 12.0, "CVC", "enteric", ABSENT, ABSENT); tests.append(tc)
    xs, _ = cross_source_tests("P8", p("P8"), "CVC", ABSENT); tests += xs
    xs, _ = cross_source_tests("P9", p("P9"), "ETT", PRESENT); tests += xs
    assert all(t is not None for t in tests)
    return tests


def test_e2_construction_classes():
    tests = _e2_tests()
    classes = sorted(t.edit_class for t in tests)
    assert classes == sorted(["placement_deletion", "removal_deletion", "placement_deletion", "removal_deletion",
                              "mirror", "insertion", "type_control", "insertion", "removal_specification"])
    for t in tests:
        assert t.expected_a != t.expected_b, t.edit_class


def _space_readout(test, present_a, present_b=None):
    """Turn a bundle-blind presence draw into the test's state space (identical in both arms)."""
    if test.sigma == "presence":
        return present_a
    if test.sigma == "presence_side":
        return (present_a, "L")
    return (present_a, present_a)


def test_bundle_blind_system_scores_zero_jda():
    tests = _e2_tests()
    bb = BundleBlind()
    readouts = {}
    for t in tests:
        for s in SEEDS:
            same = bb.generate_arm(t, "A", s)
            assert same == bb.generate_arm(t, "B", s)
            readouts[(t.test_id, "A", s)] = _space_readout(t, same)
            readouts[(t.test_id, "B", s)] = _space_readout(t, same)
    for cls in {t.edit_class for t in tests}:
        sub = [t for t in tests if t.edit_class == cls]
        assert_zero(joint_accuracy(sub, readouts, SEEDS)["JDA"], f"bundle-blind {cls}")
    # copy-prior is bundle-blind: under a perfect readout its JDA is 0, its FA 0 and EA 1 on the primary tests
    copy = {}
    for t in tests:
        for s in SEEDS:
            st = t.s0.in_space(t.sigma) if t.sigma != "type_pair" else (False, False)
            copy[(t.test_id, "A", s)] = st
            copy[(t.test_id, "B", s)] = st
    prim = [t for t in tests if t.edit_class in ("placement_deletion", "removal_deletion")]
    r = joint_accuracy(prim, copy, SEEDS)
    assert r["JDA"] == 0.0 and r["FA"] == 0.0 and r["EA"] == 1.0


def test_record_volume_control_properties():
    tests = _e2_tests()
    rv = RecordVolume()
    readouts = {(t.test_id, arm, s): rv.readout(t, arm) for t in tests for arm in ("A", "B") for s in SEEDS}
    for cls in ("removal_deletion", "mirror"):
        sub = [t for t in tests if t.edit_class == cls]
        assert_zero(joint_accuracy(sub, readouts, SEEDS)["JDA"], f"record-volume {cls}")
    # by construction it is perfect on placement deletion: the reason for the off-target and mirror checks
    sub = [t for t in tests if t.edit_class == "placement_deletion"]
    assert joint_accuracy(sub, readouts, SEEDS)["JDA"] == 1.0


def test_oracle_scores_one_and_primary_weights():
    tests = _e2_tests()
    o = ExpectedStateOracle()
    readouts = {(t.test_id, arm, s): o.readout(t, arm) for t in tests for arm in ("A", "B") for s in SEEDS}
    assert joint_accuracy(tests, readouts, SEEDS)["JDA"] == 1.0
    r = jda_primary(tests, readouts, SEEDS)
    assert r["JDA_primary"] == 1.0 and r["ETT"]["n_patients"] == 2 and r["CVC"]["n_patients"] == 2
    # unequal device accuracies: every CVC edited arm wrong -> JDA_CVC = 0, JDA_primary = 1/2 x 1 + 1/2 x 0
    wrong = dict(readouts)
    for t in tests:
        if t.device == "CVC" and t.edit_class in ("placement_deletion", "removal_deletion"):
            for s in SEEDS:
                wrong[(t.test_id, "B", s)] = t.expected_a
    r = jda_primary(tests, wrong, SEEDS)
    assert r["ETT"]["JDA"] == 1.0 and r["CVC"]["JDA"] == 0.0 and r["JDA_primary"] == 0.5
    assert jda_primary(tests, wrong, SEEDS, weights={"ETT": 1.0, "CVC": 0.0})["JDA_primary"] == 1.0


def test_unassessable_readouts_count_as_incorrect():
    tests = [t for t in _e2_tests() if t.edit_class == "mirror"]
    o = ExpectedStateOracle()
    readouts = {}
    for t in tests:
        for s in SEEDS:
            readouts[(t.test_id, "A", s)] = o.readout(t, "A")
            readouts[(t.test_id, "B", s)] = (True, "unassessed")
    assert joint_accuracy(tests, readouts, SEEDS)["JDA"] == 0.0
