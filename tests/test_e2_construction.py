"""E2 eligibility, primary cohort, secondary allocation and the scoring gate (review round 1: items 2, 9, 10, 12)."""
from dataclasses import replace

import pytest

from spectra.adapters.synthetic import (ABSENT, PRESENT, cvc_placement_stream, ett_placement_stream, ett_removal_stream,
                                        prior_film)
from spectra.config import dependency_table
from spectra.controls import ExpectedStateOracle
from spectra.e2 import (SystemBlocked, _common_checks, allocate_secondary, cross_source_tests, insertion_test,
                        logged_tests, mirror_test, primary_cohort, score_system, type_control_test)
from spectra.roundtrip import Interface, check_all
from spectra.tokenizers import RegexTokenizer
from spectra.types import CER, Device, DeviceState

DEPS = dependency_table()
CVC_STATE = CER("state", "line_present", None, device=Device("CVC", "none", "IJ", "R", "Multi Lumen"), priority="device_procedure")
BG = (CER("action", "antibiotic", 1.0, value="Cefepime", dose_value=1.0, dose_unit="dose", priority="antibiotic"),)


def _chest_tube_stream(t=2.0):
    return (CER("transition", "chest_tube_placement", t, device=Device("chest_tube", "place", "pleural", "R"),
                priority="device_procedure"),) + BG


# ---- eligibility: one set of checks for every constructor ------------------------------------------
def test_missing_records_are_not_evidence_of_absence():
    # report-verified present CVC, no CVC rows charted at t0: no conflict (absence is not documented)
    assert _common_checks(BG, "CVC", DeviceState(True, "R"), DEPS) is None
    # active rows contradict a report-verified absent state
    assert _common_checks(BG + (CVC_STATE,), "CVC", ABSENT, DEPS) == "record_report_conflict_at_prior"
    assert _common_checks(BG + (CVC_STATE, replace(CVC_STATE, device=replace(CVC_STATE.device, side="L"))), "CVC",
                          DeviceState(True, "R"), DEPS) == "more_than_one_instance"
    assert _common_checks(BG, "CVC", None, DEPS) == "initial_state_unverified"


def test_mirror_rejects_a_placement_beside_a_present_line():
    stream = tuple(sorted(cvc_placement_stream(side="L") + (CVC_STATE,), key=lambda c: c.sort_key()))
    m, r = mirror_test("P1", prior_film("P1"), stream, 12.0, "CVC", ABSENT, "L")
    assert m is None and r.rule == "record_report_conflict_at_prior"
    m, r = mirror_test("P1", prior_film("P1"), cvc_placement_stream(side="L"), 12.0, "CVC", DeviceState(True, "R"), "L")
    assert m is None and r.rule == "initial_state_not_absent"
    m, r = mirror_test("P1", prior_film("P1"), cvc_placement_stream(side="L"), 12.0, "CVC", ABSENT, "R")
    assert m is None and r.rule == "record_report_side_disagree"
    m, r = mirror_test("P1", prior_film("P1"), cvc_placement_stream(side="L", site="femoral"), 12.0, "CVC", ABSENT, "L")
    assert m is None and r.rule == "visibility:femoral"


def test_insertion_rejects_an_absent_ett_with_ongoing_invasive_ventilation():
    vent = (CER("state", "invasive_ventilation", None, priority="device_procedure"),) + BG
    t, r = insertion_test("P1", prior_film("P1"), vent, 12.0, "ETT", ABSENT, ABSENT)
    assert t is None and r.rule == "record_report_conflict_at_prior"
    trach = (CER("state", "tracheostomy", None, priority="device_procedure"),) + BG
    t, r = insertion_test("P1", prior_film("P1"), trach, 12.0, "ETT", ABSENT, ABSENT)
    assert t is None and r.rule == "exclusion:tracheostomy"
    t, r = insertion_test("P1", prior_film("P1"), BG, 12.0, "ETT", ABSENT, ABSENT)
    assert t is not None and (t.expected_a, t.expected_b) == (False, True)


def test_dependents_that_presuppose_the_device_block_insertion_and_type_control():
    feed = (CER("action", "enteral_feed", 2.0, value="Tube feeding", priority="other_medication"),) + BG
    t, r = insertion_test("P1", prior_film("P1"), feed, 12.0, "enteric", ABSENT, ABSENT)
    assert t is None and r.rule == "target_or_dependent_records_present"
    t, r = type_control_test("P1", prior_film("P1"), feed, 12.0, "CVC", "enteric", ABSENT, ABSENT)
    assert t is None and r.rule == "enteric:target_or_dependent_records_present"
    settings = (CER("action", "ventilator_setting", 2.0, value="PEEP set=5", priority="ventilator_setting"),) + BG
    t, r = insertion_test("P1", prior_film("P1"), settings, 12.0, "ETT", ABSENT, ABSENT)
    assert t is None and r.rule == "target_or_dependent_records_present"
    assert insertion_test("P1", prior_film("P1"), BG, 12.0, "enteric", ABSENT, ABSENT)[0] is not None


def test_removal_deletion_extends_only_what_the_removal_ended():
    # ventilation from before t0, PEEP 8 -> 5 at +1 h, extubation and ventilation stop at +5 h
    tests, _ = logged_tests("P2", prior_film("P2"), ett_removal_stream(), 12.0, "ETT", PRESENT, ABSENT)
    b = tests[0].bundle_b
    vent = [c for c in b if c.action_type == "invasive_ventilation"]
    peep = {c.value: c.ongoing_at_delta for c in b if c.action_type == "ventilator_setting"}
    assert vent[0].ongoing_at_delta and peep == {"PEEP set=8": False, "PEEP set=5": True}
    a = tests[0].bundle_a
    assert not any(c.ongoing_at_delta for c in a if c.action_type == "ventilator_setting")


def test_logged_tests_apply_the_same_exclusions():
    trach = tuple(sorted(ett_placement_stream() + (CER("state", "tracheostomy", None, priority="device_procedure"),),
                         key=lambda c: c.sort_key()))
    tests, rej = logged_tests("P1", prior_film("P1"), trach, 12.0, "ETT", ABSENT, PRESENT)
    assert not tests and rej[0].rule == "exclusion:tracheostomy"
    port = tuple(replace(c, device=replace(c.device, subtype="Indwelling Port (PortaCath)")) if c.device.type == "CVC" else c
                 for c in cvc_placement_stream())
    tests, rej = logged_tests("P1", prior_film("P1"), port, 12.0, "CVC", ABSENT, DeviceState(True, "R"))
    assert not tests and rej[0].rule == "exclusion:subtype"


def test_cross_source_construction_receives_the_report_flags():
    p = prior_film("P1")
    assert cross_source_tests("P1", p, "ETT", None)[1][0].rule == "initial_state_unverified"
    assert cross_source_tests("P1", p, "ETT", ABSENT, {"tracheostomy": PRESENT})[1][0].rule == "exclusion:tracheostomy"
    assert cross_source_tests("P1", p, "CVC", ABSENT, {"port": PRESENT})[1][0].rule == "exclusion:port"
    assert cross_source_tests("P1", p, "CVC", PRESENT, {"other_CVC": PRESENT})[1][0].rule == "more_than_one_instance"
    t = cross_source_tests("P1", p, "CVC", PRESENT, {})[0][0]
    assert t.edit_class == "removal_specification" and t.delta_h == 24.0 and not t.actions_known


# ---- primary cohort ------------------------------------------------------------------------------------------
def test_primary_cohort_is_test_split_only_and_independent_of_input_order():
    p = prior_film("P1")
    a, _ = logged_tests("P1", p, ett_placement_stream(), 12.0, "ETT", ABSENT, PRESENT, target_study="S12")
    b, _ = logged_tests("P1", p, ett_placement_stream(), 24.0, "ETT", ABSENT, PRESENT, target_study="S24")
    a, b = a[0], b[0]
    assert a.test_id != b.test_id                       # target and horizon are part of the identity
    assert primary_cohort([a, b]) == primary_cohort([b, a]) == [a]   # same transition: the first follow-up wins
    dev, _ = logged_tests("P2", replace(prior_film("P2"), split="dev_train"), ett_placement_stream(), 12.0, "ETT",
                          ABSENT, PRESENT)
    assert primary_cohort(dev + [a]) == [a]
    early, _ = logged_tests("P1", replace(p, study_id="P1-S1", t_h=p.t_h - 30.0), ett_removal_stream(), 12.0, "ETT",
                            PRESENT, ABSENT, target_study="S0")
    assert primary_cohort([a, b] + early) == early        # the earliest logged transition per patient and device


# ---- secondary allocation ------------------------------------------------------------------------------------
def test_secondary_logged_transitions_come_first():
    # target 5: up to floor(0.4 x 5) = 2 insertions would be eligible, so the class order decides the selection
    logged = [logged_tests(f"L{i}", prior_film(f"L{i}"), _chest_tube_stream(), 12.0, "chest_tube", ABSENT,
                           DeviceState(True, "R"))[0][0] for i in range(6)]
    ins = [insertion_test(f"I{i}", prior_film(f"I{i}"), BG, 12.0, "chest_tube", ABSENT, ABSENT)[0] for i in range(3)]
    for seed in range(5):
        got = allocate_secondary(ins + logged, {"chest_tube": 5}, seed=seed)
        assert [t.edit_class for t in got] == ["placement_deletion"] * 5
    got = allocate_secondary(ins + logged, {"chest_tube": 5}, seed=0, priority=["insertion", "mirror", "type_control"])
    assert [t.edit_class for t in got] == ["placement_deletion"] * 5      # the priority list orders the other classes
    got = allocate_secondary(ins + logged[:2], {"chest_tube": 5}, seed=0)
    assert sorted(t.edit_class for t in got) == ["insertion", "insertion", "placement_deletion", "placement_deletion"]


def test_type_controls_use_the_patient_for_both_devices_and_classes_are_capped():
    tc, _ = type_control_test("C", prior_film("C"), BG, 12.0, "enteric", "chest_tube", ABSENT, ABSENT)
    ins_c, _ = insertion_test("C", prior_film("C"), BG, 12.0, "chest_tube", ABSENT, ABSENT)
    got = allocate_secondary([ins_c, tc], {"enteric": 5, "chest_tube": 5}, seed=0)
    assert [t.edit_class for t in got] == ["type_control"]          # C is used for the chest tube as well
    many = [insertion_test(f"I{i}", prior_film(f"I{i}"), BG, 12.0, "chest_tube", ABSENT, ABSENT)[0] for i in range(6)]
    got = allocate_secondary(many, {"chest_tube": 5}, seed=0)
    assert len(got) == 2                                             # floor(0.40 x 5) insertions at most
    got = allocate_secondary(many + [tc], {"chest_tube": 5}, seed=0, exclude_patients={"chest_tube": {"I0", "I1", "I2"}})
    assert {t.patient_id for t in got} <= {"I3", "I4", "I5"} and len(got) == 2


# ---- scoring gate -------------------------------------------------------------------------------------------
def test_score_system_refuses_without_a_passing_round_trip():
    p = prior_film("P1")
    tests, _ = logged_tests("P1", p, ett_placement_stream(), 12.0, "ETT", ABSENT, PRESENT)
    o = ExpectedStateOracle()
    seeds = [1009]
    rd = {(t.test_id, arm, s): o.readout(t, arm) for t in tests for arm in ("A", "B") for s in seeds}
    with pytest.raises(SystemBlocked):
        score_system("S2", tests, rd, seeds, {})
    rt = check_all(tests, [Interface("S2", "verbal", RegexTokenizer())])
    assert score_system("S2", tests, rd, seeds, rt)["ETT"]["JDA"] == 1.0
    tiny = check_all(tests, [Interface("S2", "verbal", RegexTokenizer(max_len=30))])
    with pytest.raises(SystemBlocked):
        score_system("S2", tests, rd, seeds, tiny)
    with pytest.raises(SystemBlocked):              # a certificate must cover every scored test
        more, _ = logged_tests("P3", prior_film("P3"), ett_removal_stream(), 12.0, "ETT", PRESENT, ABSENT)
        score_system("S2", tests + more, rd, seeds, rt)
