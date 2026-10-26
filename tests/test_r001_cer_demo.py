"""R001: CER compiler, cutoff, dependency bundles and background/bundle partition on the open MIMIC-IV
demo 2.2. Expected values were read by hand from the demo rows (times in hours after t0)."""
import pandas as pd
import pytest

from spectra.cer import CERCompiler, MimicOntology
from spectra.e2 import expected_state, logged_tests, partition, record_evidence_at_t0
from spectra.types import DeviceState, Film

from conftest import needs_demo


def _dev(stream, kind=None):
    return [(c.kind, c.action_type, None if c.t_start_h is None else round(c.t_start_h, 2), c.device.type,
             c.device.transition, c.device.site, c.device.side, c.candidate)
            for c in stream if c.device.type != "none" and (kind is None or c.kind == kind)]


@needs_demo
def test_stay_33281088_lines_and_reintubation(demo_tables):
    s = CERCompiler().compile(demo_tables, 33281088, "2140-01-30 05:44", "2140-02-03 15:54")
    dev = _dev(s)
    assert ("state", "line_present", None, "PICC", "none", "none", "none", False) in dev
    assert ("transition", "line_placement", 14.27, "CVC", "place", "IJ", "R", False) in dev  # 01-30 20:00
    assert ("transition", "intubation", 58.77, "ETT", "place", "nasal/oral", "none", False) in dev  # 02-01 16:30
    assert ("transition", "extubation", 83.93, "ETT", "remove", "nasal/oral", "none", False) in dev  # 02-02 17:40
    assert ("transition", "intubation", 85.32, "ETT", "place", "nasal/oral", "none", False) in dev  # 02-02 19:03
    vent = [(c.action_type, round(c.t_start_h, 2), c.ongoing_at_delta) for c in s if "invasive_ventilation" in c.action_type]
    assert vent == [("invasive_ventilation", 58.82, False), ("invasive_ventilation_stop", 83.93, False),
                    ("invasive_ventilation", 85.33, True)]
    prior = Film("10000000", "s1", "d1", 0.0, 0, 344, "33281088", split="test")
    tests, rej = logged_tests("10000000", prior, s, 106.17, "ETT", DeviceState(False), DeviceState(True))
    assert not tests and rej[0].rule == "more_than_one_transition"


@needs_demo
def test_stay_30757476_candidate_line_removal_and_removal_deletion(demo_tables):
    s = CERCompiler().compile(demo_tables, 30757476, "2131-03-02 00:20", "2131-03-03 08:00")
    dev = _dev(s)
    assert ("state", "line_present", None, "CVC", "none", "IJ", "R", False) in dev
    assert ("transition", "line_removal", 16.67, "CVC", "remove", "IJ", "R", True) in dev   # endtime 03-02 17:00
    assert ("transition", "line_placement", 16.8, "PICC", "place", "none", "none", False) in dev
    assert any(c.action_type == "invasive_ventilation" and c.is_state and c.ongoing_at_delta for c in s)
    assert record_evidence_at_t0(s, "ETT").present
    prior = Film("10000001", "s2", "d2", 0.0, 0, 20, "30757476", split="test")
    tests, rej = logged_tests("10000001", prior, s, 31.67, "CVC", DeviceState(True, "R"), DeviceState(False))
    t = tests[0]
    assert t.edit_class == "removal_deletion" and (t.expected_a, t.expected_b) == (False, True)
    assert any(c.device.transition == "remove" for c in t.bundle_a)
    assert not any(c.device.transition == "remove" for c in t.bundle_b)
    # removal deleted: the carried-forward line state is ongoing at Delta in the edited arm, not in the logged one
    st_a = [c for c in t.bundle_a if c.is_state and c.device.type == "CVC"][0]
    st_b = [c for c in t.bundle_b if c.is_state and c.device.type == "CVC"][0]
    assert st_a.ongoing_at_delta is False and st_b.ongoing_at_delta is True
    assert all(c.device.type != "CVC" for c in t.background)


@needs_demo
def test_stay_32391858_cvc_placement_partition(demo_tables):
    s = CERCompiler().compile(demo_tables, 32391858, "2201-10-30 17:00", "2201-10-31 08:00")
    assert ("transition", "line_placement", 0.3, "CVC", "place", "subclavian", "L", False) in _dev(s)
    prior = Film("10000002", "s3", "d3", 0.0, 0, 1020, "32391858", split="test")
    tests, rej = logged_tests("10000002", prior, s, 15.0, "CVC", DeviceState(False), DeviceState(True, "L"))
    t = tests[0]
    assert t.edit_class == "placement_deletion"
    assert expected_state(t.s0, t.bundle_a, "CVC") == DeviceState(True, "L")
    assert expected_state(t.s0, t.bundle_b, "CVC") == DeviceState(False)
    assert set(t.background) == set(c.with_part("background") for c in s if c.device.type != "CVC")


@needs_demo
def test_availability_cutoff_drops_a_line_stored_after_the_window(demo_tables):
    # Multi Lumen Left Subclavian: starttime 2201-10-30 17:18, storetime 2201-11-11 15:03 (stored at removal)
    t0, tk = "2201-10-30 17:00", "2201-10-31 08:00"
    occ = CERCompiler().compile(demo_tables, 32391858, t0, tk, cutoff="occurrence")
    avail = CERCompiler().compile(demo_tables, 32391858, t0, tk, cutoff="availability")
    placed = lambda s: [c for c in s if c.device.type == "CVC" and c.device.transition == "place"]
    assert len(placed(occ)) == 1 and placed(avail) == []


@needs_demo
def test_documented_pause_splits_an_infusion_and_rate_reconstructs_the_dose(demo_tables):
    # furosemide 221794, linkorderid 9155891: 10:30-12:08 (10 mg/h) + 12:08-16:02 (5 mg/h, paused),
    # then 2171-11-15 02:00-08:17 (5 mg/h). t0 inside the pause, tk two hours after the restart.
    s = CERCompiler().compile(demo_tables, 30057454, "2171-11-14 20:00", "2171-11-15 04:00")
    furo = [c for c in s if 221794 in c.source_itemids]
    assert not any(c.is_state for c in furo), "an infusion paused at t0 is not ongoing at t0"
    restart = [c for c in furo if c.t_start_h is not None and round(c.t_start_h, 2) == 6.0]
    assert len(restart) == 1
    assert restart[0].dose_value == pytest.approx(10.0)          # 5 mg/hour x 2 h inside the window
    assert restart[0].ongoing_at_delta and restart[0].dose_unit == "mg"


@needs_demo
def test_weight_based_rate_dose(demo_tables):
    # norepinephrine 221906 row 2111-11-15 14:03-17:05, 0.391310 mcg/kg/min, 97 kg; window 15:00-16:00
    s = CERCompiler().compile(demo_tables, 34617352, "2111-11-15 15:00", "2111-11-15 16:00")
    ne = [c for c in s if 221906 in c.source_itemids and c.is_state]
    assert ne, "infusion running at t0 is carried forward"
    assert ne[0].dose_value == pytest.approx(0.3913097898475825 * 97 * 60 / 1000.0, abs=1e-6)  # stored to 1e-6


@needs_demo
def test_dose_without_documented_rate_is_unknown_across_a_boundary(demo_tables):
    # potassium chloride 225166, linkorderid 7431790: 10:07-16:45, 39.8 mEq, no rate (a prorated amount
    # would be invented: section 9 makes it unknown)
    cross = CERCompiler().compile(demo_tables, 39804682, "2178-12-21 12:00", "2178-12-21 14:00")
    kcl = [c for c in cross if 225166 in c.source_itemids and c.is_state]
    assert len(kcl) == 1 and kcl[0].dose_value is None and kcl[0].dose_unit is None
    whole = CERCompiler().compile(demo_tables, 39804682, "2178-12-21 10:00", "2178-12-21 17:00")
    inside = [c for c in whole if 225166 in c.source_itemids and round(c.t_start_h, 2) == 0.12]
    assert len(inside) == 1 and inside[0].dose_value == pytest.approx(39.8, abs=1e-4) and inside[0].dose_unit == "mEq"
    later = [c for c in whole if 225166 in c.source_itemids and round(c.t_start_h, 2) == 6.75]
    assert len(later) == 1 and later[0].dose_value is None      # 16:45-22:49 crosses tk, no rate


@needs_demo
def test_microbiology_collection_and_result_are_separate_events(demo_tables):
    # blood culture collected 2146-06-22 23:30; E. coli stored 2146-06-25 12:34 (no-growth rows stored 06-23 12:59)
    comp = CERCompiler()
    ev = lambda s, kind: sorted((round(c.t_start_h, 2), c.value) for c in s if c.action_type == f"obs:{kind}:blood_culture")
    s = comp.compile(demo_tables, 32359580, "2146-06-22 20:00", "2146-06-25 13:00", track="B")
    assert (3.5, "Blood Culture, Routine") in ev(s, "micro")
    assert (64.57, "ESCHERICHIA COLI") in ev(s, "micro_result")          # at its store time, not backdated
    # collection before t0, result inside the window: the result enters at its store time
    s = comp.compile(demo_tables, 32359580, "2146-06-24 20:00", "2146-06-25 13:00", track="B")
    assert not [c for c in ev(s, "micro") if c[0] < 0] and (16.57, "ESCHERICHIA COLI") in ev(s, "micro_result")
    for cutoff in ("occurrence", "availability"):                           # not yet stored: no result
        s = comp.compile(demo_tables, 32359580, "2146-06-22 20:00", "2146-06-24 20:00", track="B", cutoff=cutoff)
        assert "ESCHERICHIA COLI" not in [v for _, v in ev(s, "micro_result")]
    # two blood-culture specimens collected at the same minute stay two collections (specimen identity)
    s = comp.compile(demo_tables, 36753294, "2141-05-23 00:00", "2141-05-24 00:00", track="B")
    assert ev(s, "micro").count((5.8, "Blood Culture, Routine")) == 2


@needs_demo
def test_carry_in_infusion_that_stops_in_the_window_is_not_ongoing(demo_tables):
    # furosemide 221794 segment 10:30-16:02 (12:08-16:02 at 5 mg/h); window 12:30-20:00
    s = CERCompiler().compile(demo_tables, 30057454, "2171-11-14 12:30", "2171-11-14 20:00")
    furo = [c for c in s if 221794 in c.source_itemids]
    assert len(furo) == 1 and furo[0].is_state and not furo[0].ongoing_at_delta
    assert furo[0].t_end_h == pytest.approx(3.5333, abs=1e-3) and furo[0].dose_value == pytest.approx(5 * 3.5333, abs=1e-3)
    from spectra.serialize import render_event
    assert render_event(furo[0]).endswith("(stopped +3.5 h)")


@needs_demo
def test_ventilation_stop_charted_before_extubation_joins_the_removal(demo_tables):
    # stay 36091287: invasive ventilation 08:40-21:20, extubation 21:28 (stop 8 min before the removal)
    s = CERCompiler().compile(demo_tables, 36091287, "2119-10-26 12:00", "2119-10-27 08:00")
    prior = Film("10000003", "s4", "d4", 0.0, 0, 720, "36091287", split="test")
    tests, rej = logged_tests("10000003", prior, s, 20.0, "ETT", DeviceState(True), DeviceState(False))
    assert tests and not rej, rej
    t = tests[0]
    assert {c.action_type for c in t.bundle_a if c.t_start_h is not None} >= {"extubation", "invasive_ventilation_stop"}
    vent_b = [c for c in t.bundle_b if c.action_type == "invasive_ventilation" and c.is_state]
    assert vent_b and vent_b[0].ongoing_at_delta and not any(c.action_type == "invasive_ventilation_stop" for c in t.bundle_b)


def test_ambiguous_removal_dependency_is_rejected():
    from spectra.adapters.synthetic import ett_removal_stream
    from spectra.types import CER, Device

    extra = (CER("transition", "extubation", 9.0, device=Device("ETT", "remove", "nasal/oral", "none"), priority="device_procedure"),)
    s = tuple(sorted(ett_removal_stream(5.0) + extra, key=lambda c: c.sort_key()))
    assert partition(s, "ETT", 12.0).ambiguous == ("removal_dependency_association",)
    tests, rej = logged_tests("P", Film("P", "s", "d", 0.0, 0, 0, "1", split="test"), s, 12.0, "ETT", DeviceState(True),
                              DeviceState(False))
    assert not tests and rej[0].rule == "removal_dependency_association"


@needs_demo
def test_line_end_at_discharge_is_not_a_removal(demo_tables):
    # Multi Lumen in stay 35009126 ends 2186-11-17 21:16, ICU outtime 21:15:55: end of documentation
    s = CERCompiler().compile(demo_tables, 35009126, "2186-11-17 08:00", "2186-11-17 22:00")
    assert not [c for c in s if c.device.type == "CVC" and c.device.transition == "remove"]
    assert [c for c in s if c.device.type == "CVC" and c.is_state]


@needs_demo
def test_denylist_never_compiles(demo_tables):
    ont = MimicOntology()
    for stay, t0, tk in [(33281088, "2140-01-24 00:00", "2140-02-09 00:00"), (30757476, "2131-02-27 00:00", "2131-03-09 00:00")]:
        for track in ("A", "B"):
            s = CERCompiler(ont).compile(demo_tables, stay, t0, tk, track=track)
            ids = {i for c in s for i in c.source_itemids}
            assert not (ids & ont.deny_itemids), ids & ont.deny_itemids


@needs_demo
def test_track_b_dependents_withheld_and_hospital_sources_present(demo_tables):
    s = CERCompiler().compile(demo_tables, 33281088, "2140-01-30 05:44", "2140-02-03 15:54", track="B")
    mv = [c for c in s if 224687 in c.source_itemids]          # Minute Volume
    assert mv and all(c.origin == "dependent:ETT" for c in mv)
    assert any(c.action_type.startswith("obs:lab:") for c in s)
    p = partition(s, "ETT", 106.17)
    assert not any(224687 in c.source_itemids for c in p.background + p.common + p.placement + p.removal)
    assert any(224687 in c.source_itemids for c in p.withheld)
    spo2 = [c for c in s if 220277 in c.source_itemids]       # SpO2 is a general observation
    assert spo2 and all(c.origin == "record" for c in spo2)


def test_ett_partition_bundles_synthetic():
    from spectra.adapters.synthetic import ett_placement_stream, ett_removal_stream

    p = partition(ett_placement_stream(3.0), "ETT", 12.0)
    assert sorted(c.action_type for c in p.placement) == ["intubation", "invasive_ventilation", "neuromuscular_blocker",
                                                          "ventilator_setting", "ventilator_setting"]
    assert {c.action_type for c in p.background} == {"diuretic", "antibiotic", "fluid"}
    r = partition(ett_removal_stream(5.0), "ETT", 12.0)
    assert sorted(c.action_type for c in r.removal) == ["extubation", "invasive_ventilation_stop", "noninvasive_ventilation"]
    assert {c.action_type for c in r.common} == {"invasive_ventilation", "ventilator_setting"}
    assert {c.action_type for c in r.background} == {"vasoactive"}


def test_niv_interval_after_extubation_moves_with_its_stop():
    from spectra.adapters.synthetic import ett_removal_stream
    from spectra.types import CER

    s = tuple(sorted(ett_removal_stream(5.0) + (CER("action", "noninvasive_ventilation_stop", 9.0, priority="device_procedure"),),
                     key=lambda c: c.sort_key()))
    r = partition(s, "ETT", 12.0)
    assert "noninvasive_ventilation_stop" in {c.action_type for c in r.removal}
    assert "noninvasive_ventilation_stop" not in {c.action_type for c in r.background}
