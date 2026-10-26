"""Unit tests for the remaining Gate-0 components: E1 helpers, resolver R1-R5, validation intervals,
PPI++, Holm decision table, aggregate export, splits, TAIX-Ray funnel reproduction."""
import json

import numpy as np
import pandas as pd
import pytest

from spectra.decision import Elementary, decide, holm
from spectra.controls import PopulationRate
from spectra.e1 import E1Contrast, absolute_error, auroc, kappa, query_sets
from spectra.export_aggregate import ExportRefused, check, export
from spectra import ppi
from spectra.ppi import Stratum, bootstrap, draw_patients, draw_unit, estimate, ppi_lambda, stratum_estimate
from spectra.references.device_states import follow_up_state, initial_state
from spectra.references.resolver import (consistent, merge_orderings, orderings_from_grades, orderings_from_link, resolve,
                                         resolve_links)
from spectra.references.schema import validate_extraction
from spectra.references.validate import binomial_sd, initial_state_accepted, reference_type_accepted
from spectra.splits import assign, mimic_forced_test
from spectra.types import DeviceState, Film, Ordering

from conftest import TAIX_COUNTS, TAIX_META, needs_taix


def F(pid, sid, t_h, stay="s1", clock=None):
    return Film(pid, sid, sid, t_h, int(t_h // 24), clock if clock is not None else int((t_h % 24) * 60), stay)


# ---- E1 -------------------------------------------------------------------------------------
def test_query_set_stops_at_first_film_outside_horizon_and_stay():
    fs = [F("p", "a", 0), F("p", "b", 10), F("p", "c", 200), F("p", "d", 210), F("p", "e", 220, stay="s2")]
    qs = {q.prior.study_id: [f.study_id for f in q.later] for q in query_sets(fs, 1.0, 168.0, 3)}
    assert qs["a"] == ["b"]          # c is 200 h after a: outside 7 d, the set stops
    assert qs["c"] == ["d"]          # e belongs to another stay: the set stops
    fs2 = [F("p", "a", 0), F("p", "b", 0.5), F("p", "c", 5)]
    assert {q.prior.study_id: [f.study_id for f in q.later] for q in query_sets(fs2, 1.0, 168.0, 3)}.get("a") is None


def test_kappa_and_auroc():
    assert kappa(1.0, 2.0, +1) == 1.0 and kappa(1.0, 2.0, -1) == 0.0 and kappa(2.0, 2.0, +1) == 0.5
    assert auroc([0.9, 0.8, 0.1], [1, 1, 0]) == 1.0
    assert auroc([0.5, 0.5], [1, 0]) == 0.5


def test_absolute_error_is_patient_macro_over_samples():
    # patient A: one query, per-sample errors 1 and 3 -> 2; patient B: errors (0, 0) and (0, 4) -> 1
    rd = {("q1", 1): {"edema": 1.0}, ("q1", 2): {"edema": 5.0}, ("q2", 1): {"edema": 0.0}, ("q2", 2): {"edema": 0.0},
          ("q3", 1): {"edema": 0.0}, ("q3", 2): {"edema": 4.0}}
    ae = absolute_error(["q1", "q2", "q3"], rd, {"q1": 2.0, "q2": 0.0, "q3": 0.0}, "edema", [1, 2],
                        {"q1": "A", "q2": "B", "q3": "B"})
    assert ae == 1.5   # query-macro or pooled samples give 4/3; error of the seed-mean readout gives 1


def test_delta_bins_close_the_final_endpoint_and_reject_out_of_range():
    pr = PopulationRate({}, [1, 12, 48, 168, 720, 4320])
    assert [pr.bin_of(h) for h in (1, 11.99, 12, 168, 719.9, 720, 4320)] == [0, 0, 1, 3, 3, 4, 4]
    for bad in (0.5, 4320.01):
        with pytest.raises(ValueError):
            pr.bin_of(bad)


# ---- resolver ---------------------------------------------------------------------------------
def test_r1_explicit_references_must_match_exactly_one_film():
    fs = [F("p", "a", 24 * 10 + 8, clock=8 * 60), F("p", "b", 24 * 11 + 6, clock=6 * 60), F("p", "c", 24 * 11 + 20, clock=20 * 60),
          F("p", "y", 24 * 12 + 7, clock=7 * 60)]
    y = fs[-1]
    r = resolve(y, fs, {"type": "relative_day", "day_offset": -2})
    assert r.rule == "resolved" and r.film.study_id == "a" and r.provenance == "comparison_explicit"
    assert resolve(y, fs, {"type": "relative_day", "day_offset": -1}).rule == "R1"   # two films yesterday
    r = resolve(y, fs, {"type": "relative_day", "day_offset": -1, "clock_time": "06:20"})
    assert r.rule == "resolved" and r.film.study_id == "b"                          # within 30 min
    assert resolve(y, fs, {"type": "clock_time", "clock_time": "13:00"}).rule == "R1"
    assert resolve(y, fs, {"type": "multiple"}).rule == "R4"
    r = resolve(y, fs, {"type": "none"})
    assert r.film.study_id == "c" and r.provenance == "comparison_default"


def test_r1_explicit_reference_without_its_value_is_unresolved():
    fs = [F("p", "a", 24 * 10 + 8, clock=8 * 60), F("p", "y", 24 * 11 + 7, clock=7 * 60)]
    assert resolve(fs[-1], fs, {"type": "relative_day", "day_offset": None}).rule == "R1"
    assert resolve(fs[-1], fs, {"type": "clock_time", "clock_time": ""}).rule == "R1"
    assert resolve(fs[-1], fs, {"type": "relative_day", "day_offset": -1}).rule == "resolved"


def test_r5_second_film_within_15_minutes():
    fs = [F("p", "a", 100.0), F("p", "a2", 100.1), F("p", "y", 130.0)]
    assert resolve(fs[-1], fs, {"type": "none"}).rule == "R5"
    # the current film itself counts: y 12 min after its comparison film
    fs = [F("p", "a", 100.0), F("p", "y", 100.2)]
    assert resolve(fs[-1], fs, {"type": "none"}).rule == "R5"


def _ext(edema_change="not_mentioned", edema_abs="not_mentioned", ref=None):
    ev = lambda lab: "" if lab == "not_mentioned" else "x"
    return {"comparison_reference": ref or {"type": "none", "evidence": ""},
            "findings": {"edema": {"change": edema_change, "absolute": edema_abs, "evidence": ev(edema_change)}},
            "devices": {}}


def test_resolve_links_applies_r3_to_default_links():
    fs = [F("p", "a", 0.0), F("p", "b", 10.0), F("p", "c", 20.0)]
    ext = {"a": _ext(edema_abs="mild"), "b": _ext("new", "moderate"), "c": _ext("improved", "mild")}
    out, counts = resolve_links(fs, ext)
    # b says "new" edema while a states mild edema: R3; c's default link to b is consistent
    assert counts == {"no_prior": 1, "R3": 1, "resolved": 1}
    assert [(o.earlier_study, o.later_study, o.channel, o.direction) for o in out] == [("b", "c", "edema", -1)]
    # a default link whose comparison report has no accepted extraction cannot be checked: R3
    out, counts = resolve_links(fs, {"b": ext["b"], "c": ext["c"]})
    assert counts.get("R3") == 1 and counts.get("resolved") == 1


def test_r3_uses_the_side_attribution_of_the_orderings():
    prev = {"findings": {"effusion_L": {"change": "not_mentioned", "absolute": "mild", "evidence": "small left effusion"}},
            "devices": {"CVC": {"state": "repositioned"}}}
    cur = {"findings": {"effusion_L": {"change": "not_mentioned", "absolute": "moderate", "evidence": "left effusion"},
                        "effusion_unsided": {"change": "new", "absolute": "not_mentioned", "evidence": "the effusion is new"}},
           "devices": {}}
    assert not consistent(cur, prev)                 # "new" is attributed to the left side, stated present before
    fs = [F("p", "a", 0.0), F("p", "b", 10.0)]
    out, counts = resolve_links(fs, {"a": prev, "b": dict(cur, comparison_reference={"type": "none", "evidence": ""})})
    assert counts.get("R3") == 1 and out == []
    assert not consistent({"findings": {}, "devices": {"CVC": {"state": "placed_interval"}}}, prev)   # repositioned = present


def test_unsided_effusion_change_is_attributed_through_absolute_statements():
    # "small left effusion" (absolute only) + "the effusion has increased" (unsided change) -> left worsened
    rep = {"findings": {"edema": {"change": "not_mentioned"},
                        "effusion_L": {"change": "not_mentioned", "absolute": "mild", "evidence": "small left effusion"},
                        "effusion_R": {"change": "not_mentioned", "absolute": "not_mentioned", "evidence": ""},
                        "effusion_unsided": {"change": "worsened", "absolute": "not_mentioned",
                                             "evidence": "the effusion has increased"}}}
    os_ = orderings_from_link(F("p", "a", 0), F("p", "b", 10), rep, "comparison_default")
    assert [(o.channel, o.direction, o.evidence) for o in os_] == [("effusion_L", 1, "the effusion has increased")]
    # both sides named (bilateral): the unsided change cannot be attributed
    bil = json.loads(json.dumps(rep))
    bil["findings"]["effusion_R"] = {"change": "not_mentioned", "absolute": "mild", "evidence": "right effusion"}
    assert orderings_from_link(F("p", "a", 0), F("p", "b", 10), bil, "comparison_default") == []
    # a sided statement that contradicts the unsided one leaves the side unresolved
    con = json.loads(json.dumps(rep))
    con["findings"]["effusion_L"]["change"] = "improved"
    assert orderings_from_link(F("p", "a", 0), F("p", "b", 10), con, "comparison_default") == []


def test_r3_consistency_and_unsided_effusion():
    prev = {"findings": {"edema": {"absolute": "mild"}}, "devices": {"CVC": {"state": "present"}}}
    assert not consistent({"findings": {"edema": {"change": "new"}}, "devices": {}}, prev)
    assert not consistent({"findings": {}, "devices": {"CVC": {"state": "placed_interval"}}}, prev)
    assert consistent({"findings": {"edema": {"change": "improved"}}, "devices": {}}, prev)
    rep = {"findings": {"edema": {"change": "unchanged", "evidence": "x"}, "effusion_L": {"change": "not_mentioned"},
                        "effusion_R": {"change": "worsened", "evidence": "right effusion"},
                        "effusion_unsided": {"change": "worsened", "evidence": "effusion larger"}}}
    os_ = orderings_from_link(F("p", "a", 0), F("p", "b", 10), rep, "comparison_default")
    got = {(o.channel, o.direction) for o in os_}
    assert got == {("edema", 0), ("effusion_R", 1)}


def test_merge_drops_links_that_conflict_with_absolute_grades():
    fs = [F("p", "a", 0), F("p", "b", 10)]
    grades = {"a": {"edema": 1}, "b": {"edema": 3}}
    g = orderings_from_grades(fs, grades, ["edema"])
    link = [Ordering("a", "b", "edema", -1, "comparison_default")]
    assert ("a", "b", "edema") not in merge_orderings(link, g)
    ok = [Ordering("a", "b", "edema", +1, "comparison_default")]
    assert merge_orderings(ok, g)[("a", "b", "edema")].direction == 1


def test_device_state_tiers_records_never_suffice():
    assert initial_state("CVC", None, None, records_active=True) == (None, None)
    s, tier = initial_state("CVC", {"devices": {"CVC": {"state": "present", "side": "right"}}}, None, False)
    assert s == DeviceState(True, "R") and tier == "tier1"
    nxt = {"devices": {"ETT": {"state": "placed_interval", "side": "none"}}}
    assert initial_state("ETT", None, nxt, False) == (None, None)       # next report not linked to this prior
    s, tier = initial_state("ETT", None, nxt, False, next_links_prior=True)
    assert s == DeviceState(False) and tier == "tier1_next"
    enum = {"device_enumeration": True, "devices": {"ETT": {"state": "not_mentioned"}}}
    assert initial_state("ETT", enum, None, False, tier2_admitted=False) == (None, None)
    assert initial_state("ETT", enum, None, False, tier2_admitted=True) == (DeviceState(False), "tier2")
    assert initial_state("ETT", enum, None, True, tier2_admitted=True) == (None, None)
    assert follow_up_state("ETT", {"devices": {"ETT": {"state": "removed_interval"}}}) == DeviceState(False)


def test_extraction_evidence_must_be_verbatim():
    report = "Right internal jugular line in place. No pleural effusion."
    good = {"comparison_reference": {"type": "none", "evidence": ""},
            "findings": {c: {"change": "not_mentioned", "absolute": "not_mentioned", "evidence": ""}
                         for c in ["edema", "effusion_L", "effusion_R", "effusion_unsided"]},
            "devices": {d: {"state": "not_mentioned", "side": "none", "evidence": ""}
                        for d in ["ETT", "CVC", "PICC", "enteric", "chest_tube", "tracheostomy"]},
            "device_enumeration": False}
    good["devices"]["CVC"] = {"state": "present", "side": "right", "evidence": "Right internal jugular line in place"}
    assert validate_extraction(good, report) is None
    bad = json.loads(json.dumps(good))
    bad["devices"]["CVC"]["evidence"] = "right IJ catheter"
    assert validate_extraction(bad, report) == "evidence:devices.CVC:not_substring"
    bad["devices"]["CVC"]["evidence"] = "Right internal jugular ... in place"        # ellipsis-joined span
    assert validate_extraction(bad, report) == "evidence:devices.CVC:not_substring"
    case = json.loads(json.dumps(good))
    case["devices"]["CVC"]["evidence"] = "right internal  jugular line"            # case and spacing only
    assert validate_extraction(case, report) is None
    empty = json.loads(json.dumps(good))
    empty["devices"]["CVC"]["evidence"] = ""                     # asserted state without evidence
    assert validate_extraction(empty, report) == "evidence:devices.CVC:missing"
    eff = json.loads(json.dumps(good))
    eff["findings"]["effusion_unsided"] = {"change": "absent", "absolute": "absent", "evidence": ""}
    assert validate_extraction(eff, report).startswith("schema:")   # "absent" is not a change label
    eff["findings"]["effusion_unsided"] = {"change": "not_mentioned", "absolute": "absent", "evidence": ""}
    assert validate_extraction(eff, report) == "evidence:findings.effusion_unsided:missing"
    ref = json.loads(json.dumps(good))
    ref["comparison_reference"] = {"type": "relative_day", "day_offset": None, "evidence": "Right internal"}
    assert validate_extraction(ref, report) == "reference_field_missing:day_offset"
    ref["comparison_reference"]["day_offset"] = -1
    assert validate_extraction(ref, report) is None


# ---- reference validation (point estimates + SD) ---------------------------------------------------
def test_reference_acceptance_uses_point_estimates_and_sd():
    assert binomial_sd(90, 100) == pytest.approx(0.03)
    ok, r = reference_type_accepted(285, 300)
    assert ok and r["precision"] == 0.95 and r["sd"] > 0 and set(r) == {"precision", "sd", "n"}
    assert not reference_type_accepted(17, 20)[0]
    assert initial_state_accepted(1, 20)[0] and not initial_state_accepted(2, 20)[0]


# ---- PPI++ ---------------------------------------------------------------------------------------
def test_ppi_corrects_a_biased_evaluator():
    rng = np.random.default_rng(0)
    n_pool, n_aud = 2000, 150
    truth = rng.random(n_pool) < 0.7                     # human-read correctness
    auto = np.where(rng.random(n_pool) < 0.85, truth, ~truth) & (rng.random(n_pool) < 0.95)  # biased low
    pats = [f"p{i}" for i in range(n_pool)]
    pool = dict(zip(pats, auto.astype(float)))
    aud_p = draw_patients(pats, n_aud, seed=1, stratum="mimic|E2|ETT")
    audit = {p: (float(truth[pats.index(p)]), pool[p]) for p in aud_p}
    s = Stratum("mimic|E2|ETT", 1.0, pool, audit)
    naive = float(np.mean(list(pool.values())))
    est = estimate([s])["estimate"]
    b = bootstrap([s], replicates=300, seed=2)
    assert abs(est - truth.mean()) < abs(naive - truth.mean())
    assert b["se"] > 0 and abs(b["estimate"] - truth.mean()) < 3 * b["se"] and "ci" not in b
    assert estimate([s], lam=1.0)["strata"]["mimic|E2|ETT"]["lambda"] == 1.0
    u, seed = draw_unit("p1", ["t2", "t1"], [1009, 1010], seed=1, stratum="x")
    assert u in ("t1", "t2") and seed in (1009, 1010)
    assert draw_unit("p1", ["t2", "t1"], [1009, 1010], seed=1, stratum="x") == (u, seed)


def test_ppi_lambda_for_the_nested_two_stage_design():
    # hand-computed: Var(Z) = Cov(H, Z) = 1/3, Cov(H, X) = Cov(Z, X) = 0, Var(X_pool) = 2/7, f = 4/8
    # lambda = (1/3) / (1/3 + (1/2)(2/7)) = 0.7; classical PPI++ (no f terms) would give 1
    x = [0, 0, 1, 1, 0, 0, 1, 1]
    s = Stratum("x", 1.0, {f"p{i}": float(v) for i, v in enumerate(x)},
                {"p0": (0.0, 0.0), "p1": (1.0, 1.0), "p2": (0.0, 0.0), "p3": (1.0, 1.0)})
    mu, lam = stratum_estimate(s)
    assert lam == pytest.approx(0.7) and mu == pytest.approx(0.7 * 0.5 + 0.3 * 0.5)
    # constant automated scores carry no information: lambda = 0, the estimate is the audit mean
    c = Stratum("c", 1.0, {f"p{i}": 0.5 for i in range(8)}, {"p0": (1.0, 0.5), "p1": (0.0, 0.5), "p2": (1.0, 0.5)})
    mu, lam = stratum_estimate(c)
    assert lam == 0.0 and mu == pytest.approx(2 / 3)
    assert ppi_lambda(np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0


def test_bootstrap_re_estimates_lambda_in_every_replicate(monkeypatch):
    calls = []
    real = ppi.ppi_lambda
    monkeypatch.setattr(ppi, "ppi_lambda", lambda *a: calls.append(1) or real(*a))
    rng = np.random.default_rng(3)
    pool = {f"p{i}": float(v) for i, v in enumerate(rng.random(60))}
    audit = {p: (float(pool[p] > 0.5), pool[p]) for p in list(pool)[:20]}
    out = bootstrap([Stratum("s", 1.0, pool, audit)], replicates=25, seed=1)
    assert len(calls) >= 1 + out["replicates_used"] and out["replicates_used"] >= 20


# ---- decision table --------------------------------------------------------------------------------
def test_holm_and_declarations():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}
    res = {"H1": Elementary("H1", 0.005, 0.005), "H2": Elementary("H2", 0.06, 0.01), "H5": Elementary("H5", 0.05, 0.01)}
    d = decide(res, rung=0)
    assert d["declarations"]["H1"] == "does not exceed prior+delta by the margin"
    assert d["declarations"]["H2"] == "exceeds prior+delta by more than the margin"
    assert d["declarations"]["H5"] == "S2 beats copy-prior on conventional AUROC"
    assert d["claims"]["divergence"] is False   # H4 is not in the Rung-0 family
    full = dict(res, H3=Elementary("H3", 0.95, 0.02), H4=Elementary("H4", 0.70, 0.04),
                H6=Elementary("H6", 0.80, 0.03), H7=Elementary("H7", 0.88, 0.05))
    d1 = decide(full, rung=1)
    assert d1["declarations"]["H4"] == "substantial specification failure" and d1["claims"]["divergence"] is True
    assert d1["declarations"]["H7"] == "not declared" and d1["claims"]["padchest_failure"] is True
    with pytest.raises(ValueError):
        decide(full, rung=2)                     # H8, H9 missing


def test_zero_or_nonfinite_uncertainty_is_not_declarable():
    res = {"H1": Elementary("H1", 0.005, 0.0), "H2": Elementary("H2", 0.06, float("nan")), "H5": Elementary("H5", 0.05, 0.01)}
    d = decide(res, rung=0)
    assert d["declarations"]["H1"] == d["declarations"]["H2"] == "not declarable (uncertainty not estimable)"
    assert d["holm"]["H1"] == 1.0 and d["holm"]["H2"] == 1.0
    assert d["declarations"]["H5"] == "S2 beats copy-prior on conventional AUROC"   # p = 3e-7, x3 still < 0.05


# ---- export ---------------------------------------------------------------------------------------
def test_export_refuses_patient_level_tables(tmp_path):
    ok = pd.DataFrame({"endpoint": ["JDA_primary"], "system": ["S2"], "estimate": [0.81], "se": [0.03], "n": [110]})
    assert check(ok) == []
    with pytest.raises(ExportRefused):            # the analyst's inspection is required, not implied
        export(ok, tmp_path / "agg.csv", analyst="credentialed analyst", purpose="decision table")
    out = export(ok, tmp_path / "agg.csv", analyst="credentialed analyst", purpose="decision table", analyst_confirmed=True)
    man = json.loads((tmp_path / "agg.csv.manifest.json").read_text())
    assert out.exists() and man["automatic_checks_prove_aggregation"] is False and man["analyst_inspection"]["confirmed"]
    per_patient = pd.DataFrame({"endpoint": ["JDA"] * 3, "system": ["S2"] * 3, "estimate": [1.0, 0.0, 0.5], "n": [1, 1, 1]})
    for bad in (ok.assign(subject_id=[10000032]), ok.assign(note=["right IJ line in place"]),
                ok.assign(value=["10000032"]), ok.assign(value=["right IJ line"]), ok.assign(study_time=["12:00"]),
                ok.assign(system=["right IJ line"]), ok.drop(columns=["n"]), ok.assign(n=[-1]), ok.assign(n=[2.5]),
                per_patient):
        with pytest.raises(ExportRefused):
            export(bad, tmp_path / "bad.csv", analyst="a", purpose="p", analyst_confirmed=True)
    assert "duplicate categorical keys" in " ".join(check(per_patient))


# ---- splits -----------------------------------------------------------------------------------------
def test_splits_are_deterministic_and_respect_forced_test():
    ids = [f"1{i:07d}" for i in range(2000)] + ["19000001", "19000002"]
    forced = mimic_forced_test(ids, ["10000005"])
    a = assign(ids, "salt", {"dev_train": .6, "dev_val": .1, "test": .3}, forced_test=forced)
    b = assign(ids, "salt", {"dev_train": .6, "dev_val": .1, "test": .3}, forced_test=forced)
    assert a == b and a["19000001"] == "test" and a["10000005"] == "test"
    share = np.mean([a[i] == "dev_train" for i in ids if i not in forced])
    assert 0.55 < share < 0.65


# ---- TAIX-Ray funnel reproduction ---------------------------------------------------------------------
@needs_taix
def test_taix_adapter_reproduces_the_measured_funnel():
    from spectra.adapters.taix import TaixAdapter

    measured = json.loads(TAIX_COUNTS.read_text())["by_split"]["test"]
    got = TaixAdapter(str(TAIX_META)).funnel("test")
    for k in ("films", "patient_days", "patients", "same_day_films_dropped", "patients_ge3_patient_days",
              "patients_with_eligible_prior", "eligible_priors", "selected_priors", "patients_with_admitted_contrast"):
        assert got[k] == measured[k], (k, got[k], measured[k])
    assert got["patients_with_admitted_contrast"] == 3567
    m = {"edema": "PulmonaryCongestion", "effusion_L": "PleuralEffusion_Left", "effusion_R": "PleuralEffusion_Right"}
    for c, col in m.items():
        assert got["admitted_contrasts"][c] == measured["admitted_contrasts"][col]
        assert got["patients_with_admitted_contrast_by_channel"][c] == measured["patients_with_admitted_contrast_by_channel"][col]
