"""R018 census: a GraphQL error or a missing result is never read as an empty last page."""
import importlib.util
from pathlib import Path

import pytest


def _mod():
    p = Path(__file__).resolve().parents[1] / "scripts" / "r018_midrc_census.py"
    spec = importlib.util.spec_from_file_location("r018", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_graphql_errors_raise(monkeypatch):
    m = _mod()
    monkeypatch.setattr(m, "query", lambda q, v=None: {"errors": [{"message": "boom"}], "data": {"case": None}})
    with pytest.raises(RuntimeError, match="boom"):
        m.pages("case", "submitter_id", {})
    monkeypatch.setattr(m, "query", lambda q, v=None: {"data": {}})
    with pytest.raises(RuntimeError):
        m.pages("case", "submitter_id", {})
    calls = iter([{"data": {"case": [{"submitter_id": i} for i in range(3)]}}, {"data": {"case": []}}])
    monkeypatch.setattr(m, "query", lambda q, v=None: next(calls))
    assert len(m.pages("case", "submitter_id", {}, page=3)) == 3


def test_categories_match_the_type_inventory():
    m = _mod()
    c = m.CATEGORIES
    assert c["intubation"]("Insertion of Endotracheal Airway into Trachea, Via Natural or Artificial Opening Endoscopic")
    assert c["invasive_ventilation"]("Respiratory Ventilation, Greater than 96 Consecutive Hours")
    assert not c["invasive_ventilation"]("Assistance with Respiratory Ventilation, 24-96 Consecutive Hours")
    assert not c["invasive_ventilation"]("Respiratory Ventilation, Single, Nonmechanical")
    assert not c["invasive_ventilation"]("Noninvasive ventilation")
