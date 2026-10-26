"""Report sections (M4) and the extractor's output handling (M5), without a GPU or model weights."""
import json

from spectra.references.extract import jsonl_items, record
from spectra.references.sections import extraction_text, sections

MIMIC_STYLE = """                                 FINAL REPORT
 EXAMINATION:  CHEST (PORTABLE AP)

 INDICATION:  ___ year old man with ETT, eval for placement

 COMPARISON:  Chest radiograph from ___ at 04:12.

 FINDINGS:

 Endotracheal tube terminates 4 cm above the carina.  Right internal jugular
 central venous catheter tip is in the mid SVC.  Moderate pulmonary edema is
 slightly improved.  Small left pleural effusion is unchanged.

 IMPRESSION:

 Improving pulmonary edema.  Stable small left effusion.
"""

NO_HEADINGS = """Portable chest.

Lines and tubes are unchanged. Mild interstitial edema has worsened compared to yesterday."""


def test_sections_follow_the_mimic_layout():
    s = sections(MIMIC_STYLE)
    assert set(s) == {"comparison", "findings", "impression"}          # indication and examination are not kept
    assert s["comparison"] == "Chest radiograph from ___ at 04:12."
    assert s["findings"].startswith("Endotracheal tube terminates") and "\n" not in s["findings"]
    assert "Moderate pulmonary edema is slightly improved." in s["findings"]           # fixed-width wrap removed
    assert s["impression"] == "Improving pulmonary edema. Stable small left effusion."
    text = extraction_text(s)
    assert text.splitlines()[0] == "COMPARISON: Chest radiograph from ___ at 04:12."
    assert text.index("FINDINGS:") < text.index("IMPRESSION:")


def test_report_without_headings_keeps_its_last_paragraph():
    s = sections(NO_HEADINGS)
    assert s == {"impression": "Lines and tubes are unchanged. Mild interstitial edema has worsened compared to yesterday."}


def _extraction(text):
    f = {c: {"change": "not_mentioned", "absolute": "not_mentioned", "evidence": ""}
         for c in ("edema", "effusion_L", "effusion_R", "effusion_unsided")}
    f["edema"] = {"change": "improved", "absolute": "moderate", "evidence": "Moderate pulmonary edema is slightly improved."}
    d = {k: {"state": "not_mentioned", "side": "none", "evidence": ""}
         for k in ("ETT", "CVC", "PICC", "enteric", "chest_tube", "tracheostomy")}
    d["ETT"] = {"state": "present", "side": "none", "evidence": "Endotracheal tube terminates 4 cm above the carina."}
    return {"comparison_reference": {"type": "clock_time", "clock_time": "04:12", "day_offset": None, "date": None,
                                     "evidence": "Chest radiograph from ___ at 04:12."},
            "findings": f, "devices": d, "device_enumeration": False}


def test_record_accepts_valid_and_rejects_with_a_reason():
    text = extraction_text(sections(MIMIC_STYLE))
    ok = record("s1", text, json.dumps(_extraction(text)), "stop", 300)
    assert ok["rejected"] is None and ok["extraction"]["findings"]["edema"]["change"] == "improved"
    bad = _extraction(text)
    bad["findings"]["edema"]["evidence"] = "moderate edema, improved"          # paraphrase, not a verbatim span
    r = record("s1", text, json.dumps(bad), "stop", 300)
    assert r["rejected"].startswith("evidence:") and "findings.edema:not_substring" in r["rejected"]
    assert record("s1", text, '{"comparison_reference": ', "length", 1536)["rejected"] == "truncated"
    assert record("s1", text, "{not json", "stop", 3)["rejected"].startswith("json:")


def test_jsonl_items(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"id": 7, "text": "FINDINGS: clear"}) + "\n\n")
    assert jsonl_items(str(p)) == [("7", "FINDINGS: clear")]


def test_date_references_must_be_iso_calendar_dates():
    text = "COMPARISON: Two views of the chest dated XXXX.\nFINDINGS: No pleural effusion."
    ex = _extraction(text)
    ex["findings"]["edema"] = {"change": "not_mentioned", "absolute": "not_mentioned", "evidence": ""}
    ex["devices"]["ETT"] = {"state": "not_mentioned", "side": "none", "evidence": ""}
    for bad in ("XXXX", "2019-02-30", "3/12/2019"):
        ex["comparison_reference"] = {"type": "date", "date": bad, "evidence": "Two views of the chest dated XXXX."}
        assert record("r", text, json.dumps(ex), "stop", 9)["rejected"] == "reference_field_invalid:date"
    ex["comparison_reference"]["date"] = "2019-02-03"
    assert record("r", text, json.dumps(ex), "stop", 9)["rejected"] is None


def test_resume_refuses_another_prompt_or_unversioned_output(tmp_path):
    import pytest

    from spectra.cli import main

    src = tmp_path / "r.jsonl"
    src.write_text(json.dumps({"id": "a", "text": "FINDINGS: clear"}) + "\n")
    out = tmp_path / "out"
    out.mkdir()
    (out / "extractions.0000-of-0001.jsonl").write_text(json.dumps(
        {"report_id": "a", "chars": 15, "finish_reason": "stop", "output_tokens": 3, "extraction": None,
         "rejected": "json:x"}) + "\n")
    argv = ["extract-reports", "--source", "jsonl", "--config", str(src), "--out", str(out)]
    with pytest.raises(SystemExit, match="without run.json"):
        main(argv)
    (out / "run.json").write_text(json.dumps({"source": "jsonl", "model": "Qwen/Qwen3-32B", "prompt": "v1"}))
    with pytest.raises(SystemExit, match="fresh --out"):
        main(argv)
    from spectra.references.schema import PROMPT_VERSION

    (out / "run.json").write_text(json.dumps({"source": "jsonl", "model": "Qwen/Qwen3-32B", "prompt": PROMPT_VERSION}))
    assert main(argv) == 0                        # same run, nothing left to do: no model is loaded
    assert json.loads((out / "summary.0000-of-0001.json").read_text())["prompt"] == PROMPT_VERSION
