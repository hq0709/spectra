"""R010 MIMIC census on a synthetic MIMIC layout (no credentialed data)."""
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[1]


def _layout(tmp: Path):
    jpg, rep, iv = tmp / "jpg", tmp / "rep", tmp / "iv"
    (iv / "icu").mkdir(parents=True)
    jpg.mkdir()
    rep.mkdir()
    rows = []
    # subject 10000001: one ICU stay with 4 studies 6 h apart (+ a lateral); subject 19000002 (p19, forced test):
    # 2 ICU studies; subject 10000003: outpatient PA only
    for i, h in enumerate([0, 6, 12, 18]):
        rows.append(dict(dicom_id=f"d1{i}", subject_id=10000001, study_id=50000010 + i, ViewPosition="AP",
                         Rows=2000, Columns=2000, StudyDate=21500101, StudyTime=float(h * 10000)))
    rows.append(dict(dicom_id="d1x", subject_id=10000001, study_id=50000010, ViewPosition="LATERAL", Rows=2000,
                     Columns=2000, StudyDate=21500101, StudyTime=0.0))
    for i, h in enumerate([1, 3]):
        rows.append(dict(dicom_id=f"d2{i}", subject_id=19000002, study_id=50000020 + i, ViewPosition="AP",
                         Rows=2000, Columns=2000, StudyDate=21500201, StudyTime=float(h * 10000)))
    rows.append(dict(dicom_id="d30", subject_id=10000003, study_id=50000030, ViewPosition="PA", Rows=2000,
                     Columns=2000, StudyDate=21500301, StudyTime=90000.0))
    pd.DataFrame(rows).to_csv(jpg / "mimic-cxr-2.0.0-metadata.csv.gz", index=False)
    pd.DataFrame([dict(dicom_id=r["dicom_id"], study_id=r["study_id"], subject_id=r["subject_id"], split="train")
                  for r in rows]).to_csv(jpg / "mimic-cxr-2.0.0-split.csv.gz", index=False)
    pd.DataFrame([dict(subject_id=10000001, hadm_id=1, stay_id=30000001, intime="2150-01-01 00:00:00",
                       outtime="2150-01-03 00:00:00"),
                  dict(subject_id=19000002, hadm_id=2, stay_id=30000002, intime="2150-02-01 00:00:00",
                       outtime="2150-02-02 00:00:00")]).to_csv(iv / "icu" / "icustays.csv.gz", index=False)
    cfg = tmp / "mimic.yaml"
    cfg.write_text(f"source: mimic\ncxr_jpg_dir: {jpg}\ncxr_reports_dir: {rep}\nmimiciv_dir: {iv}\n"
                   "split_salt: spectra-mimic-v1\nsplit_fractions: {dev_train: 0.6, dev_val: 0.1, test: 0.3}\n")
    return cfg


def test_mimic_census_counts(tmp_path):
    cfg = _layout(tmp_path)
    out = tmp_path / "census.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "r010_mimic_census.py"), "--config", str(cfg),
                    "--out", str(out), "--work", str(tmp_path / "work")], check=True, capture_output=True)
    r = json.loads(out.read_text())
    assert r["0_release"] == {"patients": 3, "studies": 7, "images": 8, "frontal_images": 7}
    e = r["e1"]["all"]
    assert e["1_frontal_f1"] == {"patients": 3, "studies": 7}
    assert e["2_in_icu_stay"]["studies"] == 6 and e["2_in_icu_stay"]["stays"] == 2
    assert e["3_ge3_frontal_in_one_stay"] == {"patients": 1, "stays": 1, "studies": 4}
    # priors at 0 h and 6 h have >= 2 next films within 1 h-7 d; 12 h has one; the p19 pair has one later film
    assert e["4_prior_with_ge2_next_films"]["eligible_priors"] == 2
    assert e["4_prior_with_ge2_next_films"]["queries"] == 5
    assert r["forced_test"]["p19_patients"] == 1 and r["patients_by_split"].get("test", 0) >= 1
    assert (tmp_path / "work" / "films.parquet").exists()
    text = out.read_text()
    assert "10000001" not in text and "5000001" not in text          # no identifiers in the aggregate
