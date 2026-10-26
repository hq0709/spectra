"""Runner, CLI shards, packaged resources and bounded MIMIC loading (review round 1: items 17, 19, 20)."""
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from spectra import config
from spectra.cli import main, shard_of
from spectra.runner import marker, run_local

from conftest import DEMO, DEMO_ICU, ROOT, needs_demo, needs_taix


def test_runner_shards_are_resumable(tmp_path):
    out = tmp_path / "noop"
    r = run_local(["noop"], str(out), n_shards=3, gpus=["cpu"], poll_s=0.05)
    assert r == {"done": 3, "failed": []}
    assert sorted(p.name for p in out.glob("noop.*.txt")) == ["noop.0.txt", "noop.1.txt", "noop.2.txt"]
    mtimes = {p.name: p.stat().st_mtime_ns for p in out.glob("noop.*.txt")}
    marker(out, "noop", 1, 3).unlink()
    (out / "noop.1.txt").unlink()
    run_local(["noop"], str(out), n_shards=3, gpus=["cpu"], poll_s=0.05)
    assert (out / "noop.1.txt").exists()
    assert {n: (out / n).stat().st_mtime_ns for n in ("noop.0.txt", "noop.2.txt")} == \
        {n: mtimes[n] for n in ("noop.0.txt", "noop.2.txt")}           # done shards were not re-run


def test_runner_reports_failed_shards(tmp_path):
    r = run_local(["build-e1", "--config", str(tmp_path / "missing.yaml")], str(tmp_path / "x"), n_shards=2,
                  gpus=["cpu"], poll_s=0.05)
    assert r["failed"] == [0, 1] and not marker(tmp_path / "x", "build-e1", 0, 2).exists()


def test_shard_assignment_is_stable_and_covers_every_key():
    keys = [f"p{i}" for i in range(500)]
    got = [shard_of(k, 4) for k in keys]
    assert got == [shard_of(k, 4) for k in keys] and set(got) == {0, 1, 2, 3}


@needs_taix
def test_build_e1_shards_union_to_the_unsharded_result(tmp_path):
    cfg = ROOT / "config" / "taix.yaml"
    assert main(["build-e1", "--config", str(cfg), "--out", str(tmp_path / "one")]) == 0
    for i in range(2):
        assert main(["build-e1", "--config", str(cfg), "--shard", str(i), "--n-shards", "2", "--out", str(tmp_path / "two")]) == 0
    one = pd.read_parquet(next((tmp_path / "one").glob("e1_contrasts.*.parquet")))
    two = pd.concat([pd.read_parquet(p) for p in sorted((tmp_path / "two").glob("e1_contrasts.*.parquet"))])
    key = ["patient_id", "prior_study", "query_j", "query_k", "channel"]
    assert len(one) > 0 and len(one) == len(two)
    pd.testing.assert_frame_equal(one.sort_values(key).reset_index(drop=True), two.sort_values(key).reset_index(drop=True))
    shards = [json.loads(p.read_text()) for p in sorted((tmp_path / "two").glob("e1_funnel.*.json"))]
    assert sum(s["patients"] for s in shards) == json.loads(next((tmp_path / "one").glob("e1_funnel.*.json")).read_text())["patients"]


def test_shared_configuration_ships_inside_the_package(tmp_path, monkeypatch):
    from importlib import resources

    for name in config.SHARED:
        assert resources.files("spectra.resources").joinpath(f"{name}.yaml").is_file()
    assert config.constants()["capacity"]["events_llm"] == 64
    # an audited deviation through SPECTRA_CONFIG_DIR overrides one file and leaves the others packaged
    text = resources.files("spectra.resources").joinpath("constants.yaml").read_text()
    (tmp_path / "constants.yaml").write_text(text.replace("events_llm: 64", "events_llm: 32"))
    monkeypatch.setenv("SPECTRA_CONFIG_DIR", str(tmp_path))
    config._shared.cache_clear()
    try:
        assert config.constants()["capacity"]["events_llm"] == 32
        assert "ETT" in config.dependency_table()
    finally:
        monkeypatch.delenv("SPECTRA_CONFIG_DIR")
        config._shared.cache_clear()
    assert config.constants()["capacity"]["events_llm"] == 64


@pytest.mark.skipif(importlib.util.find_spec("setuptools") is None or importlib.util.find_spec("wheel") is None,
                    reason="setuptools/wheel not available offline")
def test_wheel_carries_the_resources_and_runs_relocated(tmp_path):
    import shutil

    # build from a copy: setuptools writes egg-info and build/ next to the sources (read-only mounts, clean tree)
    src = tmp_path / "src_copy"
    shutil.copytree(ROOT, src, ignore=shutil.ignore_patterns("container", ".pytest_cache", "__pycache__", "*.egg-info",
                                                             "build", "dist"))
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-q", "-w", str(tmp_path),
                    str(src)], check=True, capture_output=True, timeout=600)
    whl = next(tmp_path.glob("spectra-*.whl"))
    names = zipfile.ZipFile(whl).namelist()
    for name in config.SHARED:
        assert f"spectra/resources/{name}.yaml" in names
    # import from the unpacked wheel only (no source tree on the path): constants() and a constructor work
    site = tmp_path / "site"
    zipfile.ZipFile(whl).extractall(site)
    code = ("import spectra, spectra.config as c, spectra.e2 as e;"
            "from spectra.adapters.synthetic import ett_placement_stream, prior_film, ABSENT, PRESENT;"
            "assert str(site) in spectra.__file__;"
            "t,_ = e.logged_tests('P', prior_film('P'), ett_placement_stream(), 12.0, 'ETT', ABSENT, PRESENT);"
            "print(c.constants()['capacity']['events_llm'], len(t))").replace("str(site)", repr(str(site)))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(site)
    r = subprocess.run([sys.executable, "-I", "-c", code.replace("import spectra,", "import sys; sys.path.insert(0, "
                                                                  + repr(str(site)) + "); import spectra,")],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["64", "1"]


@needs_demo
def test_mimic_adapter_loads_only_the_requested_track():
    from spectra.adapters.mimic import MimicAdapter

    a = MimicAdapter("", "", str(DEMO))
    ta = a.icu("A", stay_ids=[30757476], loader="csv")
    tb = a.icu("B", stay_ids=[30757476], loader="csv")
    assert not ta.track_b and ta.labevents is None and ta.outputevents is None
    assert tb.track_b and tb.labevents is not None and len(tb.labevents) > 0
    assert set(ta.procedureevents["stay_id"]) == {30757476}


@needs_demo
@pytest.mark.skipif(importlib.util.find_spec("duckdb") is None, reason="duckdb not installed")
def test_duckdb_bounded_reads_compile_identically(demo_tables):
    from spectra.cer import CERCompiler, IcuTables

    windows = [(33281088, "2140-01-30 05:44", "2140-02-03 15:54"), (30757476, "2131-03-02 00:20", "2131-03-03 08:00"),
               (32391858, "2201-10-30 17:00", "2201-10-31 08:00"), (30057454, "2171-11-14 20:00", "2171-11-15 04:00"),
               (32359580, "2146-06-22 20:00", "2146-06-25 13:00")]
    db = IcuTables.from_duckdb(DEMO_ICU, DEMO / "hosp", [w[0] for w in windows], track_b=True)
    comp = CERCompiler()
    for stay, t0, tk in windows:
        for track in ("A", "B"):
            for cutoff in ("occurrence", "availability"):
                a = comp.compile(demo_tables, stay, t0, tk, track=track, cutoff=cutoff)
                b = comp.compile(db, stay, t0, tk, track=track, cutoff=cutoff)
                assert a == b, (stay, track, cutoff, len(a), len(b))


@needs_demo
def test_cli_compiles_one_window(capsys):
    assert main(["cer", "--icu-dir", str(DEMO_ICU), "--stay", "32391858", "--t0", "2201-10-30 17:00",
                 "--tk", "2201-10-31 08:00"]) == 0
    out = capsys.readouterr().out
    assert "[+0.3 h] central venous catheter placement (left subclavian)" in out


@needs_demo
def test_r004_mimic_server_path_on_a_demo_layout(tmp_path):
    """The MIMIC-server branch of scripts/r004_smoke.py on a fake MIMIC-CXR-JPG layout over the open demo:
    film metadata, rule F1, stay matching, splits, DuckDB window compilation, prior rendering."""
    import importlib.util as iu
    import pickle

    pytest.importorskip("PIL")
    from PIL import Image

    jpg = tmp_path / "mimic-cxr-jpg"
    meta = pd.DataFrame({"dicom_id": ["d1", "d1lat", "d2"], "subject_id": [10027602] * 3,
                         "study_id": [50000001, 50000001, 50000002], "ViewPosition": ["AP", "LATERAL", "AP"],
                         "Rows": [512, 512, 512], "Columns": [512, 512, 512],
                         "StudyDate": [22011030, 22011030, 22011031], "StudyTime": [170000.0, 170000.0, 80000.0]})
    jpg.mkdir()
    meta.to_csv(jpg / "mimic-cxr-2.0.0-metadata.csv.gz", index=False)
    meta.assign(split="train")[["dicom_id", "study_id", "subject_id", "split"]].to_csv(
        jpg / "mimic-cxr-2.0.0-split.csv.gz", index=False)
    d = jpg / "files" / "p10" / "p10027602" / "s50000001"
    d.mkdir(parents=True)
    Image.new("L", (64, 48), 128).save(d / "d1.jpg")
    cfg = tmp_path / "mimic.yaml"
    cfg.write_text(f"source: mimic\ncxr_jpg_dir: {jpg}\ncxr_reports_dir: {tmp_path}\nmimiciv_dir: {DEMO}\n"
                   "split_salt: s\nsplit_fractions: {dev_train: 1.0}\n")
    spec = iu.spec_from_file_location("r004_smoke", ROOT / "scripts" / "r004_smoke.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    uid, img, stream, delta = mod._mimic_prior(str(cfg))
    assert uid == "50000001" and img.mode == "L" and delta == pytest.approx(15.0)
    dev = [(c.action_type, round(c.t_start_h, 2), c.device.side) for c in stream if c.device.type == "CVC"]
    assert ("line_placement", 0.3, "L") in dev
    mod.prepare(tmp_path / "out", mimic_config=str(cfg))
    prep = pickle.loads((tmp_path / "out" / "prep.pkl").read_bytes())
    assert prep["uid"] == "50000001" and len(prep["stream"]) == len(stream) and (tmp_path / "out" / "prior.png").exists()


def test_runner_groups_gpus_per_shard(tmp_path, monkeypatch):
    import subprocess

    seen = []

    class P:
        def __init__(self, cmd, env, stdout, stderr):
            seen.append(env.get("CUDA_VISIBLE_DEVICES"))
            self.returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", P)
    r = run_local(["noop"], str(tmp_path / "g"), n_shards=3, gpus=["0", "1", "2", "3", "4"], poll_s=0.01,
                  gpus_per_shard=2)
    assert r == {"done": 3, "failed": []} and sorted(seen) == ["0,1", "0,1", "2,3"]   # GPU 4 alone holds no shard


def test_cli_run_shards_a_step(tmp_path):
    from spectra.cli import main

    assert main(["run", "--out", str(tmp_path / "r"), "--n-shards", "2", "--gpus", "cpu", "noop"]) == 0
    assert marker(tmp_path / "r", "noop", 1, 2).exists()
