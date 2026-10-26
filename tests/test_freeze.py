"""E-d freeze: each channel scored by its E-a head, by its E-c head, or unscored; never overwritten; read-only."""
import importlib.util
import json
from pathlib import Path

import pytest


def _mod():
    p = Path(__file__).resolve().parents[1] / "scripts" / "r019_freeze.py"
    spec = importlib.util.spec_from_file_location("r019_freeze", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _metrics(passed):
    return {"channels": {ch: {"qwk": {"estimate": 0.7 if ok else 0.4}, "criterion": {"pass": ok}} for ch, ok in passed.items()},
            "features_meta": {"encoder_revision": "abc", "rule": "to_eval"}}


def test_channel_status_and_read_only(tmp_path):
    m = _mod()
    heads = tmp_path / "heads"
    heads.mkdir()
    (heads / "heads.pt").write_bytes(b"x")
    (heads / "metrics.json").write_text(json.dumps(_metrics({"edema": False, "effusion_L": True, "effusion_R": True})))
    ad = tmp_path / "ad"
    ad.mkdir()
    (ad / "adapted.pt").write_bytes(b"y")
    ec = {"revalidation": {"qwk": {"estimate": 0.5}, "criterion": {"pass": False}}, "note": "epoch 2 kept", "best_epoch": 2}
    (ad / "metrics.json").write_text(json.dumps(ec))
    out = tmp_path / "frozen"
    assert m.main(["--heads", str(heads), "--adapted", f"edema={ad}", "--out", str(out)]) == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["channels"]["edema"]["status"] == "unscored" and man["channels"]["effusion_L"]["head"] == "E-a"
    assert not (out / "adapted_edema.pt").exists()
    assert not (out / "manifest.json").stat().st_mode & 0o222                     # read-only
    with pytest.raises(SystemExit, match="never overwritten"):
        m.main(["--heads", str(heads), "--out", str(out)])
    ec["revalidation"]["criterion"]["pass"] = True
    (ad / "metrics.json").write_text(json.dumps(ec))
    out2 = tmp_path / "frozen2"
    m.main(["--heads", str(heads), "--adapted", f"edema={ad}", "--out", str(out2)])
    man2 = json.loads((out2 / "manifest.json").read_text())
    assert man2["channels"]["edema"] == {**man2["channels"]["edema"], "status": "scored", "head": "E-c",
                                         "weights": "adapted_edema.pt"}
