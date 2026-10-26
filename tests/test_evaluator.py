"""Evaluator pieces that need no GPU: the evaluation image rule, acceptance metrics, heads and the test-split guard."""
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")


def test_eval_image_pads_instead_of_cropping():
    Image = pytest.importorskip("PIL.Image")
    from spectra.images import to_eval

    a = np.full((421, 512), 1000, dtype=np.uint16)      # landscape 12-bit film in a 16-bit container (TAIX-Ray)
    a[:, :6] = a[:, -6:] = 4095                            # bright lateral margins (chest walls, costophrenic angles)
    out = to_eval(Image.fromarray(a))
    g = np.asarray(out)
    assert out.mode == "L" and out.size == (512, 512)
    assert g[256, :3].min() > 200 and g[256, -3:].min() > 200    # the lateral margins survive
    assert g[:40].max() == 0 and g[-40:].max() == 0              # black padding above and below
    rgb = Image.fromarray(np.stack([np.asarray(out)] * 3, -1))
    assert np.array_equal(np.asarray(to_eval(rgb)), g)          # RGB outputs are read by luminance
    big = Image.fromarray(np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1)))
    assert to_eval(big).size == (512, 512) and to_eval(big, via=256).size == (512, 512)


def test_weighted_kappa_and_discretization():
    from spectra.evaluator.metrics import discretize, weighted_kappa

    a, b = [0, 1, 2], [0, 2, 2]
    assert weighted_kappa(a, b, 3, "quadratic") == pytest.approx(0.8)       # worked by hand: 1 - 0.25 / 1.25
    assert weighted_kappa(a, b, 3, "linear") == pytest.approx(2 / 3)        # 1 - 0.5 / 1.5
    assert weighted_kappa([0, 1, 4, 2], [0, 1, 4, 2], 5) == pytest.approx(1.0)
    assert discretize([0.49, 0.5, 1.5, 2.5, 4.7, -0.2], 5).tolist() == [0, 1, 2, 3, 4, 0]   # half up, clipped


def test_patient_mean_sd():
    from spectra.evaluator.metrics import patient_mean_sd

    r = patient_mean_sd([0.0, 1.0, 1.0, 0.0])
    assert r["mean"] == 0.5 and abs(r["sd"] - 0.57735) < 1e-4 and r["N"] == 4
    assert np.isnan(patient_mean_sd([0.3])["sd"])


def test_heads_mask_and_readouts():
    torch = pytest.importorskip("torch")
    from spectra.evaluator.heads import Heads, masked_loss, readouts

    torch.manual_seed(0)
    h = Heads({"edema": 5, "ETT": 2}, dim=32, n_heads=4, hidden=16)
    x = torch.randn(6, 5, 32)
    lg = h(x)
    assert lg["edema"].shape == (6, 5) and lg["ETT"].shape == (6, 2)
    y = torch.tensor([[2, 1]] * 6)
    m = torch.tensor([[True, False]] * 3 + [[False, True]] * 3)
    l1 = masked_loss(lg, y, m, ["edema", "ETT"])
    y2 = y.clone()
    y2[:3, 1] = 0                                   # masked entries do not enter the loss
    y2[3:, 0] = 4
    assert torch.allclose(l1, masked_loss(lg, y2, m, ["edema", "ETT"]))
    r = readouts(lg)
    assert ((r["edema"] >= 0) & (r["edema"] <= 4)).all() and ((r["ETT"] >= 0) & (r["ETT"] <= 1)).all()


def test_feature_script_refuses_test_films_before_the_freeze(tmp_path):
    pytest.importorskip("torch")
    import importlib.util

    p = Path(__file__).resolve().parents[1] / "scripts" / "r019_features.py"
    spec = importlib.util.spec_from_file_location("r019_features", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = tmp_path / "taix.yaml"
    cfg.write_text(f"source: taix\nmetadata: {tmp_path}/m.parquet\nimages: {tmp_path}\n")
    with pytest.raises(SystemExit, match="freeze"):
        mod.main(["--config", str(cfg), "--out", str(tmp_path / "o"), "--splits", "val", "test"])


@pytest.mark.skipif(os.environ.get("SPECTRA_EVALUATOR_WEIGHTS") != "1", reason="needs the cached RAD-DINO weights")
def test_rad_dino_tokens_shape():
    torch = pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")
    from spectra.evaluator.encoder import RadDino, pixel_values, processor

    enc = RadDino(device="cpu", dtype="float32")
    ims = [Image.fromarray(np.random.default_rng(0).integers(0, 4096, (421, 512)).astype(np.uint16))] * 2
    t = enc.tokens(pixel_values(ims, processor()))
    assert t.shape == (2, 65, 768) and torch.isfinite(t).all()
    assert torch.allclose(t[0], t[1])


@pytest.mark.skipif(os.environ.get("SPECTRA_EVALUATOR_WEIGHTS") != "1", reason="needs the cached RAD-DINO processor")
def test_uint8_batches_give_the_processor_output_exactly():
    torch = pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")
    from spectra.evaluator.encoder import eval_u8, pixel_values, pixel_values_u8, processor

    rng = np.random.default_rng(1)
    ims = [Image.fromarray(rng.integers(0, 4096, (421, 512)).astype(np.uint16)),
           Image.fromarray(rng.integers(0, 255, (600, 480)).astype(np.uint8)).convert("RGB")]
    p = processor()
    u8 = torch.from_numpy(np.stack([eval_u8(im) for im in ims]))
    assert torch.equal(pixel_values_u8(u8, p), pixel_values(ims, p))


def test_ranzcr_presence_any_grade_and_a_fixed_patient_split(tmp_path):
    """RANZCR device tasks are presence at any grade of the device's columns; the val split is a salted hash of the
    patient, so every film of a patient lands in one split and the split never depends on the file order."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("torch")                     # evaluator.data defines torch datasets
    from spectra.evaluator.data import RANZCR_COLUMNS, ranzcr_table

    cols = [c for cs in RANZCR_COLUMNS.values() for c in cs]
    rows = []
    for i in range(400):
        r = {c: 0 for c in cols}
        r.update({"StudyInstanceUID": f"1.2.{i}", "PatientID": f"p{i // 4}"})
        if i % 3 == 0:
            r["ETT - Borderline"] = 1
        if i % 5 == 0:
            r["NGT - Incompletely Imaged"] = 1
        rows.append(r)
    csv = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    t = ranzcr_table(str(csv), "/imgs", 0.2, "salt")
    assert t.loc[t.uid == "1.2.0", ["airway_tube", "cvc_any", "enteric"]].values.tolist() == [[1, 0, 1]]
    assert t.loc[t.uid == "1.2.1", ["airway_tube", "cvc_any", "enteric"]].values.tolist() == [[0, 0, 0]]
    assert t.groupby("patient")["split"].nunique().max() == 1
    assert 0.1 < (t.groupby("patient")["split"].first() == "val").mean() < 0.3
    t2 = ranzcr_table(str(csv), "/imgs", 0.2, "salt")
    pd.DataFrame(rows[::-1]).to_csv(csv, index=False)
    t3 = ranzcr_table(str(csv), "/imgs", 0.2, "salt").set_index("uid").loc[t2.uid]
    assert (t3["split"].values == t2["split"].values).all() and t.path[0] == "/imgs/1.2.0.jpg"


def test_rank_auroc_equals_the_pairwise_definition():
    from spectra.e1 import auroc
    from spectra.evaluator.metrics import rank_auroc

    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=300), 1)            # ties on purpose
    y = (rng.random(300) < 0.3).astype(int)
    assert abs(rank_auroc(s, y) - auroc(s, y)) < 1e-12
    assert np.isnan(rank_auroc(s, np.zeros(300, int)))
