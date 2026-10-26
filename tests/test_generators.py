"""Generation-adapter contract that needs no GPU: state parsing (the failure rule) and registry consistency."""
import json

from spectra.config import systems, template
from spectra.generators import ADAPTERS
from spectra.generators.base import parse_state

GOOD = {"edema": 2, "effusion_left": 0, "effusion_right": 3, "endotracheal_tube": True, "central_venous_catheter": True,
        "central_venous_catheter_side": "right", "picc": False, "enteric_tube": False, "chest_tube": False}


def test_parse_state_reads_the_answer_format():
    s = parse_state("Here is the prediction:\n```json\n" + json.dumps(GOOD) + "\n```")
    assert s == {"grades": {"edema": 2, "effusion_L": 0, "effusion_R": 3},
                 "devices": {"ETT": True, "CVC": True, "PICC": False, "enteric": False, "chest_tube": False}, "CVC_side": "R"}


def test_unparseable_states_are_none():
    bad = [
        "no json here",
        json.dumps({k: v for k, v in GOOD.items() if k != "chest_tube"}),        # a key missing
        json.dumps(dict(GOOD, edema=7)),                                        # grade out of range
        json.dumps(dict(GOOD, endotracheal_tube="yes")),                        # not a boolean
        json.dumps(dict(GOOD, central_venous_catheter_side="upper")),           # unknown side
        "{" + json.dumps(GOOD)[1:-1] + ",}",                                    # invalid JSON
    ]
    assert all(parse_state(b) is None for b in bad)


def test_every_adapter_names_a_registered_system_with_an_env():
    reg = systems()
    for name in ADAPTERS:
        assert name in reg and reg[name].get("env") and reg[name].get("weights"), name
    # every prompted system of the panel has an adapter; reference-family and MIMIC-only systems are built at Gate 3
    prompted = {n for n, s in reg.items() if isinstance(s, dict) and s.get("style") in ("verbal", "compact")}
    assert prompted <= set(ADAPTERS), prompted - set(ADAPTERS)
    for name in ("progemu", "unimedvl", "prism"):
        assert "@" in reg[name]["code"] and len(reg[name]["code"].split("@")[1]) == 40   # pinned commits
    assert template("lingshu_32b").count("{input}") == 1


def test_films_are_rendered_to_8_bits_by_min_max():
    import pytest

    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from spectra.images import to_uint8

    a = np.array([[0, 1000], [2047, 4095]], dtype=np.uint16)          # 12-bit values in a 16-bit container
    out = to_uint8(Image.fromarray(a))
    assert out.mode == "L" and np.asarray(out).tolist() == [[0, 62], [127, 255]]
    assert np.asarray(Image.fromarray(a).convert("RGB")).max() == 255   # the naive conversion clips instead
    flat = to_uint8(Image.fromarray(np.full((2, 2), 7, dtype=np.uint16)))
    assert np.asarray(flat).max() == 0
    rgb = Image.new("RGB", (2, 2))
    assert to_uint8(rgb) is rgb


def test_radedit_masks_stay_on_the_film_and_follow_the_side():
    """The positive control edits only inside the film, never its black padding (a 512 x 421 TAIX-Ray film has
    45-46 padding rows above and below on the 512² canvas), and a sided device is masked on its own side."""
    import pytest

    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL.Image")
    from spectra.generators.hf import DEVICE_REGIONS, TARGET_PROMPTS, device_mask

    m = np.asarray(device_mask("ETT", "none", (512, 421))) > 0
    assert m.any() and not m[:45].any() and not m[467:].any()
    right, left = (np.asarray(device_mask("CVC", s, (421, 512))) > 0 for s in ("R", "L"))
    cols = lambda a: np.nonzero(a.any(axis=0))[0]
    assert cols(right).mean() < 256 < cols(left).mean()               # image left = patient right
    assert cols(right).min() >= 45 and cols(left).max() < 467          # no padding columns either
    assert all((d, p) in TARGET_PROMPTS for d in DEVICE_REGIONS for p in (True, False))
