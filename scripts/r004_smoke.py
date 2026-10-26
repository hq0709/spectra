"""R004: one generation per system on this machine, through the certified interface.

Per system: (1) the round-trip check of the smoke test, with the system's own tokenizer, style and template,
must pass; (2) the adapter generates both arms of one E2 test and one mode-R E1 query from one prior film;
(3) outputs, seconds per generation and peak GPU memory are recorded. The driver prepares the prior film and
the mode-R stream once; each system then runs in its own env and process (`--one`), so a crash or an
out-of-memory error cannot affect the next system.

    python scripts/r004_smoke.py --prior-parquet <TAIX-Ray val shard> --out <dir> [--systems a b] [--gpus 0,2]

The E2 pair is the cross-source protocol: Delta = 24 h, actions unknown, ETT insertion at +1 h, with the
initial state ASSUMED absent (the smoke checks that a system runs, not what it predicts). The mode-R query
is "actions unknown" on TAIX-Ray (no action log) and, on the MIMIC server (`--mimic-config`), the compiled
Track-A window between a dev-split prior and its next film in the same stay. MIMIC-derived outputs (images,
inputs, responses) stay on the MIMIC server; only summary.json and the timing fields are aggregate.

    python scripts/r004_smoke.py --mimic-config config/mimic.yaml --out runs/R004/mimic [--gpus 0]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SEED = 1009  # constants.yaml generation.seed_base


def _no_mps_env() -> dict:
    """Our jobs never use CUDA MPS. Point clients at an empty pipe directory so that another user's MPS
    control daemon on a shared node cannot fail CUDA initialization (error 805)."""
    env = dict(os.environ)
    if "CUDA_MPS_PIPE_DIRECTORY" not in env:
        d = Path(tempfile.gettempdir()) / f"spectra-no-mps-{os.getuid()}"
        d.mkdir(exist_ok=True)
        env["CUDA_MPS_PIPE_DIRECTORY"] = env["CUDA_MPS_LOG_DIRECTORY"] = str(d)
    return env


def _prior(parquet: str):
    import pyarrow.parquet as pq
    from PIL import Image

    from spectra.images import to_uint8

    t = pq.ParquetFile(parquet).read_row_group(0, columns=["UID", "Image"]).slice(0, 1).to_pylist()[0]
    return t["UID"], to_uint8(Image.open(io.BytesIO(t["Image"]["bytes"])))


def _mimic_prior(cfg_path: str):
    """First dev_train prior (by study id) with a next frontal film in the same stay 1 h - 7 d later:
    (study id, 8-bit film, compiled Track-A stream, Delta in hours)."""
    from PIL import Image

    from spectra.adapters.mimic import MimicAdapter
    from spectra.config import source_config
    from spectra.images import to_uint8

    cfg = source_config(cfg_path)
    a = MimicAdapter(cfg["cxr_jpg_dir"], cfg["cxr_reports_dir"], cfg["mimiciv_dir"], salt=cfg["split_salt"],
                     fractions=cfg["split_fractions"])
    by_stay = {}
    for f in sorted(a.films(), key=lambda f: (f.stay_id or "", f.t_h)):
        if f.stay_id and f.split == "dev_train":
            by_stay.setdefault(f.stay_id, []).append(f)
    for prior, nxt in sorted(((fs[i], fs[i + 1]) for fs in by_stay.values() for i in range(len(fs) - 1)
                              if 1.0 <= fs[i + 1].t_h - fs[i].t_h <= 168.0), key=lambda pn: pn[0].study_id):
        img = a.cxr_jpg_dir / "files" / f"p{prior.patient_id[:2]}" / f"p{prior.patient_id}" / f"s{prior.study_id}" / f"{prior.image_id}.jpg"
        if img.exists():
            stream = a.window_stream(prior, nxt, track="A", stay_ids=[int(prior.stay_id)])
            return prior.study_id, to_uint8(Image.open(img)), tuple(stream), nxt.t_h - prior.t_h
    raise RuntimeError("no dev_train prior with a next film in the same stay")


def _prior_corr(prior, out) -> float:
    """Pearson r between 32 x 32 thumbnails of the prior and the output: a follow-up of the same patient
    correlates with its prior; a scrambled or unrelated image does not. A diagnostic, not a pass criterion."""
    import numpy as np

    a = np.asarray(prior.convert("L").resize((32, 32)), float).ravel()
    b = np.asarray(out.convert("L").resize((32, 32)), float).ravel()
    return round(float(np.corrcoef(a, b)[0, 1]), 3) if a.std() > 0 and b.std() > 0 else 0.0


def prepare(out: Path, parquet: str = None, mimic_config: str = None) -> None:
    """Once per run, in the driver's env: the 8-bit prior film (prior.png) and the mode-R stream and Delta
    (prep.pkl), so system processes need neither MIMIC tables nor DuckDB."""
    import pickle

    if mimic_config:
        uid, img, stream, delta_r = _mimic_prior(mimic_config)
    else:
        (uid, img), stream, delta_r = _prior(parquet), None, 24.0
    out.mkdir(parents=True, exist_ok=True)
    img.save(out / "prior.png")
    (out / "prep.pkl").write_bytes(pickle.dumps({"uid": uid, "stream": stream, "delta_r": delta_r}))


def run_one(system: str, out: Path, tid: str = "t1") -> dict:
    import pickle

    import torch
    from PIL import Image

    from spectra.adapters.synthetic import ABSENT, prior_film
    from spectra.config import systems, template
    from spectra.e2 import cross_source_tests
    from spectra.generators import make
    from spectra.roundtrip import Interface, check
    from spectra.tokenizers import HFTokenizer

    spec = systems()[system]
    prep = pickle.loads((out / "prep.pkl").read_bytes())
    uid, stream, delta_r = prep["uid"], prep["stream"], prep["delta_r"]
    img = Image.open(out / "prior.png")
    rec = {"system": system, "prior_uid": uid, "prior_size": list(img.size), "host": platform.node(),
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "torch": torch.__version__, "outputs": []}
    test = cross_source_tests("smoke", prior_film("smoke"), "ETT", ABSENT, source="taix")[0][0]
    tok = None
    if spec.get("tokenizer"):
        tok = HFTokenizer(spec["tokenizer"], max_len=spec.get("max_tokens"), subfolder=spec.get("tokenizer_subfolder"),
                          trust_remote_code=bool(spec.get("tokenizer_remote_code")))
    iface = Interface(system, spec["style"], tok, template=template(system, tid))
    rec["template"] = tid
    rt = check(test, iface)
    rec["roundtrip"] = {"ok": rt.ok, "reasons": rt.reasons}
    if not rt.ok:
        return rec
    inputs = [("S", "A", iface.inputs(test, "A").text), ("S", "B", iface.inputs(test, "B").text),
              ("R", "-", iface.mode_r_inputs(stream or (), delta_r, actions_known=stream is not None).text)]
    gen = make(system)
    t = time.time()
    gen.load()
    gen._loaded = True
    rec["load_seconds"] = round(time.time() - t, 1)
    if hasattr(gen, "att_layers"):          # PRISM: attention layers under prompt-to-prompt control (0 = degraded)
        rec["att_layers"] = gen.att_layers
    for mode, arm, text in inputs:
        o = gen.generate(img, text, SEED)
        r = {"mode": mode, "arm": arm, "input": text, "seconds": round(o.seconds, 2), "peak_gpu_gb": round(o.peak_gpu_gb, 2)}
        if mode == "S":                     # the arm's expected ETT state g (arm A: no insertion; arm B: insertion)
            r["expected_ETT"] = test.expected_a if arm == "A" else test.expected_b
        if o.kind == "image":
            f = out / f"{system}{'' if tid == 't1' else '_' + tid}_{mode}{arm if arm != '-' else ''}.png"
            o.image.save(f)
            r.update(file=str(f), native_size=o.meta.get("native_size"), prior_corr=_prior_corr(img, o.image))
        else:
            r.update(response=o.text, state=o.state, parsed=o.state is not None)
        if getattr(gen, "last_understanding", None) is not None:   # UniMedVL's own description, appended to the input
            r["understanding"] = gen.last_understanding
        rec["outputs"].append(r)
    rec["ok"] = True
    return rec


def main(argv=None) -> int:
    from spectra.generators import ADAPTERS

    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--prior-parquet", help="TAIX-Ray val shard (development data): its first film is the prior")
    src.add_argument("--mimic-config", help="config/mimic.yaml with the MIMIC server's paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--gpus", default=None, help="CUDA_VISIBLE_DEVICES for every system process")
    ap.add_argument("--envs", default="/path/to/home/myconda/envs", help="root of the per-system envs")
    ap.add_argument("--template", default="t1", help="prompt template id (Gate 2 chooses among t1-t3 on dev data)")
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.one:
        rec = {"system": args.one, "ok": False}
        try:
            rec = run_one(args.one, out, args.template)
        except Exception as e:  # recorded, then the driver moves on
            import traceback

            rec.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
        tag = args.one if args.template == "t1" else f"{args.one}_{args.template}"
        (out / f"{tag}.json").write_text(json.dumps(rec, indent=1, default=str))
        return 0 if rec.get("ok") else 1
    if not (args.prior_parquet or args.mimic_config):
        ap.error("one of --prior-parquet or --mimic-config is required")
    prepare(out, args.prior_parquet, args.mimic_config)
    env = _no_mps_env()
    if args.gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    from spectra.config import systems

    summary = {}
    for system in args.systems or list(ADAPTERS):
        # each system in its own env (systems.yaml `env`), as on the MIMIC server
        py = Path(args.envs) / systems()[system].get("env", "spectra-gpu") / "bin" / "python"
        tag = system if args.template == "t1" else f"{system}_{args.template}"
        with open(out / f"{tag}.log", "w") as log:
            rc = subprocess.run([str(py), __file__, "--one", system, "--out", str(out), "--template", args.template],
                                env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        summary[system] = "ok" if rc == 0 else "failed"
        print(system, summary[system], flush=True)
    # aggregate per system (the only part of a MIMIC-server run that may leave it)
    agg = {}
    for system, status in summary.items():
        tag = system if args.template == "t1" else f"{system}_{args.template}"
        rec = json.loads((out / f"{tag}.json").read_text()) if (out / f"{tag}.json").exists() else {}
        agg[system] = {"status": status, "roundtrip_ok": rec.get("roundtrip", {}).get("ok"),
                       "load_seconds": rec.get("load_seconds"), "gpus": rec.get("gpus"),
                       "seconds": [o["seconds"] for o in rec.get("outputs", [])],
                       "peak_gpu_gb": max([o["peak_gpu_gb"] for o in rec.get("outputs", [])], default=None),
                       "prior_corr": [o.get("prior_corr") for o in rec.get("outputs", []) if "prior_corr" in o] or None,
                       "error": re.sub(r"\d{5,}", "#", rec.get("error") or "")[:200] or None}  # no MIMIC ids
    (out / ("summary.json" if args.template == "t1" else f"summary_{args.template}.json")).write_text(json.dumps(agg, indent=1))
    return 0 if all(v == "ok" for v in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
