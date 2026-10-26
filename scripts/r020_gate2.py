"""R020 / R021 on this cluster (Gate 2, TAIX-Ray development data; rules in constants.yaml `gate2`).

    prepare   dev queries from the TAIX-Ray val E1 sets (effusion channels), the 8-bit prior films, and the readouts of
              the real follow-up films (frozen heads; reference performance on the same queries)
    generate  one configuration (system, template) on one shard of the queries, in the system's own env: every query
              in mode R (Track Delta, actions unknown) x every seed; outputs, seconds and peak GPU memory per output
    score     readouts of every output (frozen finding heads for images; parsed grades for state predictors), E1
              concordance per configuration (mean ± SD over patients), copy-prior (must be exactly 1/2), throughput
              against the plan's envelope, and the template choice

    python scripts/r020_gate2.py prepare --out ../runs/R020/taix_dev
    python scripts/r020_gate2.py generate --out ../runs/R020/taix_dev --system flux2_klein_4b --template t1 [--shard 0/2]
    python scripts/r020_gate2.py score --out ../runs/R020/taix_dev
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["effusion_L", "effusion_R"]                       # scored on TAIX-Ray (edema unscored after E-b / E-c)
FROZEN = ROOT.parent / "runs" / "R019" / "frozen" / "taix_findings_v1"
REAL_READOUTS = ROOT.parent / "runs" / "R019" / "heads" / "taix_findings_v1" / "val_readouts.parquet"
ENVELOPE_S = {"progemu": 120, "unimedvl": 40, "prism": 5, "qwen_image_edit_2511": 40, "flux2_klein_4b": 5,
              "medgemma_1_5_4b": 5, "lingshu_32b": 5}           # upper end of EXPERIMENT_PLAN.md (b), s / generation


def _safe(qid: str) -> str:
    return qid.replace("->", "__")


def prepare(out: Path) -> None:
    import hashlib

    import pandas as pd
    import pyarrow.parquet as pq
    from PIL import Image

    from spectra.adapters.taix import TaixAdapter
    from spectra.config import constants, source_config
    from spectra.e1 import query_id
    from spectra.images import to_uint8

    G = constants()["gate2"]
    cfg = source_config(ROOT / "config" / "taix.yaml")
    _, _, chosen = TaixAdapter(cfg["metadata"]).e1_sets(G["source_split"], channels=CHANNELS)
    order = sorted(chosen, key=lambda it: hashlib.sha256(f"{G['query_seed']}|{it[0].prior.study_id}".encode()).hexdigest())
    priors, contrasts, n = [], [], 0
    for qs, cons in order:
        if n >= G["dev_queries"]:
            break
        priors.append({"prior": qs.prior.study_id, "patient": qs.prior.patient_id,
                       "later": [{"uid": f.study_id, "query": query_id(qs.prior, f), "delta_h": f.t_h - qs.prior.t_h}
                                 for f in qs.later]})
        contrasts += [{"patient": qs.prior.patient_id, "query_j": c.query_j, "query_k": c.query_k,
                       "channel": c.channel, "rho": c.rho} for c in cons]
        n += len(qs.later)
    (out / "priors").mkdir(parents=True, exist_ok=True)
    need = {p["prior"] for p in priors}
    for f in sorted(Path(cfg["images"]).glob(f"{G['source_split']}-*.parquet")):
        if not need:
            break
        pf = pq.ParquetFile(f)
        for rg in range(pf.metadata.num_row_groups):
            uids = pf.read_row_group(rg, columns=["UID"]).column("UID").to_pylist()
            hit = need.intersection(uids)
            if not hit:
                continue
            t = pf.read_row_group(rg, columns=["UID", "Image"])
            for u, x in zip(t.column("UID").to_pylist(), t.column("Image").to_pylist()):
                if u in hit:
                    to_uint8(Image.open(io.BytesIO(x["bytes"]))).save(out / "priors" / f"{u}.png")
                    need.discard(u)
    if need:
        raise SystemExit(f"prior films not found: {sorted(need)[:3]}")
    real = pd.read_parquet(REAL_READOUTS).set_index("UID")
    q = {"source": "taix", "split": G["source_split"], "channels": CHANNELS, "seeds": G["seeds"],
         "priors": priors, "contrasts": contrasts,
         "real_readouts": {u: {c: float(real.loc[u, f"phi_{c}"]) for c in CHANNELS}
                           for p in priors for u in [p["prior"]] + [f["uid"] for f in p["later"]]}}
    (out / "queries.json").write_text(json.dumps(q, indent=1))
    print(f"{len(priors)} priors, {n} queries, {len(contrasts)} contrasts, "
          f"{len({p['patient'] for p in priors})} patients", flush=True)


def generate(out: Path, system: str, tid: str, shard: str) -> None:
    import torch
    from PIL import Image

    from spectra.config import constants, systems, template
    from spectra.generators import make
    from spectra.generators.base import NoImage
    from spectra.roundtrip import Interface
    from spectra.tokenizers import HFTokenizer

    q = json.loads((out / "queries.json").read_text())
    seeds = [constants()["generation"]["seed_base"] + s for s in range(q["seeds"])]
    k, n = (int(x) for x in shard.split("/"))
    jobs = [(p["prior"], f) for p in q["priors"] for f in p["later"]][k::n]
    spec = systems()[system]
    tok = None
    if spec.get("tokenizer"):
        tok = HFTokenizer(spec["tokenizer"], max_len=spec.get("max_tokens"), subfolder=spec.get("tokenizer_subfolder"),
                          trust_remote_code=bool(spec.get("tokenizer_remote_code")))
    iface = Interface(system, spec["style"], tok, template=template(system, tid))
    d = out / "gen" / f"{system}_{tid}"
    d.mkdir(parents=True, exist_ok=True)
    log = d / f"records.{k}of{n}.jsonl"
    done = set()
    if log.exists():
        done = {(r["query"], r["seed"]) for r in map(json.loads, log.read_text().splitlines())}
    gen = make(system)
    t = time.time()
    gen.load()
    gen._loaded = True
    load_s = round(time.time() - t, 1)
    warm = True                    # the first output after a model load carries the warm-up
    with open(log, "a") as fh:
        for prior, f in jobs:
            img = Image.open(out / "priors" / f"{prior}.png")
            text = iface.mode_r_inputs((), f["delta_h"], actions_known=False).text
            for s in seeds:
                if (f["query"], s) in done:
                    continue
                base = {"query": f["query"], "seed": s, "system": system, "template": tid, "load_seconds": load_s,
                        "gpu": torch.cuda.get_device_name(0), "input": text, "warm": warm}
                warm = False
                t0 = time.time()
                try:
                    o = gen.generate(img, text, s)
                except NoImage as e:   # a completed generation without an image: failure rule, kappa = 0
                    fh.write(json.dumps(dict(base, kind="failed", reason=str(e), seconds=round(time.time() - t0, 3),
                                             peak_gpu_gb=None)) + "\n")
                    fh.flush()
                    continue
                rec = dict(base, seconds=round(o.seconds, 3), peak_gpu_gb=round(o.peak_gpu_gb, 2))
                if o.kind == "image":
                    p = d / f"{_safe(f['query'])}_s{s}.png"
                    o.image.save(p)
                    rec.update(kind="image", file=p.name, native_size=o.meta.get("native_size"))
                else:
                    rec.update(kind="state", response=o.text, state=o.state)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
    print(f"{system} {tid} shard {shard}: {len(jobs)} queries x {len(seeds)} seeds done", flush=True)


def _image_readouts(paths, device="cuda", bs=64):
    """phi_c (expected grade) of image files through the evaluation rule and the frozen finding heads."""
    import numpy as np
    import torch
    from PIL import Image

    from spectra.config import constants
    from spectra.evaluator.encoder import RadDino, eval_u8, pixel_values_u8, processor
    from spectra.evaluator.heads import Heads, readouts

    ck = torch.load(FROZEN / "heads.pt", map_location="cpu", weights_only=False)
    heads = Heads(ck["tasks"], dim=ck["dim"]).to(device).eval()
    heads.load_state_dict(ck["state_dict"])
    enc, proc = RadDino(device=device), processor()
    size = constants()["generation"]["evaluation_resolution"]
    out, ok = {}, []
    for p in paths:                # an image file that cannot be decoded is unreadable (None)
        try:
            with Image.open(p) as im:
                im.load()
            ok.append(p)
        except (OSError, ValueError):
            out[p] = None
    paths = ok
    for i in range(0, len(paths), bs):
        chunk = paths[i:i + bs]
        u8 = torch.from_numpy(np.stack([eval_u8(Image.open(p), size) for p in chunk]))
        with torch.no_grad():
            ro = readouts(heads(enc.tokens(pixel_values_u8(u8, proc)).float()))
        for j, p in enumerate(chunk):
            out[p] = {c: float(ro[c][j]) for c in CHANNELS}
    return out


def score(out: Path) -> None:
    import numpy as np

    from spectra.config import constants
    from spectra.e1 import kappa
    from spectra.evaluator.metrics import patient_mean_sd

    q = json.loads((out / "queries.json").read_text())
    seeds = [constants()["generation"]["seed_base"] + s for s in range(q["seeds"])]
    prior_of = {f["query"]: p["prior"] for p in q["priors"] for f in p["later"]}
    target_of = {f["query"]: f["uid"] for p in q["priors"] for f in p["later"]}
    queries = list(prior_of)
    contrasts = q["contrasts"]
    pooled = {"effusion": CHANNELS}

    def concord(readout):          # readout(query, seed) -> {channel: phi} or None (unreadable)
        per = {}
        for c in contrasts:
            ks = []
            for s in seeds:
                a, b = readout(c["query_j"], s), readout(c["query_k"], s)
                ks.append(0.0 if a is None or b is None else kappa(a[c["channel"]], b[c["channel"]], c["rho"]))
            for ch in (c["channel"], "effusion"):
                per.setdefault(ch, {}).setdefault(c["patient"], []).append(float(np.mean(ks)))
        res = {}
        for ch, pats in per.items():
            r = patient_mean_sd(np.mean(x) for x in pats.values())
            res[ch] = {"C": r["mean"], "sd": r["sd"], "N": r["N"]}
        return res

    real = q["real_readouts"]
    summary = {"queries": len(queries), "priors": len(q["priors"]), "contrasts": len(contrasts),
               "patients": len({c["patient"] for c in contrasts}), "seeds": seeds,
               "controls": {"copy_prior": concord(lambda qq, s: real[prior_of[qq]]),
                            "real_follow_up": concord(lambda qq, s: real[target_of[qq]])},
               "configurations": {}}
    cp = summary["controls"]["copy_prior"]
    summary["copy_prior_exact_half"] = all(r["C"] == 0.5 for r in cp.values())
    if not summary["copy_prior_exact_half"]:   # R021: exact controls hold, else stop
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        raise SystemExit(f"copy-prior C != 1/2: { {k: v['C'] for k, v in cp.items()} }")
    recs = {}
    for d in sorted((out / "gen").glob("*_t*")):
        rows = []
        for f in sorted(d.glob("records.*.jsonl")):         # the first output of each shard carries the warm-up
            for i, line in enumerate(f.read_text().splitlines()):
                r = json.loads(line)
                r.setdefault("warm", i == 0)   # records written before the warm flag existed: first line of the shard
                rows.append(r)
        recs[d.name] = (d, rows)
    files = [d / r["file"] for d, rows in recs.values() for r in rows if r.get("kind") == "image"]
    phi = _image_readouts([str(p) for p in files]) if files else {}
    for name, (d, rows) in recs.items():
        by = {(r["query"], r["seed"]): r for r in rows}
        missing = [(qq, s) for qq in queries for s in seeds if (qq, s) not in by]

        def readout(qq, s, by=by, d=d):
            r = by.get((qq, s))
            if r is None:
                return None
            if r["kind"] == "failed":
                return None
            if r["kind"] == "image":
                return phi[str(d / r["file"])]
            st = r.get("state")
            return None if st is None else {c: float(st["grades"][c]) for c in CHANNELS}

        system, tid = name.rsplit("_", 1)
        steady = [r["seconds"] for r in rows if not r["warm"]]   # failed generations cost time too
        med = float(np.median(steady)) if steady else None   # no budget decision without a steady-state timing
        env = ENVELOPE_S.get(system)
        summary["configurations"][name] = {
            "system": system, "template": tid, "outputs": len(rows), "missing": len(missing),
            "unreadable": sum(1 for r in rows if r["kind"] == "failed" or (r["kind"] == "state" and r.get("state") is None)
                              or (r["kind"] == "image" and phi.get(str(d / r["file"])) is None)),
            "E1": concord(readout) if not missing else None,
            "median_s_per_generation": med, "envelope_s": env,
            "reduced_subset": None if med is None or env is None else bool(med > 2 * env),
            "peak_gpu_gb": max((r["peak_gpu_gb"] for r in rows if r["peak_gpu_gb"] is not None), default=None),
            "load_seconds": max((r["load_seconds"] for r in rows), default=None)}
    from spectra.config import load_yaml

    declared = {s: sorted(t) for s, t in load_yaml(ROOT / "src" / "spectra" / "resources" / "templates.yaml").items()
                if isinstance(t, dict) and any(k.startswith("t") and k[1:].isdigit() for k in t)}
    choice = {}
    for system in sorted({r["system"] for r in summary["configurations"].values()}):
        cands = [k for k in declared.get(system, ["t1"]) if k.startswith("t") and k[1:].isdigit()]
        res = [summary["configurations"].get(f"{system}_{t}") for t in cands]
        if any(r is None or r["E1"] is None for r in res):   # every declared template must be complete
            choice[system] = "pending"
            continue
        best = max(res, key=lambda r: (r["E1"]["effusion"]["C"], -int(r["template"][1:])))
        choice[system] = best["template"]
    summary["template_choice"] = choice
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"copy_prior_exact_half": summary["copy_prior_exact_half"],
                      "real_follow_up": summary["controls"]["real_follow_up"]["effusion"],
                      **{k: (v["E1"] or {}).get("effusion") for k, v in summary["configurations"].items()},
                      "template_choice": summary["template_choice"]}, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "generate", "score"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--system")
    ap.add_argument("--template", default="t1")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.step == "prepare":
        prepare(out)
    elif args.step == "generate":
        generate(out, args.system, args.template, args.shard)
    else:
        score(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
