"""`spectra <step> --config config/<source>.yaml` — one entry point on both machines.

Shard-aware steps accept `--shard i --n-shards n --out DIR`, so `spectra.runner.run_local` (MIMIC server,
no scheduler) and sbatch arrays (this cluster) call them identically."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def shard_of(key: str, n_shards: int) -> int:
    return int(hashlib.sha256(str(key).encode()).hexdigest(), 16) % n_shards


def _selftest(args) -> int:
    import pytest

    here = Path(__file__).resolve().parents[2] / "tests"
    if not here.exists():  # installed package: tests ship with the repository, not the wheel
        print(f"tests not found next to the package ({here}); run pytest from the repository", file=sys.stderr)
        return 2
    return int(pytest.main(["-q", str(here)] + (["-k", args.k] if args.k else [])))


def _cer(args) -> int:
    from .cer import CERCompiler, IcuTables, MimicOntology
    from .serialize import render_event

    ont = MimicOntology()
    tables = IcuTables.from_dir(args.icu_dir, stay_ids=[args.stay], ontology=ont, track_b=args.track == "B")
    stream = CERCompiler(ont).compile(tables, args.stay, args.t0, args.tk, track=args.track, cutoff=args.cutoff)
    for c in stream:
        print(render_event(c))
    return 0


def _build_e1(args) -> int:
    """E1 query sets and admitted contrasts for one shard of patients (TAIX-Ray at Gate 0; other sources
    plug in their adapters at Gate 1)."""
    import pandas as pd

    from .adapters.taix import TaixAdapter
    from .config import source_config

    cfg = source_config(args.config)
    if cfg["source"] != "taix":
        raise NotImplementedError(f"build-e1 for source {cfg['source']!r} lands with its adapter at Gate 1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a = TaixAdapter(cfg["metadata"])
    rows, n_pat = [], 0
    for split in cfg.get("splits", ["test"]):
        _, _, chosen = a.e1_sets(split)
        mine = [(qs, cons) for qs, cons in chosen if shard_of(qs.prior.patient_id, args.n_shards) == args.shard]
        n_pat += len({qs.prior.patient_id for qs, _ in mine})
        for qs, cons in mine:
            for c in cons:
                rows.append({"split": split, "patient_id": qs.prior.patient_id, "prior_study": c.prior_study,
                             "query_j": c.query_j, "query_k": c.query_k, "channel": c.channel, "rho": c.rho})
    pd.DataFrame(rows).to_parquet(out / f"e1_contrasts.shard-{args.shard:04d}-of-{args.n_shards:04d}.parquet", index=False)
    (out / f"e1_funnel.shard-{args.shard:04d}-of-{args.n_shards:04d}.json").write_text(
        json.dumps({"patients": n_pat, "contrasts": len(rows)}))
    return 0


def _noop(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"noop.{args.shard}.txt").write_text(f"{args.shard}/{args.n_shards}")
    return 0


def _taix_funnel(args) -> int:
    from .adapters.taix import TaixAdapter

    a = TaixAdapter(args.metadata)
    res = {s: a.funnel(s) for s in args.splits}
    text = json.dumps(res, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


def _export(args) -> int:
    import pandas as pd

    from .export_aggregate import ExportRefused, export

    df = pd.read_csv(args.table)
    try:
        p = export(df, args.out, args.analyst, args.purpose, analyst_confirmed=args.confirm)
    except ExportRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"exported {p}")
    return 0


def _extract_reports(args) -> int:
    """Report extraction (M5 / X4 / D3) for one shard: resumable JSONL of validated extractions or rejections."""
    import time
    from collections import Counter

    from .references import extract as ex

    if args.source == "mimic":
        items = ex.mimic_items(args.config, icu_only=not args.all_films)
    elif args.source == "openi":
        items = ex.openi_items(args.config)
    else:
        items = ex.jsonl_items(args.config)
    items = sorted(it for it in items if shard_of(it[0], args.n_shards) == args.shard)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"extractions.{args.shard:04d}-of-{args.n_shards:04d}.jsonl"
    from .references.schema import PROMPT_VERSION

    run = {"source": args.source, "model": args.model, "prompt": PROMPT_VERSION}
    run_file = out / "run.json"
    if run_file.exists():
        prev = json.loads(run_file.read_text())
        if prev != run:
            raise SystemExit(f"{out} holds extractions of {prev}, not {run}: use a fresh --out")
    elif any(out.glob("extractions.*.jsonl")):
        raise SystemExit(f"{out} holds extractions without run.json (unknown prompt): use a fresh --out")
    else:
        run_file.write_text(json.dumps(run, indent=1))
    done = set()
    if path.exists():
        done = {json.loads(l)["report_id"] for l in path.read_text().splitlines() if l.strip()}
    todo = [it for it in items if it[0] not in done][: args.limit]
    print(f"shard {args.shard}/{args.n_shards}: {len(items)} reports, {len(done)} done, {len(todo)} to run", flush=True)
    t0, n_new = time.time(), 0
    if todo:
        model = ex.Extractor(args.model, tensor_parallel=args.tp, max_model_len=args.max_model_len,
                             gpu_memory_utilization=args.gpu_mem)
        t0 = time.time()
        with open(path, "a") as fh:
            for s in range(0, len(todo), args.chunk):
                for rec in model.extract(todo[s:s + args.chunk]):
                    fh.write(json.dumps(rec) + "\n")
                    n_new += 1
                fh.flush()
                print(f"{s + len(todo[s:s + args.chunk])}/{len(todo)} reports, {n_new / (time.time() - t0):.2f}/s", flush=True)
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    why = Counter((r["rejected"] or "accepted").split(":")[0] for r in recs)
    summary = {"source": args.source, "model": args.model, "prompt": PROMPT_VERSION, "shard": args.shard, "n_shards": args.n_shards,
               "reports": len(recs), "outcomes": dict(why), "new_reports": n_new,
               "seconds_new": round(time.time() - t0, 1),
               "mean_output_tokens": (sum(r["output_tokens"] for r in recs) / len(recs)) if recs else None}
    (out / f"summary.{args.shard:04d}-of-{args.n_shards:04d}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    return 0


def _run(args) -> int:
    """Shard one step over local GPUs (the MIMIC server has no scheduler): resumable, a failed shard is reported."""
    from .runner import run_local

    gpus = args.gpus.split(",") if args.gpus else None
    r = run_local(args.step_argv, args.out, args.n_shards, gpus=gpus, gpus_per_shard=args.gpus_per_shard)
    print(json.dumps(r), flush=True)
    return 1 if r["failed"] else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spectra")
    sub = ap.add_subparsers(dest="step", required=True)
    shard = argparse.ArgumentParser(add_help=False)
    shard.add_argument("--shard", type=int, default=0)
    shard.add_argument("--n-shards", type=int, default=1)
    shard.add_argument("--out", default=".")

    s = sub.add_parser("selftest", help="run the Gate-0 unit tests (exact controls, CER, round-trip, ...)")
    s.add_argument("-k", default=None)
    s.set_defaults(func=_selftest)
    s = sub.add_parser("cer", help="compile one MIMIC-IV window into canonical event records")
    s.add_argument("--icu-dir", required=True)
    s.add_argument("--stay", type=int, required=True)
    s.add_argument("--t0", required=True)
    s.add_argument("--tk", required=True)
    s.add_argument("--track", choices=["A", "B"], default="A")
    s.add_argument("--cutoff", choices=["occurrence", "availability"], default="occurrence")
    s.set_defaults(func=_cer)
    s = sub.add_parser("build-e1", parents=[shard], help="E1 query sets and admitted contrasts for one patient shard")
    s.add_argument("--config", required=True)
    s.set_defaults(func=_build_e1)
    s = sub.add_parser("noop", parents=[shard], help="shard-aware no-op (runner smoke test)")
    s.set_defaults(func=_noop)
    s = sub.add_parser("taix-funnel", help="E1 funnel on TAIX-Ray metadata with the authoritative adapter")
    s.add_argument("--metadata", required=True)
    s.add_argument("--splits", nargs="+", default=["test"])
    s.add_argument("--out", default=None)
    s.set_defaults(func=_taix_funnel)
    s = sub.add_parser("extract-reports", parents=[shard], help="report extraction (Qwen3-32B, vLLM) for one shard")
    s.add_argument("--source", choices=["mimic", "openi", "jsonl"], required=True)
    s.add_argument("--config", required=True, help="mimic: config/mimic.yaml; openi: dataset snapshot; jsonl: file")
    s.add_argument("--all-films", action="store_true", help="mimic: every F1 film, not only ICU films")
    s.add_argument("--model", default="Qwen/Qwen3-32B")
    s.add_argument("--tp", type=int, default=1)
    s.add_argument("--max-model-len", type=int, default=4096)
    s.add_argument("--gpu-mem", type=float, default=0.92)
    s.add_argument("--chunk", type=int, default=512, help="reports per generate call (and per flush)")
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=_extract_reports)
    s = sub.add_parser("run", help="shard a step over local GPUs: spectra run --out DIR --n-shards N STEP ARGS...")
    s.add_argument("--out", required=True)
    s.add_argument("--n-shards", type=int, required=True)
    s.add_argument("--gpus", default=None, help="comma-separated GPU ids (default: all visible)")
    s.add_argument("--gpus-per-shard", type=int, default=1)
    s.add_argument("step_argv", nargs=argparse.REMAINDER)
    s.set_defaults(func=_run)
    s = sub.add_parser("export", help="the only outbound path from the MIMIC server: checked aggregate tables")
    s.add_argument("--table", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--analyst", required=True)
    s.add_argument("--purpose", required=True)
    s.add_argument("--confirm", action="store_true", help="the credentialed analyst inspected row granularity and content")
    s.set_defaults(func=_export)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
