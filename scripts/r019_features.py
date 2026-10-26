"""R019 (E-a input): frozen RAD-DINO tokens of every film of a source split, read through the evaluation rule
(spectra.images.to_eval at 512^2, then RAD-DINO's processor). One run per source (TAIX-Ray: train / val / test;
RANZCR CLiP: dev / val, its salted patient split, constants.yaml evaluator); outputs per split:

    <out>/<split>.tokens.f16.npy   [N, 1 + G*G, 768] float16 (CLS, pooled grid)
    <out>/<split>.uids.txt         film id of each row
    <out>/<split>.meta.json        counts, timing, encoder revision, rule

Test-split films are read only after the evaluator is frozen (E-d): --split test needs --frozen <manifest>.

    python scripts/r019_features.py --config config/taix.yaml --out ../runs/R019/features/taix --splits train val
    python scripts/r019_features.py --config config/ranzcr.yaml --out ../runs/R019/features/ranzcr --splits dev val
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402


from spectra.evaluator.data import FileFilms, TaixFilms, ranzcr_table, taix_tasks  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--frozen", default=None, help="frozen evaluator manifest (required for the test split)")
    ap.add_argument("--limit", type=int, default=None, help="smoke: about this many films per split")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--via", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu-threads", type=int, default=8)
    args = ap.parse_args(argv)

    import numpy as np
    import transformers

    from spectra.config import constants, source_config
    from spectra.evaluator.encoder import DIM, RadDino, pixel_values_u8, processor

    cfg = source_config(args.config)
    if cfg["source"] not in ("taix", "ranzcr"):
        raise SystemExit("TAIX-Ray and RANZCR CLiP layouts are implemented; PadChest joins at its access")
    if cfg["source"] == "ranzcr":
        ev = constants()["evaluator"]
        table = ranzcr_table(cfg["labels"], cfg["images"], ev["ranzcr_val_fraction"], ev["ranzcr_split_salt"])
    if "test" in args.splits and not (args.frozen and Path(args.frozen).is_file()):
        raise SystemExit("test-split films are read only after the freeze (E-d): pass --frozen <manifest>")
    size = constants()["generation"]["evaluation_resolution"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    enc = RadDino(device=args.device)
    proc = processor()
    torch.set_num_threads(args.cpu_threads)      # the processor runs here, on uint8 batches from the workers
    t_load = time.time()
    for split in args.splits:
        if cfg["source"] == "taix":
            tasks, n = taix_tasks(cfg["images"], split, args.limit)
            films = TaixFilms(tasks, size, args.via)
        else:
            rows = table[table["split"] == split].head(args.limit) if args.limit else table[table["split"] == split]
            n = len(rows)
            films = FileFilms(zip(rows["uid"], rows["path"]), size, args.via)
        T = 1 + enc.grid * enc.grid
        # rows are buffered in RAM and written once: page-faulting a large memmap on NFS (network filesystem) stalled the
        # job in I/O wait with the GPU idle (Slurm job 4522374)
        arr = np.empty((n, T, DIM), dtype=np.float16)
        loader = torch.utils.data.DataLoader(films, batch_size=args.batch,
                                             num_workers=args.workers, pin_memory=False, prefetch_factor=2,
                                             persistent_workers=False)
        uids, i, t0, t_print = [], 0, time.time(), time.time()
        for bu, bx in loader:
            tok = enc.tokens(pixel_values_u8(bx, proc)).to(torch.float16).cpu().numpy()
            arr[i:i + len(bu)] = tok
            uids += list(bu)
            i += len(bu)
            if time.time() - t_print > 60:
                t_print = time.time()
                print(f"{split}: {i}/{n} films, {i / (time.time() - t0):.0f}/s", flush=True)
        np.save(out / f"{split}.tokens.f16.npy", arr)
        if i != n or len(set(uids)) != n:
            raise SystemExit(f"{split}: wrote {i} rows, {len(set(uids))} unique ids, expected {n}")
        (out / f"{split}.uids.txt").write_text("\n".join(uids) + "\n")
        dt = time.time() - t0
        snap = next(Path(os.environ.get("HF_HOME", "")).glob("hub/models--microsoft--rad-dino/snapshots/*"), None)
        meta = {"source": cfg["source"], "split": split, "n": n, "tokens": T, "dim": DIM, "grid": enc.grid,
                "rule": f"to_eval({size}, via={args.via}) + RAD-DINO processor (CPU)", "encoder": enc.repo,
                "encoder_revision": snap.name if snap else None, "dtype_on_disk": "float16", "seconds": round(dt, 1),
                "films_per_s": round(n / dt, 1), "batch": args.batch, "workers": args.workers,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "host": platform.node(), "torch": torch.__version__, "transformers": transformers.__version__,
                "limit": args.limit}
        (out / f"{split}.meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta), flush=True)
    print(f"done in {time.time() - t_load:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
