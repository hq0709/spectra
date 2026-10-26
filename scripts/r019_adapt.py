"""R019, step E-c: the one prespecified adaptation of a finding head that missed its section-12 criterion on a source
(constants.yaml `evaluator.adaptation`, fixed before this ran). TAIX-Ray edema missed it at E-b (QWK 0.448 < 0.60).

- films: <cache>/<split>.u8.npy, [N, 512, 512] uint8 evaluation films (spectra.images.to_eval) with <split>.uids.txt,
  built once from the TAIX-Ray parquet shards and written sequentially; RAD-DINO's processor runs on the CPU per batch.
- model: RAD-DINO with its last `trainable_blocks` blocks and final layer norm trainable, and the channel's E-a head
  (initialised from heads.pt) on the same tokens (CLS + 8 x 8 pooled grid).
- train: TAIX-Ray train minus the E-a holdout patients; the epoch with the lowest holdout cross-entropy is kept.
- re-validation, once: TAIX-Ray val; the QWK point estimate decides. Test-split films are never read.

    python scripts/r019_adapt.py --heads ../runs/R019/heads/taix_findings_v1 --channel edema \
        --cache ../runs/R019/films/taix --out ../runs/R019/adapt/taix_edema_v1
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

COLUMNS = {"edema": "PulmonaryCongestion", "effusion_L": "PleuralEffusion_Left", "effusion_R": "PleuralEffusion_Right"}


def build_cache(images_dir: str, cache: Path, split: str, size: int, workers: int) -> None:
    from spectra.evaluator.data import TaixFilms, taix_tasks

    if (cache / f"{split}.u8.npy").exists() and (cache / f"{split}.uids.txt").exists():
        return
    tasks, n = taix_tasks(images_dir, split)
    arr = np.empty((n, size, size), dtype=np.uint8)
    loader = torch.utils.data.DataLoader(TaixFilms(tasks, size), batch_size=256, num_workers=workers,
                                         prefetch_factor=2)
    uids, i, t0 = [], 0, time.time()
    for bu, bx in loader:
        arr[i:i + len(bu)] = bx.numpy()
        uids += list(bu)
        i += len(bu)
    if i != n or len(set(uids)) != n:
        raise SystemExit(f"{split}: cached {i} films, {len(set(uids))} unique, expected {n}")
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / f"{split}.u8.npy", arr)
    (cache / f"{split}.uids.txt").write_text("\n".join(uids) + "\n")
    print(f"cached {split}: {n} films in {time.time() - t0:.0f}s", flush=True)


def tokens(enc, px, grid: int):
    h = enc(pixel_values=px).last_hidden_state
    cls, patches = h[:, :1], h[:, 1:]
    n = int(round(patches.shape[1] ** 0.5))
    g = patches.reshape(-1, n, n, h.shape[-1]).permute(0, 3, 1, 2)
    g = F.adaptive_avg_pool2d(g.float(), grid).flatten(2).transpose(1, 2)
    return torch.cat([cls.float(), g], 1)


def batches(films, idx, bs, proc, depth=4):
    """Background thread: uint8 films -> RAD-DINO processor (CPU) -> float batches, in the order of idx."""
    from spectra.evaluator.encoder import pixel_values_u8

    q: "queue.Queue" = queue.Queue(maxsize=depth)

    def work():
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            q.put((b, pixel_values_u8(torch.from_numpy(films[np.sort(b)]), proc), np.sort(b)))
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not None:
        yield item


def predict(enc, head, films, idx, proc, grid, dev, bs=256):
    enc.eval(), head.eval()
    out = np.empty(len(idx), dtype=np.float32)
    logits = np.empty((len(idx), head.mlp[-1].out_features), dtype=np.float32)
    pos = {int(v): k for k, v in enumerate(idx)}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _, px, order in batches(films, idx, bs, proc):
            lg = head(tokens(enc, px.to(dev, non_blocking=True), grid)).float()
            rows = [pos[int(v)] for v in order]
            logits[rows] = lg.cpu().numpy()
    p = torch.from_numpy(logits).softmax(-1).numpy()
    out[:] = (p * np.arange(p.shape[1])).sum(1)
    return out, logits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True, help="E-a run directory (heads.pt, val_readouts.parquet)")
    ap.add_argument("--channel", required=True, choices=list(COLUMNS))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "taix.yaml"))
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", type=int, default=None, help="pipeline check: N films per set, one epoch")
    ap.add_argument("--revalidate-only", action="store_true", help="skip training; re-validate <out>/adapted.pt")
    args = ap.parse_args(argv)

    import pandas as pd
    from transformers import AutoModel

    from spectra.config import constants, source_config
    from spectra.evaluator.encoder import processor
    from spectra.evaluator.heads import AttentiveHead
    from spectra.evaluator.metrics import ordinal_report

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from r019_heads import holdout_mask

    C = constants()
    ev, acc, A = C["evaluator"], C["evaluator_acceptance"], C["evaluator"]["adaptation"]
    cfg = source_config(args.config)
    size, grid, ch = C["generation"]["evaluation_resolution"], ev["token_grid"], args.channel
    n_cls = ev["finding_grades"][ch]
    out, cache, dev = Path(args.out), Path(args.cache), args.device
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    t_start = time.time()

    for split in ("train", "val"):
        build_cache(cfg["images"], cache, split, size, args.workers)
    meta = pd.read_parquet(cfg["metadata"]).set_index("UID")
    films = {s: np.load(cache / f"{s}.u8.npy") for s in ("train", "val")}
    uids = {s: (cache / f"{s}.uids.txt").read_text().split() for s in ("train", "val")}
    m_tr = meta.loc[uids["train"]]
    if (m_tr["Split"] != "train").any() or (meta.loc[uids["val"], "Split"] != "val").any():
        raise SystemExit("cached films from another split")
    y_tr = m_tr[COLUMNS[ch]].to_numpy()
    hold = holdout_mask(m_tr["PatientID"].to_numpy(), ev["holdout_salt"], ev["holdout_fraction"])
    tr_idx = np.flatnonzero(~hold & (y_tr >= 0))
    ho_idx = np.flatnonzero(hold & (y_tr >= 0))
    epochs = A["epochs"]
    if args.smoke:
        tr_idx, ho_idx, epochs = tr_idx[:args.smoke], ho_idx[:args.smoke], 1

    proc = processor()
    enc = AutoModel.from_pretrained(ev["encoder"], dtype=torch.float32).to(dev)
    for p in enc.parameters():
        p.requires_grad = False
    k = A["trainable_blocks"]
    train_params = [p for blk in enc.encoder.layer[-k:] for p in blk.parameters()] + list(enc.layernorm.parameters())
    for p in train_params:
        p.requires_grad = True
    ea = torch.load(Path(args.heads) / "heads.pt", map_location="cpu", weights_only=False)
    head = AttentiveHead(ea["dim"], n_cls)
    pre = f"heads.{ch}."
    head.load_state_dict({kk[len(pre):]: v for kk, v in ea["state_dict"].items() if kk.startswith(pre)})
    head.to(dev)
    opt = torch.optim.AdamW([{"params": train_params, "lr": A["lr_encoder"]},
                             {"params": head.parameters(), "lr": A["lr_head"]}], weight_decay=A["weight_decay"])
    steps_per_epoch = math.ceil(len(tr_idx) / A["batch"])
    total, warm = steps_per_epoch * epochs, max(1, int(A["warmup_fraction"] * steps_per_epoch * epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    y_t = torch.as_tensor(y_tr, dtype=torch.long)
    rng = np.random.default_rng(args.seed)

    def holdout_ce():
        phi, lg = predict(enc, head, films["train"], ho_idx, proc, grid, dev)
        return float(F.cross_entropy(torch.from_numpy(lg), y_t[ho_idx]).item())

    log, best, best_state, best_epoch = [], float("nan"), None, -1
    if args.revalidate_only:
        saved = torch.load(out / "adapted.pt", map_location="cpu", weights_only=False)
        best_state, best_epoch = {"encoder_trainable": saved["encoder_trainable"], "head": saved["head"]}, saved["best_epoch"]
        best = saved.get("holdout_ce", float("nan"))
        log = json.loads((out / "train_log.json").read_text()) if (out / "train_log.json").exists() else []
    else:
        best = holdout_ce()
        log.append({"epoch": -1, "holdout_ce": best, "note": "E-a head on the frozen encoder"})
        print(json.dumps(log[-1]), flush=True)
        (out / "train_log.json").write_text(json.dumps(log, indent=1))
    for ep in ([] if args.revalidate_only else range(epochs)):
        enc.train(), head.train()
        t0, run, n_seen = time.time(), 0.0, 0
        for b, px, order in batches(films["train"], rng.permutation(tr_idx), A["batch"], proc):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = head(tokens(enc, px.to(dev, non_blocking=True), grid))
                loss = F.cross_entropy(lg.float(), y_t[order].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params + list(head.parameters()), 1.0)
            opt.step()
            sched.step()
            run += float(loss.detach()) * len(order)
            n_seen += len(order)
            if (n_seen // A["batch"]) % 200 == 0:
                print(f"epoch {ep}: {n_seen}/{len(tr_idx)} films, {n_seen / (time.time() - t0):.0f}/s, "
                      f"train ce {run / n_seen:.4f}", flush=True)
        ce = holdout_ce()
        log.append({"epoch": ep, "train_ce": run / n_seen, "holdout_ce": ce, "s": round(time.time() - t0, 1)})
        print(json.dumps(log[-1]), flush=True)
        (out / "train_log.json").write_text(json.dumps(log, indent=1))
        if ce < best - 1e-5:
            best, best_epoch = ce, ep
            best_state = {"encoder_trainable": {kk: v.detach().cpu().clone() for kk, v in enc.state_dict().items()
                                                if kk.startswith("layernorm") or any(
                                                    kk.startswith(f"encoder.layer.{j}.") for j in
                                                    range(len(enc.encoder.layer) - k, len(enc.encoder.layer)))},
                          "head": {kk: v.detach().cpu().clone() for kk, v in head.state_dict().items()}}
            torch.save({**best_state, "channel": ch, "best_epoch": ep, "holdout_ce": ce, "adaptation": A,
                        "args": vars(args)}, out / "adapted.pt")
    if best_state is None:
        verdict_note = "no epoch improved the holdout cross-entropy of the E-a head; the E-a head is kept"
        head.load_state_dict({kk[len(pre):]: v for kk, v in ea["state_dict"].items() if kk.startswith(pre)})
        enc = AutoModel.from_pretrained(ev["encoder"], dtype=torch.float32).to(dev)
    else:
        verdict_note = f"epoch {best_epoch} kept"
        enc.load_state_dict(best_state["encoder_trainable"], strict=False)
        head.load_state_dict(best_state["head"])

    # re-validation, once, on the held-out real films of TAIX-Ray val
    va_idx = np.arange(len(uids["val"]))[: args.smoke] if args.smoke else np.arange(len(uids["val"]))
    m_va = meta.loc[[uids["val"][i] for i in va_idx]]
    phi, _ = predict(enc, head, films["val"], va_idx, proc, grid, dev)
    g = m_va[COLUMNS[ch]].to_numpy()
    ok = g >= 0
    rep = ordinal_report(phi[ok], g[ok], m_va["PatientID"].to_numpy()[ok], n_cls, acc["ordinal_weighted_kappa_vs_taix"])
    base = pd.read_parquet(Path(args.heads) / "val_readouts.parquet").set_index("UID")
    ro = base.loc[[uids["val"][i] for i in va_idx]].copy()
    ro[f"phi_{ch}"] = phi
    ro.reset_index().to_parquet(out / "val_readouts.parquet", index=False)   # adapted channel + the other E-a heads
    summary = {"step": "R019 E-c adaptation", "source": "taix", "channel": ch, "adaptation": A, "note": verdict_note,
               "best_epoch": best_epoch, "holdout_ce_best": best, "train_films": int(len(tr_idx)),
               "holdout_films": int(len(ho_idx)), "val_films": int(ok.sum()), "revalidation": rep,
               "minutes": round((time.time() - t_start) / 60, 1)}
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    q = rep["qwk"]
    print(f"{ch} adapted: QWK {q['estimate']:.3f} "
          f"AUROC>=2 {rep['auroc_grade_ge2']:.3f} pass={rep['criterion']['pass']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
