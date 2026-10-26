"""R019, steps E-a and E-b for the finding heads (FINAL_PROPOSAL.md, Training Plan; constants.yaml `evaluator`).

E-a: attentive heads on frozen RAD-DINO tokens of TAIX-Ray train (r019_features.py), one per primary channel
(edema = PulmonaryCongestion, effusion_L, effusion_R; grades 0-4), masked cross-entropy; one configuration, no
grid; the epoch is chosen by the cross-entropy of a 10 % patient holdout of the train split.
E-b: held-out real films = TAIX-Ray val. Criterion (section 12): quadratic weighted kappa >= 0.60 per channel
between the film's rounded expected grade and the radiologist grade (point estimate decides). Test-split films are never read here.

    python scripts/r019_heads.py --features ../runs/R019/features/taix --out ../runs/R019/heads/taix_findings_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

COLUMNS = {"edema": "PulmonaryCongestion", "effusion_L": "PleuralEffusion_Left", "effusion_R": "PleuralEffusion_Right"}


def load_split(features: Path, split: str, meta):
    import numpy as np

    uids = (features / f"{split}.uids.txt").read_text().split()
    tok = np.load(features / f"{split}.tokens.f16.npy", mmap_mode="r")
    m = meta.set_index("UID").loc[uids]
    if (m["Split"] != split).any():
        raise SystemExit(f"{split}: feature rows from another split")
    return uids, tok, m


def holdout_mask(patients, salt: str, frac: float):
    import numpy as np

    h = [int(hashlib.sha256(f"{salt}|{p}".encode()).hexdigest()[:12], 16) / 16 ** 12 for p in patients]
    return np.asarray(h) < frac


def batched_logits(model, tok, idx, device, bs=4096):
    import torch

    outs = {}
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(idx), bs):
            x = torch.as_tensor(tok[idx[s:s + bs]]).to(device, non_blocking=True)
            for t, l in model(x).items():
                outs.setdefault(t, []).append(l.float().cpu())
    return {t: torch.cat(v) for t, v in outs.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "taix.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd
    import torch

    from spectra.config import constants, source_config
    from spectra.evaluator.heads import Heads, masked_loss, readouts
    from spectra.evaluator.metrics import discretize, ordinal_report, weighted_kappa

    C = constants()
    ev, acc = C["evaluator"], C["evaluator_acceptance"]
    tasks = dict(ev["finding_grades"])
    order = list(tasks)
    cfg = source_config(args.config)
    meta = pd.read_parquet(cfg["metadata"])
    feats, out = Path(args.features), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = args.device

    uids, tok_np, m = load_split(feats, "train", meta)
    y_all = np.stack([m[COLUMNS[t]].to_numpy() for t in order], 1)
    mask_all = y_all >= 0
    hold = holdout_mask(m["PatientID"].to_numpy(), ev["holdout_salt"], ev["holdout_fraction"])
    tr_idx, ho_idx = np.flatnonzero(~hold), np.flatnonzero(hold)
    print(f"train films {len(tr_idx)} ({m.PatientID[~hold].nunique()} patients); holdout {len(ho_idx)} "
          f"({m.PatientID[hold].nunique()} patients)", flush=True)
    tok = torch.as_tensor(np.ascontiguousarray(tok_np)).to(dev)          # [N, T, 768] float16 on the GPU
    y = torch.as_tensor(np.where(mask_all, y_all, 0), dtype=torch.long, device=dev)
    msk = torch.as_tensor(mask_all, device=dev)

    model = Heads(tasks, dim=tok.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = math.ceil(len(tr_idx) / args.batch)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / steps_per_epoch) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    tr_t = torch.as_tensor(tr_idx)
    log, best, best_state, bad = [], float("inf"), None, 0
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        perm = tr_t[torch.randperm(len(tr_t), generator=g)].to(dev)
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = masked_loss(model(tok[b]), y[b], msk[b], order)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss) * len(b)
        lg = batched_logits(model, tok, torch.as_tensor(ho_idx).to(dev), dev)
        ho_ce = float(masked_loss({t: lg[t].to(dev) for t in order}, y[ho_idx], msk[ho_idx], order))
        ro = readouts(lg)
        ho_qwk = {t: weighted_kappa(discretize(ro[t].numpy(), tasks[t]), y_all[ho_idx, i], tasks[t])
                  for i, t in enumerate(order)}
        rec = {"epoch": ep, "train_ce": run / len(tr_idx), "holdout_ce": ho_ce, "holdout_qwk": ho_qwk,
               "lr": sched.get_last_lr()[0], "s": round(time.time() - t0, 1)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if ho_ce < best - 1e-5:
            best, bad = ho_ce, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    del tok

    # E-b on the held-out real films of TAIX-Ray val
    v_uids, v_tok, vm = load_split(feats, "val", meta)
    v_tok_t = torch.as_tensor(np.ascontiguousarray(v_tok)).to(dev)
    lg = batched_logits(model, v_tok_t, torch.arange(len(v_uids), device=dev), dev)
    ro = {t: v.numpy() for t, v in readouts(lg).items()}
    pats = vm["PatientID"].to_numpy()
    res = {}
    for t in order:
        gr = vm[COLUMNS[t]].to_numpy()
        ok = gr >= 0
        res[t] = ordinal_report(ro[t][ok], gr[ok], pats[ok], tasks[t],
                                acc["ordinal_weighted_kappa_vs_taix"])
    pd.DataFrame({"UID": v_uids, "PatientID": pats, **{f"phi_{t}": ro[t] for t in order}}).to_parquet(
        out / "val_readouts.parquet", index=False)
    torch.save({"state_dict": best_state, "tasks": tasks, "order": order, "dim": int(v_tok.shape[-1]),
                "tokens": int(v_tok.shape[1]), "best_epoch": best_epoch, "args": vars(args)}, out / "heads.pt")
    summary = {"step": "R019 E-a/E-b findings", "source": "taix", "train_films": int(len(tr_idx)),
               "holdout_films": int(len(ho_idx)), "val_films": int(len(v_uids)), "val_patients": int(len(set(pats))),
               "best_epoch": best_epoch, "holdout_ce": best, "kappa_weights": acc["kappa_weights"],
               "discretization": acc["ordinal_discretization"], "channels": res,
               "features_meta": json.loads((feats / "train.meta.json").read_text())}
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    for t in order:
        q = res[t]["qwk"]
        print(f"{t}: QWK {q['estimate']:.3f} "
              f"linear {res[t]['linear_kappa']:.3f} AUROC>=2 {res[t]['auroc_grade_ge2']:.3f} "
              f"pass={res[t]['criterion']['pass']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
