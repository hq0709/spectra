"""R019, steps E-a and E-b for the device heads on RANZCR CLiP (FINAL_PROPOSAL.md, Training Plan, head table;
constants.yaml `evaluator`).

Tasks (two-class presence, readout psi_d = P(present)): airway tube (RANZCR "ETT", which includes tracheostomy;
auxiliary), any central venous catheter (auxiliary), enteric tube (NGT, any grade; secondary E2 endpoint). Same head,
loss and recipe as the finding heads (r019_heads.py): one attentive head per task on frozen RAD-DINO tokens, masked
cross-entropy, epoch chosen by the cross-entropy of a 10 % salted-hash patient holdout of the dev split.
E-b: held-out real films = the RANZCR val patients (salted-hash split, constants.yaml); the enteric head is accepted at
AUROC >= 0.90 (section 12, point estimate decides); the auxiliary heads have no
criterion. Presence thresholds are not set here: they are set per E2 source (MIMIC, PadChest) on that source's
development films. With PadChest dev, E-a is rerun on both sources (masked multi-source heads) and validated again.

    python scripts/r019_device_heads.py --features ../runs/R019/features/ranzcr --out ../runs/R019/heads/ranzcr_devices_v1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from r019_heads import batched_logits, holdout_mask  # noqa: E402

CRITERION = {"enteric": "enteric"}   # task -> key of evaluator_acceptance.presence_auroc (auxiliary tasks: none)


def load_split(features: Path, split: str, table):
    import numpy as np

    uids = (features / f"{split}.uids.txt").read_text().split()
    tok = np.load(features / f"{split}.tokens.f16.npy", mmap_mode="r")
    m = table.set_index("uid").loc[uids]
    if (m["split"] != split).any():
        raise SystemExit(f"{split}: feature rows from another split")
    return uids, tok, m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "ranzcr.yaml"))
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
    from spectra.evaluator.data import ranzcr_table
    from spectra.evaluator.heads import Heads, masked_loss, readouts
    from spectra.evaluator.metrics import presence_report, rank_auroc

    C = constants()
    ev, acc = C["evaluator"], C["evaluator_acceptance"]
    tasks = dict(ev["device_tasks_ranzcr"])
    order = list(tasks)
    cfg = source_config(args.config)
    table = ranzcr_table(cfg["labels"], cfg["images"], ev["ranzcr_val_fraction"], ev["ranzcr_split_salt"])
    feats, out = Path(args.features), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = args.device

    uids, tok_np, m = load_split(feats, "dev", table)
    y_all = m[order].to_numpy().astype(int)
    hold = holdout_mask(m["patient"].to_numpy(), ev["holdout_salt"], ev["holdout_fraction"])
    tr_idx, ho_idx = np.flatnonzero(~hold), np.flatnonzero(hold)
    print(f"dev films {len(tr_idx)} ({m.patient[~hold].nunique()} patients); holdout {len(ho_idx)} "
          f"({m.patient[hold].nunique()} patients); prevalence {dict(zip(order, y_all.mean(0).round(3)))}", flush=True)
    tok = torch.as_tensor(np.ascontiguousarray(tok_np)).to(dev)
    y = torch.as_tensor(y_all, dtype=torch.long, device=dev)
    msk = torch.ones_like(y, dtype=torch.bool)

    model = Heads(tasks, dim=tok.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = math.ceil(len(tr_idx) / args.batch)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / steps_per_epoch) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    tr_t = torch.as_tensor(tr_idx)
    log, best, best_state, best_epoch, bad = [], float("inf"), None, -1, 0
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
        rec = {"epoch": ep, "train_ce": run / len(tr_idx), "holdout_ce": ho_ce,
               "holdout_auroc": {t: rank_auroc(ro[t].numpy(), y_all[ho_idx, i]) for i, t in enumerate(order)},
               "lr": sched.get_last_lr()[0], "s": round(time.time() - t0, 1)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if ho_ce < best - 1e-5:
            best, bad, best_epoch = ho_ce, 0, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    del tok

    # E-b on the held-out real films of the RANZCR val patients
    v_uids, v_tok, vm = load_split(feats, "val", table)
    v_tok_t = torch.as_tensor(np.ascontiguousarray(v_tok)).to(dev)
    lg = batched_logits(model, v_tok_t, torch.arange(len(v_uids), device=dev), dev)
    ro = {t: v.numpy() for t, v in readouts(lg).items()}
    pats = vm["patient"].to_numpy()
    res = {t: presence_report(ro[t], vm[t].to_numpy(), pats,
                              acc["presence_auroc"][CRITERION[t]] if t in CRITERION else None) for t in order}
    pd.DataFrame({"uid": v_uids, "patient": pats, **{f"psi_{t}": ro[t] for t in order},
                  **{t: vm[t].to_numpy() for t in order}}).to_parquet(out / "val_readouts.parquet", index=False)
    torch.save({"state_dict": best_state, "tasks": tasks, "order": order, "dim": int(v_tok.shape[-1]),
                "tokens": int(v_tok.shape[1]), "best_epoch": best_epoch, "args": vars(args)}, out / "heads.pt")
    summary = {"step": "R019 E-a/E-b devices", "source": "ranzcr", "dev_films": int(len(tr_idx)),
               "holdout_films": int(len(ho_idx)), "val_films": int(len(v_uids)), "val_patients": int(len(set(pats))),
               "best_epoch": best_epoch, "holdout_ce": best, "tasks": res,
               "features_meta": json.loads((feats / "dev.meta.json").read_text())}
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    for t in order:
        a = res[t]["auroc"]
        crit = res[t]["criterion"]
        print(f"{t}: AUROC {a['estimate']:.3f} prevalence {res[t]['prevalence']:.3f} "
              f"{'pass=' + str(crit['pass']) if crit else 'auxiliary'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
