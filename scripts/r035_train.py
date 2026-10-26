"""R035: one configuration of the external reference pair (prior_only | prior_delta) on TAIX-Ray train pairs
(constants.yaml `reference`; spectra.reference). Both configurations use the same pair-draw schedule, batch, steps,
optimizer and seed and differ only in their conditioning. Multi-GPU via torchrun (data parallel; the global batch is
split across ranks, so every rank draws its slice of the shared schedule). The last step is kept; the dev-val loss on
fixed TAIX-Ray val pairs is logged, never used to choose anything.

    torchrun --nproc_per_node 4 scripts/r035_train.py --config prior_delta --latents ../runs/R035/latents \
        --out ../runs/R035/prior_delta [--steps N] [--pilot]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def pairs_for(split: str, latents: Path, R: dict):
    from spectra.adapters.taix import TaixAdapter
    from spectra.config import source_config
    from spectra.reference import taix_pairs

    cfg = source_config(Path(__file__).resolve().parents[1] / "config" / "taix.yaml")
    films = [f for f in TaixAdapter(cfg["metadata"]).films() if f.split == split]
    row = {u: i for i, u in enumerate((latents / f"{split}.uids.txt").read_text().split())}
    pairs = taix_pairs(films, tuple(R["taix_pairs_days"]), R["max_pairs_per_patient"], R["pair_seed"])
    return [(row[a], row[b], h) for a, b, h in pairs if a in row and b in row], len({f.patient_id for f in films})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["prior_only", "prior_delta"])
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None,
                    help="pilot: constants reference.pilot_steps; full run: the common N of both configurations")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--prior-lr-mult", type=float, default=None, help="development override of reference.prior_lr_mult")
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    import torch.distributed as dist

    from spectra.config import constants
    from spectra.reference import (diffusion_loss, load_reference, pair_schedule, prior_parameters, step_noise,
                                   upgrade_optimizer_state, upgrade_state_dict)

    R = constants()["reference"]
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")
    steps = args.steps or (R["pilot_steps"] if args.pilot else None)
    if steps is None:
        raise SystemExit("--steps is required for a full run (from the pilot's measured rate, constants reference)")
    if R["batch"] % world:
        raise SystemExit(f"global batch {R['batch']} not divisible by {world} ranks")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lat_dir = Path(args.latents)
    pairs, n_pat = pairs_for("train", lat_dir, R)
    vpairs, _ = pairs_for("val", lat_dir, R)
    lat = torch.from_numpy(np.load(lat_dir / "train.lat.f16.npy"))           # whole cache in RAM (~4.5 GB)
    vlat = torch.from_numpy(np.load(lat_dir / "val.lat.f16.npy"))
    sched_idx = pair_schedule(pairs, steps, R["batch"], R["pair_seed"])
    vfix = np.random.default_rng(0).choice(len(vpairs), size=min(512, len(vpairs)), replace=False)

    torch.manual_seed(R["init_seed"])                     # same initialization policy for every configuration
    unet, cond, noise_sched = load_reference(R["init"], args.config, R["context_dim"], R["delta_embedding_dim"])
    if args.grad_ckpt:
        unet.enable_gradient_checkpointing()
    unet.to(dev).train()
    cond.to(dev).train()
    params = list(unet.parameters()) + list(cond.parameters())
    pp = {id(p) for p in prior_parameters(unet)}
    mult = args.prior_lr_mult if args.prior_lr_mult is not None else R["prior_lr_mult"]
    opt = torch.optim.AdamW([{"params": [p for p in params if id(p) not in pp]},
                             {"params": prior_parameters(unet), "lr": R["lr"] * mult}],
                            lr=R["lr"], weight_decay=R["weight_decay"])
    lr_at = lambda s: min(1.0, (s + 1) / R["warmup_steps"])
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    start = 0
    ck = out / "last.pt"
    if ck.exists():                                       # resume: same schedule position
        st = torch.load(ck, map_location="cpu", weights_only=False)
        legacy = "conv_in.weight" in st["unet"]           # written before the prior conv had its own tensor
        unet.load_state_dict(upgrade_state_dict(st["unet"])); cond.load_state_dict(st["cond"])
        opt.load_state_dict(upgrade_optimizer_state(st["opt"], unet, cond, R["lr"], mult) if legacy else st["opt"])
        ss = dict(st["sched"])
        if legacy:
            ss["base_lrs"] = [R["lr"], R["lr"] * mult]
            ss["_last_lr"] = [g["lr"] for g in opt.param_groups]
        sch.load_state_dict(ss); start = st["step"]
    model_u = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[dev.index]) if ddp else unet
    model_c = torch.nn.parallel.DistributedDataParallel(cond, device_ids=[dev.index]) if ddp else cond
    per = R["batch"] // world
    log = open(out / f"train_log.rank{rank}.jsonl", "a")
    if rank == 0:
        (out / "run.json").write_text(json.dumps({"config": args.config, "steps": steps, "world": world, "recipe": R,
                                                  "train_pairs": len(pairs), "train_patients": n_pat,
                                                  "val_pairs": len(vpairs), "pilot": args.pilot,
                                                  "prior_lr_mult": mult}, indent=1))
    t0, tl = time.time(), time.time()
    for step in range(start, steps):
        idx = sched_idx[step][rank * per:(rank + 1) * per]
        pr = torch.tensor([pairs[i][0] for i in idx]); tg = torch.tensor([pairs[i][1] for i in idx])
        dh = torch.tensor([pairs[i][2] for i in idx], device=dev)
        nz, ts = step_noise(R["train_seed"], step, R["batch"], noise_sched.config.num_train_timesteps)
        nz, ts = nz[rank * per:(rank + 1) * per].to(dev), ts[rank * per:(rank + 1) * per].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = diffusion_loss(model_u, model_c, noise_sched, lat[tg].to(dev).float(), lat[pr].to(dev).float(), dh,
                                  noise=nz, t=ts)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, R["grad_clip"])
        opt.step()
        sch.step()
        if rank == 0 and (step % 50 == 0 or step == steps - 1):
            el = time.time() - tl
            rec = {"step": step, "loss": float(loss), "lr": sch.get_last_lr()[0],
                   "prior_w_norm": float(unet.conv_in.prior.weight.detach().norm()), "s_per_step": el / 50 if step else None,
                   "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            log.write(json.dumps(rec) + "\n"); log.flush(); tl = time.time()
        if rank == 0 and ((step + 1) % args.val_every == 0 or step == steps - 1):
            unet.eval(); cond.eval()
            g = torch.Generator(device=dev).manual_seed(7)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vl = [float(diffusion_loss(unet, cond, noise_sched, vlat[[vpairs[i][1] for i in vfix[j:j + 64]]].to(dev).float(),
                                           vlat[[vpairs[i][0] for i in vfix[j:j + 64]]].to(dev).float(),
                                           torch.tensor([vpairs[i][2] for i in vfix[j:j + 64]], device=dev), g))
                      for j in range(0, len(vfix), 64)]
            log.write(json.dumps({"step": step, "dev_val_loss": float(np.mean(vl))}) + "\n"); log.flush()
            unet.train(); cond.train()
        if rank == 0 and ((step + 1) % args.ckpt_every == 0 or step == steps - 1):
            torch.save({"unet": unet.state_dict(), "cond": cond.state_dict(), "opt": opt.state_dict(),
                        "sched": sch.state_dict(), "step": step + 1}, ck.with_suffix(".tmp"))
            os.replace(ck.with_suffix(".tmp"), ck)
    if rank == 0:
        total = time.time() - t0
        done = steps - start
        summ = {"config": args.config, "steps_run": done, "seconds": round(total, 1), "s_per_step": total / max(done, 1),
                "gpus": world, "gpu_hours_per_1000_steps": total / max(done, 1) * 1000 * world / 3600,
                "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
        (out / ("pilot.json" if args.pilot else "done.json")).write_text(json.dumps(summ, indent=1))
        if not args.pilot:
            torch.save({"unet": unet.state_dict(), "cond": cond.state_dict(), "config": args.config, "recipe": R,
                        "steps": steps}, out / "final.pt")
        print(json.dumps(summ), flush=True)
    if ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
