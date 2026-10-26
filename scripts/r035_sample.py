"""R035 readout: one trained reference configuration on the Gate-2 TAIX-Ray val dev queries (runs/R020/taix_dev,
mode R, Track Delta). Writes outputs and records in r020_gate2.py's format under <out>/gen/ref_<config>_t1/, so
`r020_gate2.py score --out <out>` reads them through the same frozen heads and concordance code.

The prior latent is the cached VAE latent of the prior's evaluation film (the training input); the starting noise is
prior_noise(prior, seed); one query per forward pass (batch 1), so a model that ignores Delta gives the same output
for every query of a prior bit for bit (prior_only: E1 C = 1/2 exactly).

    python scripts/r035_sample.py --ckpt ../runs/R035/prior_only/final.pt --dev ../runs/R020/taix_dev \
        --latents ../runs/R035/latents --out ../runs/R035/e1_dev
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dev", required=True, help="Gate-2 dev directory with queries.json")
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", choices=["prior_only", "prior_delta"], help="for checkpoints without it (last.pt)")
    ap.add_argument("--start", choices=["prior", "noise"], default=None, help="default: constants reference.sampler.start")
    ap.add_argument("--limit-priors", type=int, default=None, help="smoke test only")
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    from diffusers import AutoencoderKL

    from spectra.config import constants
    from spectra.reference import decode, load_reference, prior_noise, sample, upgrade_state_dict

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    R = constants()["reference"]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ck.get("config") or args.config
    unet, cond, _ = load_reference(R["init"], config, R["context_dim"], R["delta_embedding_dim"])
    unet.load_state_dict(upgrade_state_dict(ck["unet"])); cond.load_state_dict(ck["cond"])
    dev = torch.device("cuda")
    unet.to(dev).eval(); cond.to(dev).eval()
    vae = AutoencoderKL.from_pretrained(R["init"], subfolder="vae").to(dev).eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(args.dev) / "queries.json", out / "queries.json")
    q = json.loads((out / "queries.json").read_text())
    seeds = [constants()["generation"]["seed_base"] + s for s in range(q["seeds"])]
    lat_dir = Path(args.latents)
    row = {u: i for i, u in enumerate((lat_dir / f"{q['split']}.uids.txt").read_text().split())}
    lat = np.load(lat_dir / f"{q['split']}.lat.f16.npy", mmap_mode="r")
    d = out / "gen" / f"ref_{config}_t1"
    d.mkdir(parents=True, exist_ok=True)
    S = R["sampler"]
    start = args.start or S["start"]
    first = {}                                   # prior_only: (prior, seed) -> first latent, for the identity check
    with open(d / "records.0of1.jsonl", "w") as fh:
        warm = True
        for p in q["priors"][:args.limit_priors]:
            prior = torch.from_numpy(np.asarray(lat[row[p["prior"]]], dtype=np.float32))[None].to(dev)
            for f in p["later"]:
                for s in seeds:
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                    x = sample(unet, cond, R["init"], prior, torch.tensor([f["delta_h"]], device=dev),
                               prior_noise(p["prior"], s, (1, 4, 64, 64), dev), steps=S["steps"], start=start)
                    im = decode(vae, x)[0]
                    torch.cuda.synchronize()
                    sec = time.time() - t0
                    if config == "prior_only":
                        ref = first.setdefault((p["prior"], s), x)
                        if not torch.equal(ref, x):
                            raise SystemExit(f"prior_only output differs across queries of {p['prior']} seed {s}")
                    name = f"{f['query'].replace('->', '__')}_s{s}.png"
                    im.save(d / name)
                    fh.write(json.dumps({"query": f["query"], "seed": s, "system": f"ref_{config}", "template": "t1",
                                         "load_seconds": None, "gpu": torch.cuda.get_device_name(0), "input": None,
                                         "warm": warm, "seconds": round(sec, 3),
                                         "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                                         "kind": "image", "file": name, "native_size": list(im.size),
                                         "steps": ck.get("steps"), "start": start}) + "\n")
                    fh.flush()
                    warm = False
    print(f"ref_{config}: {sum(len(p['later']) for p in q['priors'])} queries x {len(seeds)} seeds", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
