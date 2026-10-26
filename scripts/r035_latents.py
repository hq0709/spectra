"""R035 input: RoentGen-v2 VAE latents (posterior mean x scaling factor, [N, 4, 64, 64] float16) of the TAIX-Ray
evaluation-rule films (runs/R019/films/taix/<split>.u8.npy, 512^2 uint8, built by r019_adapt.py). Rows follow
<split>.uids.txt. Buffered in RAM and written once (no memmap writes on network storage).

    python scripts/r035_latents.py --films ../runs/R019/films/taix --out ../runs/R035/latents --splits train val
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
    ap.add_argument("--films", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    from diffusers import AutoencoderKL

    from spectra.config import constants
    from spectra.reference import encode

    init = constants()["reference"]["init"]
    vae = AutoencoderKL.from_pretrained(init, subfolder="vae", torch_dtype=torch.bfloat16).cuda().eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        films = np.load(Path(args.films) / f"{split}.u8.npy", mmap_mode="r")
        n = len(films) if args.limit is None else min(args.limit, len(films))
        lat = np.empty((n, 4, 64, 64), dtype=np.float16)
        t0 = time.time()
        for i in range(0, n, args.batch):
            x = torch.from_numpy(np.ascontiguousarray(films[i:i + args.batch][: n - i])).cuda()
            with torch.no_grad():
                lat[i:i + len(x)] = encode(vae, x).float().cpu().numpy().astype(np.float16)
            if (i // args.batch) % 200 == 0:
                print(f"{split}: {i + len(x)}/{n} {(i + len(x)) / (time.time() - t0):.0f}/s", flush=True)
        np.save(out / f"{split}.lat.f16.npy", lat)
        uids = (Path(args.films) / f"{split}.uids.txt").read_text().split()[:n]
        (out / f"{split}.uids.txt").write_text("\n".join(uids) + "\n")
        meta = {"split": split, "n": n, "init": init, "seconds": round(time.time() - t0, 1), "std": float(lat[:2048].astype(np.float32).std()),
                "source": str(Path(args.films) / f"{split}.u8.npy")}
        (out / f"{split}.meta.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
