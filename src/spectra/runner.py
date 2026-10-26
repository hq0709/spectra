"""Built-in local runner: shards a step over the visible GPUs in resumable shards, so no scheduler is
assumed on the MIMIC server. On the Slurm cluster the same `spectra` commands are wrapped by sbatch.

A shard is done when its marker file exists; re-running skips done shards (resumable)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence


def shard_slice(items: Sequence, shard: int, n_shards: int) -> List:
    if not 0 <= shard < n_shards:
        raise ValueError("shard index out of range")
    return [x for i, x in enumerate(items) if i % n_shards == shard]


def marker(out_dir: Path, step: str, shard: int, n_shards: int) -> Path:
    return out_dir / f".done.{step}.{shard:04d}-of-{n_shards:04d}"


def visible_gpus() -> List[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g for g in env.split(",") if g.strip() != ""]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30)
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except (FileNotFoundError, subprocess.SubprocessError):
        return []


def run_local(step_argv: Sequence[str], out_dir: str, n_shards: int, gpus: Optional[Sequence[str]] = None,
              max_parallel: Optional[int] = None, poll_s: float = 5.0, gpus_per_shard: int = 1) -> dict:
    """Launch `spectra <step_argv> --shard i --n-shards n` per pending shard on `gpus_per_shard` GPUs each
    (e.g. 2 for Qwen3-32B with --tp 2 on 40-48 GB cards), resumable."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gpus = list(gpus if gpus is not None else visible_gpus()) or ["cpu"]
    if gpus != ["cpu"] and gpus_per_shard > 1:
        if len(gpus) < gpus_per_shard:
            raise ValueError(f"{len(gpus)} GPUs cannot hold a shard of {gpus_per_shard}")
        gpus = [",".join(gpus[i:i + gpus_per_shard]) for i in range(0, len(gpus) - gpus_per_shard + 1, gpus_per_shard)]
    max_parallel = max_parallel or len(gpus)
    step = step_argv[0]
    pending = [i for i in range(n_shards) if not marker(out, step, i, n_shards).exists()]
    running, failed = {}, []
    free = list(gpus)
    while pending or running:
        while pending and free and len(running) < max_parallel:
            i = pending.pop(0)
            g = free.pop(0)
            env = dict(os.environ)
            pkg_root = str(Path(__file__).resolve().parents[1])  # works from a source tree or an install
            env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            if g != "cpu":
                env["CUDA_VISIBLE_DEVICES"] = g
            cmd = [sys.executable, "-m", "spectra.cli", *step_argv, "--shard", str(i), "--n-shards", str(n_shards),
                   "--out", str(out)]
            log = open(out / f"{step}.{i:04d}.log", "w")
            running[i] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), g, log)
        time.sleep(poll_s)
        for i, (p, g, log) in list(running.items()):
            if p.poll() is not None:
                log.close()
                free.append(g)
                del running[i]
                if p.returncode == 0:
                    marker(out, step, i, n_shards).write_text(json.dumps({"gpu": g, "t": time.time()}))
                else:
                    failed.append(i)
    return {"done": n_shards - len(failed), "failed": failed}
