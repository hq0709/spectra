"""Stage the released weights of the registered systems and evaluators into the HF cache (R004 smoke tests,
then every generation gate). Run on each machine for the systems it runs (`runs_on`).

    HF_HOME=... HF_TOKEN=... python scripts/fetch_weights.py --machine cluster [--systems prism ...]

Gated repositories need a token whose account accepted their terms (refine-logs/ACCESS_REQUESTS.md, section C);
a gated or missing repository is reported and the others continue.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectra.config import systems  # noqa: E402


def use_token_file() -> None:
    """The token file of the current login wins over HF_TOKEN in the environment: the `research` env's
    activate hook exports a stale token, which huggingface_hub would otherwise prefer (gated repos then 401)."""
    import os

    f = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    if f.is_file():
        tok = f.read_text().strip()
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = tok


# files a generation never reads: duplicate single-file checkpoints and README assets
IGNORE = {"black-forest-labs/FLUX.2-klein-4B": ["flux-2-klein-4b.safetensors"],
          "General-Medical-AI/UniMedVL": ["images/*"],
          "microsoft/rad-dino": ["training_images.csv"],
          "microsoft/radedit": ["*.png", "*.jpg"],
          "stanfordmimi/RoentGen-v2": ["*.png"]}
# further repositories a system's inference loads: (repo, allow_patterns or None for everything)
EXTRA = {"progemu": [("BAAI/Emu3-VisionTokenizer", None)],       # the Emu3 VQ image tokenizer
         "prism": [("stable-diffusion-v1-5/stable-diffusion-v1-5",   # VAE, text encoder, tokenizer, scheduler;
                    ["model_index.json", "scheduler/*", "text_encoder/config.json", "text_encoder/model.safetensors",
                     "tokenizer/*", "unet/config.json", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"])],
         # RadEdit's model card: the UNet is its own; VAE = SDXL-VAE, text encoder and tokenizer = BioViL-T
         "radedit": [("stabilityai/sdxl-vae", ["config.json", "diffusion_pytorch_model.safetensors"]),
                     ("microsoft/BiomedVLP-BioViL-T", ["config.json", "configuration_cxrbert.py", "modeling_cxrbert.py",
                                                       "model.safetensors", "special_tokens_map.json",
                                                       "tokenizer_config.json", "vocab.txt"])]}
ALLOW = {repo: pats for extras in EXTRA.values() for repo, pats in extras}  # the UNet is PRISM's own


def repos(machine: str, only=None):
    reg = systems()
    out = []
    for name, spec in reg.items():
        if not isinstance(spec, dict) or (only and name not in only):
            continue
        if spec.get("runs_on") not in ("both", machine):
            continue
        for key in ("weights", "init"):
            if spec.get(key):
                out.append((name, spec[key]))
        out += [(name, r) for r, _ in EXTRA.get(name, [])]
    if not only:
        ev = reg.get("evaluator", {})
        out += [("evaluator", ev["encoder"])] + [("evaluator", r) for r in ev.get("secondary", []) if "/" in r]
        out += [("extractor", reg["extractor"]["model"])]
    seen, uniq = set(), []
    for n, r in out:
        if r not in seen:
            seen.add(r)
            uniq.append((n, r))
    return uniq


def main(argv=None) -> int:
    use_token_file()
    from huggingface_hub import snapshot_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=["cluster", "mimic"], required=True)
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--max-workers", type=int, default=4)
    args = ap.parse_args(argv)
    failed = []
    for name, repo in repos(args.machine, args.systems):
        try:
            path = snapshot_download(repo, allow_patterns=ALLOW.get(repo), ignore_patterns=IGNORE.get(repo),
                                     max_workers=args.max_workers)
            print(f"{name}: {repo} -> {path}", flush=True)
        except Exception as e:  # report every repository, then fail
            failed.append(repo)
            print(f"{name}: {repo} FAILED {type(e).__name__}: {str(e).splitlines()[0][:160]}", file=sys.stderr, flush=True)
    print(f"done; failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
