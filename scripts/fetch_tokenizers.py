"""Stage the tokenizer files (never the weights) of every registered system into the HF cache, so the
R003 round-trip check runs offline on either machine with each system's actual tokenizer.

    HF_HOME=... HF_TOKEN=... python scripts/fetch_tokenizers.py [--systems prism progemu ...]

Gated repositories need a token whose account accepted their terms. Staging loads every tokenizer once
online (so later offline loads succeed), which executes remote tokenizer code for systems marked
`tokenizer_remote_code: true` (Emu3 for ProgEmu: its tokenization_emu3.py and configuration_emu3.py were
read before the first run); the R003 check executes it again.
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


PATTERNS = ["tokenizer*", "*.tiktoken", "vocab*", "merges.txt", "special_tokens_map.json", "added_tokens.json",
            "chat_template*", "config.json", "tokenization_*.py", "configuration_*.py", "*_vision_tokens.txt", "*.model"]


def main(argv=None) -> int:
    use_token_file()
    from huggingface_hub import snapshot_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="*", default=None)
    args = ap.parse_args(argv)
    failed = []
    for name, spec in systems().items():
        if not isinstance(spec, dict) or "tokenizer" not in spec or (args.systems and name not in args.systems):
            continue
        sub = spec.get("tokenizer_subfolder")
        allow = [f"{sub}/{p}" for p in PATTERNS] if sub else PATTERNS
        try:
            path = snapshot_download(spec["tokenizer"], allow_patterns=allow, max_workers=2)
            # one online load records the optional files a repository lacks (.no_exist markers), without
            # which transformers refuses the offline load (e.g. tokenizer/config.json in diffusers layouts)
            from spectra.tokenizers import HFTokenizer

            n = len(HFTokenizer(spec["tokenizer"], subfolder=sub, trust_remote_code=bool(spec.get("tokenizer_remote_code")))
                    .encode("endotracheal tube placement"))
            print(f"{name}: {spec['tokenizer']}{'/' + sub if sub else ''} -> {path} (probe: {n} tokens)")
        except Exception as e:  # report every system, then fail
            failed.append(name)
            print(f"{name}: {spec['tokenizer']} FAILED {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
