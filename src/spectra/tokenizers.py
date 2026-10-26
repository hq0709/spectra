"""Tokenizers used by the round-trip check.

The round-trip check runs after each system's ACTUAL tokenization, so production checks use the
system's own tokenizer through :class:`HFTokenizer` (transformers). The two built-in tokenizers are
exact, deterministic stand-ins for unit tests on machines without model files:
- :class:`RegexTokenizer`: words, punctuation and whitespace runs are separate tokens (lossless).
- :class:`ClipLikeTokenizer`: CLIP-style lower-casing, BOS/EOS and a hard 77-token limit; long words
  cost several tokens (conservative relative to CLIP BPE for clinical vocabulary).
"""
from __future__ import annotations

import re
import zlib
from typing import Dict, List, Optional

_PIECE = re.compile(r"\s+|[A-Za-z]+|\d|[^\sA-Za-z\d]")


class RegexTokenizer:
    name = "regex"

    def __init__(self, max_len: Optional[int] = None):
        self.max_len = max_len
        self._vocab: Dict[int, str] = {}
        self.bos_len = 0

    def _id(self, piece: str) -> int:
        i = zlib.crc32(piece.encode()) & 0x7FFFFFFF
        self._vocab[i] = piece
        return i

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        return [self._id(p) for p in _PIECE.findall(text)]

    def decode(self, ids: List[int]) -> str:
        return "".join(self._vocab[i] for i in ids)


class ClipLikeTokenizer:
    """CLIP-style: lower-cased, whitespace dropped, words longer than 6 letters split into 4-letter pieces,
    BOS and EOS added, hard limit of 77 tokens including BOS/EOS. Decoding re-inserts single spaces."""

    name = "clip_like"
    BOS, EOS = -1, -2

    def __init__(self, max_len: int = 77):
        self.max_len = max_len
        self._vocab: Dict[int, str] = {}
        self.bos_len = 1

    def _pieces(self, text: str) -> List[str]:
        out = []
        for w in re.findall(r"[a-z]+|\d|[^\sa-z\d]", text.lower()):
            if w.isalpha() and len(w) > 6:
                parts = [w[i:i + 4] for i in range(0, len(w), 4)]
                out += [parts[0]] + ["##" + p for p in parts[1:]]
            else:
                out.append(w)
        return out

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        ids = []
        for p in self._pieces(text):
            i = zlib.crc32(p.encode()) & 0x7FFFFFFF
            self._vocab[i] = p
            ids.append(i)
        return [self.BOS] + ids + [self.EOS] if add_special else ids

    def decode(self, ids: List[int]) -> str:
        words: List[str] = []
        for i in ids:
            if i in (self.BOS, self.EOS):
                continue
            p = self._vocab[i]
            if p.startswith("##") and words:
                words[-1] += p[2:]
            else:
                words.append(p)
        return " ".join(words)  # lower-cased and space-joined, as CLIP decoding is; parsers must tolerate it


class HFTokenizer:
    """A system's own tokenizer from its released files (requires `transformers`)."""

    def __init__(self, name_or_path: str, max_len: Optional[int] = None, subfolder: Optional[str] = None,
                 trust_remote_code: bool = False):
        from transformers import AutoTokenizer, PretrainedConfig  # optional dependency

        # a tokenizer in a subfolder (diffusers layout) has no model config: its class comes from
        # tokenizer_config.json, and an empty config keeps the load offline-safe
        kw = {"subfolder": subfolder, "config": PretrainedConfig()} if subfolder else {}
        if trust_remote_code:  # e.g. Emu3 (ProgEmu): a tiktoken tokenizer shipped as repository code
            kw["trust_remote_code"] = True
        self.tok = AutoTokenizer.from_pretrained(name_or_path, **kw)
        self.name = f"hf:{name_or_path}"
        self.max_len = max_len
        probe = self.tok("x", add_special_tokens=True)["input_ids"]
        bare = self.tok("x", add_special_tokens=False)["input_ids"]
        self.bos_len = 1 if probe and bare and probe[0] != bare[0] else 0  # leading special tokens added by encode

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        return list(self.tok(text, add_special_tokens=add_special)["input_ids"])

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
