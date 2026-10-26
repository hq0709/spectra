"""Report extraction (M5 on the MIMIC server; X4 and D3 on this cluster): the local labeler of references/schema.py.

Qwen3-32B served by vLLM, temperature 0, decoding guided by the JSON schema, thinking disabled (Qwen3's chat
template, enable_thinking=False). Every output passes validate_extraction (schema, reference fields, verbatim
evidence spans) or is kept as a rejection with its reason; a rejected report yields no reference.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

from .schema import PROMPT, SCHEMA, validate_extraction

MODEL = "Qwen/Qwen3-32B"


def record(report_id: str, text: str, completion: str, finish_reason: Optional[str], n_tokens: int) -> Dict:
    """One output line: the parsed extraction and its rejection reason (None when accepted)."""
    rec = {"report_id": report_id, "chars": len(text), "finish_reason": finish_reason, "output_tokens": n_tokens}
    if finish_reason == "length":
        return {**rec, "extraction": None, "rejected": "truncated"}
    try:
        ex = json.loads(completion)
    except json.JSONDecodeError as e:
        return {**rec, "extraction": None, "rejected": f"json:{e.msg}"}
    return {**rec, "extraction": ex, "rejected": validate_extraction(ex, text)}


class Extractor:
    def __init__(self, model: str = MODEL, tensor_parallel: int = 1, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.90, max_tokens: int = 1536, seed: int = 0):
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self.model = model
        self.llm = LLM(model=model, tensor_parallel_size=tensor_parallel, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, seed=seed, enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed,
                                     structured_outputs=StructuredOutputsParams(json=SCHEMA))

    def prompt(self, text: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": PROMPT.format(report=text)}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    def extract(self, items: Sequence[Tuple[str, str]]) -> List[Dict]:
        outs = self.llm.generate([self.prompt(t) for _, t in items], self.params, use_tqdm=False)
        return [record(rid, text, o.outputs[0].text, o.outputs[0].finish_reason, len(o.outputs[0].token_ids))
                for (rid, text), o in zip(items, outs)]


# ---------------------------------------------------------------------------------------------
# report sources: (report id, extraction text) pairs
# ---------------------------------------------------------------------------------------------
def openi_items(snapshot_dir: str) -> List[Tuple[str, str]]:
    """Open-i / IU-Xray reports (HF `ykumards/open-i`, CC BY-ND 4.0): open real reports for pipeline checks."""
    from pathlib import Path

    import pyarrow.parquet as pq

    from .sections import extraction_text

    out = []
    for f in sorted(Path(snapshot_dir).glob("data/*.parquet")):
        t = pq.read_table(f, columns=["uid", "comparison", "findings", "impression"]).to_pylist()
        for r in t:
            secs = {k: (r[k] or "").strip() for k in ("comparison", "findings", "impression")}
            text = extraction_text(secs)
            if text:
                out.append((f"openi-{r['uid']}", text))
    return out


def mimic_items(mimic_config: str, icu_only: bool = True) -> List[Tuple[str, str]]:
    """MIMIC-CXR 2.1.0 reports of the F1 films (ICU films by default), sectioned as in M4. MIMIC server only."""
    from ..adapters.mimic import MimicAdapter
    from ..config import load_yaml
    from .sections import extraction_text, sections

    cfg = load_yaml(mimic_config)
    ad = MimicAdapter(cfg["cxr_jpg_dir"], cfg["cxr_reports_dir"], cfg["mimiciv_dir"])
    keep = {f.study_id for f in ad.films() if f.stay_id is not None or not icu_only}
    out = []
    for sid, raw in ad.reports().items():
        if sid in keep:
            text = extraction_text(sections(raw))
            if text:
                out.append((sid, text))
    return out


def jsonl_items(path: str) -> List[Tuple[str, str]]:
    out = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out.append((str(r["id"]), r["text"]))
    return out
