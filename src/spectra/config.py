"""Configuration loading.

The shared, prespecified files (constants, MIMIC item ontology, dependency table, system registry) ship
inside the package (`spectra/resources/*.yaml`) so an installed or relocated environment finds them;
`SPECTRA_CONFIG_DIR` overrides them for an audited deviation. Per-source configs (`config/<source>.yaml`)
are passed explicitly by path.
"""
from __future__ import annotations

import os
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Dict

import yaml

SHARED = ("constants", "mimic_itemids", "dependency_table", "systems", "templates")


@lru_cache(maxsize=None)
def _shared(name: str) -> Dict[str, Any]:
    override = os.environ.get("SPECTRA_CONFIG_DIR")
    if override and (Path(override) / f"{name}.yaml").exists():
        text = (Path(override) / f"{name}.yaml").read_text()
    else:
        text = resources.files("spectra.resources").joinpath(f"{name}.yaml").read_text()
    return yaml.safe_load(text)


def load_yaml(name_or_path: str | os.PathLike) -> Dict[str, Any]:
    name = str(name_or_path)
    if name in SHARED:
        return _shared(name)
    path = Path(name_or_path)
    if not path.suffix:
        path = path.with_suffix(".yaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def constants() -> Dict[str, Any]:
    return _shared("constants")


def mimic_ontology_config() -> Dict[str, Any]:
    return _shared("mimic_itemids")


def dependency_table() -> Dict[str, Any]:
    return _shared("dependency_table")


def systems() -> Dict[str, Any]:
    return _shared("systems")


def template(system: str, tid: str = "t1") -> str:
    """Prompt template `tid` of a prompted system, with the fixed answer format filled in; exactly one
    `{input}` slot remains. A system without an entry receives the serialized input unwrapped."""
    t = _shared("templates").get(system, {}).get(tid, "{input}")
    t = t.replace("{answer_format}", _shared("templates")["answer_format"].strip())
    if t.count("{input}") != 1:
        raise ValueError(f"template {system}/{tid} must contain exactly one {{input}}")
    return t


def source_config(path: str | os.PathLike) -> Dict[str, Any]:
    cfg = load_yaml(path)
    if "source" not in cfg:
        raise ValueError(f"{path}: a source config needs a 'source' key")
    return cfg
