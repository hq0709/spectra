import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEMO = Path(os.environ.get("SPECTRA_MIMIC_DEMO", "/path/to/workspace/data/mimic-iv-demo/2.2"))
DEMO_ICU = DEMO / "icu"
TAIX_META = Path(os.environ.get("SPECTRA_TAIX_META",
                                "/path/to/workspace/data/taix-ray-metadata/taix_metadata.parquet"))
TAIX_COUNTS = ROOT.parent / "refine-logs" / "feasibility" / "taix_adapter_counts_2026-10-20.json"

needs_demo = pytest.mark.skipif(not DEMO_ICU.exists(), reason="open MIMIC-IV demo 2.2 not staged")
needs_taix = pytest.mark.skipif(not TAIX_META.exists(), reason="TAIX-Ray metadata not staged")


@pytest.fixture(scope="session")
def demo_tables():
    """All demo stays, Track A and Track B sources (the demo is small)."""
    from spectra.cer import IcuTables, MimicOntology

    if not DEMO_ICU.exists():
        pytest.skip("open MIMIC-IV demo 2.2 not staged")
    return IcuTables.from_dir(DEMO_ICU, DEMO / "hosp", stay_ids=None, ontology=MimicOntology(), track_b=True)
