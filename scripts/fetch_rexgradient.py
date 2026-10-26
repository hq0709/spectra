"""Stage the needed ReXGradient-160K metadata (HF `rajpurkarlab/ReXGradient-160K`, gated, non-commercial DUA) into the
HF cache on the login node (compute nodes have no internet): README, LICENSE and the view-position film tables of the
val and public test splits (the test pool; they carry the report sections), about 40 MB. Not fetched: the train
tables (the extractor prompt is frozen, so no tuning reads them; all 119 CheXTemporal ReXGradient gold pairs are in
public test), the redundant CSV / plain JSON copies and the RadGame bounding boxes. The account behind
$HF_HOME/token must have accepted the dataset terms.

Images are not fetched. DUA 3.A.5 allows downloading only the portion the project needs and prohibits downloading
the entire repository, and the images come as one zstd-compressed tar split into deid_png.part00-09 (10 x 15.5 GB;
probed 2026-10-21: a single frame, no independent frames in sampled windows), so the val + public test images cannot
be read without downloading the whole archive. They wait for the data provider's answer
(refine-logs/ACCESS_REQUESTS.md).

    HF_HOME=... python scripts/fetch_rexgradient.py [--max-workers 4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_weights import use_token_file  # noqa: E402

REPO = "rajpurkarlab/ReXGradient-160K"


def main(argv=None) -> int:
    use_token_file()
    from huggingface_hub import snapshot_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=4)
    args = ap.parse_args(argv)
    pats = ["README.md", "LICENSE", "metadata/valid_metadata_view_position.json", "metadata/test_metadata_view_position.json"]
    path = snapshot_download(REPO, repo_type="dataset", allow_patterns=pats, max_workers=args.max_workers)
    print(f"metadata -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
