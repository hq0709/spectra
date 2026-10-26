"""R019, step E-d for the finding heads of one source: freeze the weights and record each channel's status.

A channel is scored with its E-a head when it passed E-b, with its E-c adapted head when E-b failed and the one
adaptation passed its re-validation, and is unscored on this source otherwise. The frozen directory holds copies of
the weights used and `manifest.json`; files are made read-only. Device heads are frozen when they exist (their
training data, RANZCR CLiP and PadChest, wait for access). The manifest is what `r019_features.py --frozen` requires
before test-split films are read.

    python scripts/r019_freeze.py --heads ../runs/R019/heads/taix_findings_v1 \
        --adapted edema=../runs/R019/adapt/taix_edema_v1 --out ../runs/R019/frozen/taix_findings_v1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True, help="E-a/E-b run directory (heads.pt, metrics.json)")
    ap.add_argument("--adapted", nargs="*", default=[], help="channel=E-c run directory (adapted.pt, metrics.json)")
    ap.add_argument("--source", default="taix")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from spectra.config import constants

    C = constants()
    heads = Path(args.heads)
    eb = json.loads((heads / "metrics.json").read_text())
    adapted = dict(a.split("=", 1) for a in args.adapted)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: a frozen evaluator is never overwritten")
    out.mkdir(parents=True)
    shutil.copy2(heads / "heads.pt", out / "heads.pt")
    channels = {}
    for ch, r in eb["channels"].items():
        rec = {"E-b": {"qwk": r["qwk"], "pass": r["criterion"]["pass"]}}
        if r["criterion"]["pass"]:
            rec.update(status="scored", weights="heads.pt", head="E-a")
        elif ch in adapted:
            ec = json.loads((Path(adapted[ch]) / "metrics.json").read_text())
            rv = ec["revalidation"]
            rec["E-c"] = {"qwk": rv["qwk"], "pass": rv["criterion"]["pass"], "note": ec["note"],
                          "best_epoch": ec["best_epoch"]}
            if rv["criterion"]["pass"] and ec["best_epoch"] >= 0:
                shutil.copy2(Path(adapted[ch]) / "adapted.pt", out / f"adapted_{ch}.pt")
                rec.update(status="scored", weights=f"adapted_{ch}.pt", head="E-c")
            else:
                rec.update(status="unscored", weights=None, head=None)
        else:
            rec.update(status="unscored", weights=None, head=None, note="E-b failed and no E-c run given")
        channels[ch] = rec
    feats = eb.get("features_meta", {})
    manifest = {"step": "E-d freeze of the finding heads", "source": args.source,
                "frozen_at": time.strftime("%Y-%m-%d %H:%M %Z"), "encoder": C["evaluator"]["encoder"],
                "encoder_revision": feats.get("encoder_revision"), "evaluation_rule": feats.get("rule"),
                "token_grid": C["evaluator"]["token_grid"], "channels": channels,
                "devices": "not frozen: device heads wait for RANZCR CLiP and PadChest access",
                "constants": {"evaluator": C["evaluator"], "evaluator_acceptance": C["evaluator_acceptance"]}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for f in out.iterdir():
        os.chmod(f, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(json.dumps({ch: (v["status"], v["head"]) for ch, v in channels.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
