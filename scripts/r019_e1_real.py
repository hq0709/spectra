"""R019 diagnostic on development data: E1 with the real follow-up films as the "system" output (the real-image
reference performance of the evaluator on factual follow-ups) and copy-prior (exactly 1/2, property P2a).

The val-split E1 sets come from the authoritative TAIX-Ray adapter; phi_c are the evaluator readouts of the real
films (r019_heads.py, val_readouts.parquet). C_c is patient-macro (mean ± SD over patients). Test films are not read.

    python scripts/r019_e1_real.py --heads ../runs/R019/heads/taix_findings_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def macro_sd(values: dict):
    from spectra.evaluator.metrics import patient_mean_sd

    r = patient_mean_sd(values.values())
    return {"C": r["mean"], "sd": r["sd"], "N": r["N"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "taix.yaml"))
    ap.add_argument("--split", default="val", choices=["val"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import pandas as pd

    from spectra.adapters.taix import TaixAdapter
    from spectra.config import constants, source_config
    from spectra.e1 import concordance, query_id

    C = constants()
    cfg = source_config(args.config)
    ro = pd.read_parquet(Path(args.heads) / f"{args.split}_readouts.parquet").set_index("UID")
    chans = C["e1"]["primary_channels"]
    phi = {u: {c: float(r[f"phi_{c}"]) for c in chans} for u, r in ro.iterrows()}
    _, _, chosen = TaixAdapter(cfg["metadata"]).e1_sets(args.split)
    contrasts = [(qs.prior.patient_id, c) for qs, cons in chosen for c in cons]
    real, copy = {}, {}
    for qs, _ in chosen:
        for f in qs.later:
            real[(query_id(qs.prior, f), 0)] = phi[f.study_id]
            copy[(query_id(qs.prior, f), 0)] = phi[qs.prior.study_id]
    out = {"split": args.split, "priors": len(chosen), "contrasts": len(contrasts),
           "patients": len({p for p, _ in contrasts})}
    for name, rd in (("real_follow_up", real), ("copy_prior", copy)):
        res = concordance(contrasts, rd, [0])
        out[name] = {ch: macro_sd(r["patients"]) for ch, r in sorted(res.items())}
    bad = {ch: r["C"] for ch, r in out["copy_prior"].items() if r["C"] != 0.5}
    out["copy_prior_exact_half"] = not bad
    (Path(args.heads) / f"e1_real_{args.split}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items()}, indent=1))
    if bad:
        print(f"P2a FAILED: copy-prior C != 1/2 on {bad}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
