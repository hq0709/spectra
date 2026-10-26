"""Film streams of the evaluator: TAIX-Ray parquet row groups read in DataLoader workers as 512^2 uint8 evaluation
films (spectra.images.to_eval); RAD-DINO's processor runs on these batches in the main process (encoder.py)."""
from __future__ import annotations

import io
from pathlib import Path

import torch


def taix_tasks(images_dir: str, split: str, limit=None):
    """(shard, row group, rows) of one TAIX-Ray split, in file order; limit cuts whole row groups."""
    import pyarrow.parquet as pq

    out, n = [], 0
    for f in sorted(Path(images_dir).glob(f"{split}-*.parquet")):
        md = pq.ParquetFile(f).metadata
        for rg in range(md.num_row_groups):
            rows = md.row_group(rg).num_rows
            out.append((str(f), rg, rows))
            n += rows
            if limit and n >= limit:
                return out, n
    return out, n


class TaixFilms(torch.utils.data.IterableDataset):
    """(uid, [size, size] uint8 film) of the given row groups, split across DataLoader workers."""

    def __init__(self, tasks, size, via=None):
        self.tasks, self.size, self.via = tasks, size, via

    def __iter__(self):
        import pyarrow.parquet as pq
        from PIL import Image

        from .encoder import eval_u8

        torch.set_num_threads(1)
        info = torch.utils.data.get_worker_info()
        mine = self.tasks if info is None else self.tasks[info.id::info.num_workers]
        for path, rg, _ in mine:
            t = pq.ParquetFile(path).read_row_group(rg, columns=["UID", "Image"])
            for u, x in zip(t.column("UID").to_pylist(), t.column("Image").to_pylist()):
                yield u, torch.from_numpy(eval_u8(Image.open(io.BytesIO(x["bytes"])), self.size, self.via))


# RANZCR CLiP: label columns per device task (constants.yaml evaluator.device_tasks_ranzcr)
RANZCR_COLUMNS = {
    "airway_tube": ["ETT - Abnormal", "ETT - Borderline", "ETT - Normal"],
    "cvc_any": ["CVC - Abnormal", "CVC - Borderline", "CVC - Normal"],
    "enteric": ["NGT - Abnormal", "NGT - Borderline", "NGT - Incompletely Imaged", "NGT - Normal"],
}


def salted_fraction(key: str, salt: str) -> float:
    import hashlib

    return int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()[:12], 16) / 16 ** 12


def ranzcr_table(labels_csv: str, images_dir: str, val_fraction: float, salt: str):
    """One row per RANZCR CLiP train film: uid, patient, path, a 0/1 column per device task, split (dev | val by a
    salted hash of PatientID)."""
    import pandas as pd

    df = pd.read_csv(labels_csv)
    missing = [c for cols in RANZCR_COLUMNS.values() for c in cols + ["StudyInstanceUID", "PatientID"] if c not in df]
    if missing:
        raise ValueError(f"{labels_csv}: missing columns {sorted(set(missing))}")
    out = pd.DataFrame({"uid": df["StudyInstanceUID"].astype(str), "patient": df["PatientID"].astype(str)})
    out["path"] = [str(Path(images_dir) / f"{u}.jpg") for u in out["uid"]]
    for task, cols in RANZCR_COLUMNS.items():
        out[task] = (df[cols].sum(axis=1) > 0).astype(int)
    out["split"] = ["val" if salted_fraction(p, salt) < val_fraction else "dev" for p in out["patient"]]
    return out


class FileFilms(torch.utils.data.IterableDataset):
    """(uid, [size, size] uint8 film) of image files, split across DataLoader workers."""

    def __init__(self, items, size, via=None):
        self.items, self.size, self.via = list(items), size, via   # [(uid, path)]

    def __iter__(self):
        from PIL import Image

        from .encoder import eval_u8

        torch.set_num_threads(1)
        info = torch.utils.data.get_worker_info()
        mine = self.items if info is None else self.items[info.id::info.num_workers]
        for u, p in mine:
            with Image.open(p) as im:
                yield u, torch.from_numpy(eval_u8(im, self.size, self.via))
