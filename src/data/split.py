"""Leakage-safe dataset splitting.

Images are grouped by *near-duplicate cluster* (see ``src.data.prepare``) and
the whole cluster is assigned to a single split. This guarantees that no
near-duplicate (or exact duplicate) of a training image leaks into the
validation or test set, which would otherwise inflate reported metrics.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path


def make_splits(records: list[dict], train_frac: float = 0.8,
                val_frac: float = 0.1, seed: int = 42) -> list[dict]:
    """Assign each record a split, keeping clusters together and stratified
    by class. Mutates and returns ``records`` (adds ``split`` key)."""
    clusters: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        clusters[r["cluster_id"]].append(r)

    rng = random.Random(seed)
    assign: dict[str, str] = {}
    for label in (0, 1):
        cids = [cid for cid, rs in clusters.items() if rs[0]["label"] == label]
        rng.shuffle(cids)
        n = len(cids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        for cid in cids[:n_train]:
            assign[cid] = "train"
        for cid in cids[n_train:n_train + n_val]:
            assign[cid] = "val"
        for cid in cids[n_train + n_val:]:
            assign[cid] = "test"

    for r in records:
        r["split"] = assign[r["cluster_id"]]
    return records


def write_splits(records: list[dict], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / f"{k}.csv" for k in ("train", "val", "test")}
    handles = {k: open(v, "w", newline="", encoding="utf-8") for k, v in paths.items()}
    writers = {k: csv.writer(h) for k, h in handles.items()}
    for w in writers.values():
        w.writerow(["path", "label", "cluster_id"])
    for r in records:
        writers[r["split"]].writerow([r["path"], r["label"], r["cluster_id"]])
    for h in handles.values():
        h.close()
    return paths


def load_split(path: str | Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "path": row["path"],
                "label": int(row["label"]),
                "cluster_id": row["cluster_id"],
            })
    return rows
