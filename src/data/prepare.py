"""Data preparation pipeline.

Steps
-----
1. Extract the two provided archives (``georges.zip`` -> positive,
   ``non_georges.zip`` -> negative).
2. Validate every image (decode + size); record corrupt files.
3. Compute a perceptual hash (dHash) for each image and cluster near-duplicates
   with union-find. Cross-class near-duplicate clusters are *contaminated* and
   dropped entirely (they would corrupt supervision and cause leakage).
4. Split by cluster (leakage-safe, stratified) into train/val/test.
5. Emit a data sheet (``reports/EDA.md``) and machine-readable stats JSON.

Only legal, ethical operations are performed: no external data is fetched and
no labels are altered beyond dropping ambiguous/contaminated samples.
"""
from __future__ import annotations

import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..common import (DATA_RAW, DATA_SPLITS, REPORTS_DIR, ensure_dirs,
                      POS_LABEL, NEG_LABEL)
from .split import make_splits, write_splits


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def extract_archives(pos_zip: Path, neg_zip: Path, out_dir: Path,
                     force: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for zip_path, sub in ((pos_zip, "georges"), (neg_zip, "non_georges")):
        dest = out_dir / sub
        if dest.exists() and any(dest.iterdir()) and not force:
            print(f"[prepare] {sub} already extracted, skipping.")
            continue
        print(f"[prepare] extracting {zip_path.name} -> {dest}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest)
        # Some archives nest an extra top-level folder; flatten one level.
        nested = [p for p in dest.iterdir() if p.is_dir()]
        if len(nested) == 1 and not any(dest.glob("*.jpg")):
            for f in nested[0].iterdir():
                f.rename(dest / f.name)
            try:
                nested[0].rmdir()
            except OSError:
                pass


def compute_dhash(path: str, hash_size: int = 8) -> int:
    """Difference hash (dHash) as a 64-bit integer."""
    with Image.open(path) as im:
        im = im.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    val = 0
    for b in diff.flatten():
        val = (val << 1) | int(b)
    return val


def scan_images(root: Path, sub: str, label: int, hash_size: int) -> tuple[list[dict], list[dict]]:
    """Single pass: validate + size + dHash. Returns (records, corrupt)."""
    records, corrupt = [], []
    files = sorted(root.glob(f"{sub}/*.*"))
    total = len(files)
    for i, f in enumerate(files, 1):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        try:
            with Image.open(f) as im:
                im.verify()
            with Image.open(f) as im:
                w, h = im.size
                dhash = compute_dhash(str(f), hash_size)
            records.append({
                "path": str(f), "label": label, "w": w, "h": h, "dhash": dhash,
            })
        except Exception as e:  # corrupt / unreadable
            corrupt.append({"path": str(f), "error": str(e)[:120]})
        if i % 500 == 0:
            print(f"[prepare] scanned {i}/{total} in {sub}")
    return records, corrupt


def cluster_and_clean(records: list[dict], dup_hamming: int, stats: dict) -> list[dict]:
    """Cluster near-duplicates; drop cross-class (contaminated) clusters."""
    n = len(records)
    hashes = [r["dhash"] for r in records]
    uf = _UnionFind(n)
    for i in range(n):
        hi = hashes[i]
        for j in range(i + 1, n):
            if (hi ^ hashes[j]).bit_count() <= dup_hamming:
                uf.union(i, j)

    roots: dict[int, int] = {}
    cid_map: dict[int, int] = {}
    nxt = 0
    for i in range(n):
        r = uf.find(i)
        if r not in roots:
            roots[r] = nxt
            nxt += 1
        cid_map[i] = roots[r]
    for i, rec in enumerate(records):
        rec["cluster_id"] = str(cid_map[i])

    # cluster -> labels
    cluster_labels: dict[str, set] = defaultdict(set)
    cluster_sizes: dict[str, int] = defaultdict(int)
    for rec in records:
        cluster_labels[rec["cluster_id"]].add(rec["label"])
        cluster_sizes[rec["cluster_id"]] += 1

    contaminated = {cid for cid, labs in cluster_labels.items() if len(labs) > 1}
    dup_clusters = sum(1 for s in cluster_sizes.values() if s > 1)
    removed = sum(cluster_sizes[c] for c in contaminated)

    clean = [r for r in records if r["cluster_id"] not in contaminated]

    # write duplicate report
    with open(DATA_SPLITS.parent / "duplicates.txt", "w", encoding="utf-8") as fh:
        fh.write("# near-duplicate / cross-class leakage report\n")
        fh.write(f"# total images scanned: {n}\n")
        fh.write(f"# duplicate clusters (size>1): {dup_clusters}\n")
        fh.write(f"# cross-class contaminated clusters (dropped): {len(contaminated)}\n")
        fh.write(f"# images dropped due to contamination: {removed}\n\n")
        for cid, labs in sorted(cluster_labels.items(), key=lambda x: -cluster_sizes[x[0]]):
            flag = "CROSS-CLASS" if cid in contaminated else "intra-class"
            fh.write(f"cluster={cid} size={cluster_sizes[cid]} labels={sorted(labs)} {flag}\n")

    stats["n_clusters"] = len(cluster_sizes)
    stats["n_dup_clusters"] = dup_clusters
    stats["n_contaminated_clusters"] = len(contaminated)
    stats["n_dropped_contaminated"] = removed
    return clean


def write_eda(records: list[dict], corrupt: list[dict], stats: dict) -> None:
    widths = [r["w"] for r in records] + [r["h"] for r in records]
    widths_sorted = sorted(widths)
    median = float(np.median(widths_sorted)) if widths_sorted else 0.0

    def pct(x):
        return round(100.0 * x / max(1, stats["n_valid"]), 1)

    lines = [
        "# Data Exploration (EDA)",
        "",
        f"- Total images scanned: **{stats['n_scanned']}**",
        f"- Valid images kept: **{stats['n_valid']}**",
        f"  - positive (Saint George): {stats['n_pos']} ({pct(stats['n_pos'])}%)",
        f"  - negative (no Saint George): {stats['n_neg']} ({pct(stats['n_neg'])}%)",
        f"- Corrupt / unreadable: **{len(corrupt)}** (excluded, see corrupt.txt)",
        f"- Near-duplicate clusters: **{stats['n_dup_clusters']}**",
        f"- Cross-class contaminated clusters (dropped): **{stats['n_contaminated_clusters']}** "
        f"({stats['n_dropped_contaminated']} images removed)",
        f"- Image width/height median: **{int(median)} px** "
        f"(min {min(widths_sorted) if widths_sorted else 0}, "
        f"max {max(widths_sorted) if widths_sorted else 0})",
        "",
        "## Split sizes (leakage-safe, cluster-grouped, stratified)",
        f"- train: {stats['n_train']}",
        f"- val:   {stats['n_val']}",
        f"- test:  {stats['n_test']}",
        "",
        "## Notes",
        "- Saint George appears in many forms (paintings, sculptures, badges, "
        "stained glass); negatives may contain other saints, knights, dragons "
        "or churches — a hard semantic distinction.",
        "- Cross-class near-duplicates are dropped because they ambiguously "
        "label the same visual content as both classes.",
        "- Splits group near-duplicate clusters together to prevent leakage.",
    ]
    (REPORTS_DIR / "EDA.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_all(cfg: dict, pos_zip: str | Path, neg_zip: str | Path,
                force: bool = False) -> list[dict]:
    """Run the full preparation pipeline. Returns the clean record list."""
    ensure_dirs()
    t0 = time.time()
    hash_size = cfg["data"]["dhash_size"]
    dup_hamming = cfg["data"]["dup_hamming"]

    extract_archives(Path(pos_zip), Path(neg_zip), DATA_RAW, force)

    pos, corr_pos = scan_images(DATA_RAW, "georges", POS_LABEL, hash_size)
    neg, corr_neg = scan_images(DATA_RAW, "non_georges", NEG_LABEL, hash_size)
    records = pos + neg
    corrupt = corr_pos + corr_neg

    stats = {
        "n_scanned": len(records) + len(corrupt),
        "n_pos": len(pos), "n_neg": len(neg),
        "n_valid": len(records),
    }

    clean = cluster_and_clean(records, dup_hamming, stats)
    make_splits(clean, cfg["data"]["train_frac"], cfg["data"]["val_frac"], cfg["seed"])
    write_splits(clean, DATA_SPLITS)

    # corrupt log
    with open(DATA_SPLITS.parent / "corrupt.txt", "w", encoding="utf-8") as fh:
        for c in corrupt:
            fh.write(f"{c['path']}\t{c['error']}\n")

    for sp in ("train", "val", "test"):
        stats[f"n_{sp}"] = sum(1 for r in clean if r["split"] == sp)
    stats["n_corrupt"] = len(corrupt)
    stats["elapsed_sec"] = round(time.time() - t0, 1)

    write_eda(clean, corrupt, stats)
    (REPORTS_DIR / "eda_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[prepare] done in {stats['elapsed_sec']}s | clean={len(clean)} "
          f"train/val/test={stats['n_train']}/{stats['n_val']}/{stats['n_test']}")
    return clean
