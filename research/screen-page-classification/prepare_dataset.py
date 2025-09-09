from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import sqlite3
from PIL import Image

# This exporter reads directly from the annotation SQLite database used by tools/annotation.
# It creates a directory with images and three jsonl files for train/val/test splits.


def read_labeled_rows(db_path: str, dataset_id: int) -> List[Tuple[str, int, str]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT a.id as ann_id,
               a.class_id as class_id,
               f.path_rel as frame_path_rel
        FROM annotations a
        JOIN frames f ON f.id = a.frame_id
        WHERE a.dataset_id = ? AND a.status = 'labeled'
        ORDER BY a.id
        """,
        (dataset_id,),
    ).fetchall()
    conn.close()
    out: List[Tuple[str, int, str]] = []
    for r in rows or []:
        out.append((str(r["ann_id"]), int(r["class_id"]), str(r["frame_path_rel"])) )
    return out


def copy_image(root_dir: Path, source_rel: str, out_dir: Path, out_name: str) -> str:
    src = root_dir / source_rel
    dst = out_dir / f"{out_name}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.convert("RGB").save(dst, format="JPEG", quality=95)
    return str(dst.relative_to(out_dir.parent))


def split_indices(n: int, split: Tuple[float, float, float]) -> Tuple[List[int], List[int], List[int]]:
    import random

    idxs = list(range(n))
    random.shuffle(idxs)
    n_train = int(split[0] * n)
    n_val = int(split[1] * n)
    train = idxs[:n_train]
    val = idxs[n_train : n_train + n_val]
    test = idxs[n_train + n_val :]
    return train, val, test


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True, help="Output dataset root directory")
    p.add_argument("--dataset_id", type=int, required=True)
    p.add_argument("--db_path", default=os.environ.get("ANNOTATION_DB", "tools/annotation/annotation.db"))
    p.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "data"), help="Root containing frames")
    p.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    args = p.parse_args()

    out_root = Path(args.output_dir)
    images_dir = out_root / "images"
    splits_dir = out_root
    out_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = read_labeled_rows(args.db_path, int(args.dataset_id))
    if not rows:
        raise SystemExit("No labeled rows found. Ensure the dataset is labeled and the DB path is correct.")

    train_idx, val_idx, test_idx = split_indices(len(rows), tuple(args.split))

    def dump_split(name: str, indices: List[int]) -> None:
        with open(splits_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for i in indices:
                ann_id, class_id, frame_rel = rows[i]
                out_rel = copy_image(Path(args.data_root), frame_rel, images_dir, out_name=str(ann_id))
                f.write(json.dumps({"image": out_rel, "label": int(class_id)}) + "\n")

    dump_split("train", train_idx)
    dump_split("val", val_idx)
    dump_split("test", test_idx)
    print(f"Wrote dataset to {out_root}")


if __name__ == "__main__":
    main()

