"""Convert train_ground_truth.tsv into compact integer pairs and assign validation folds.

Output work/train/gt.parquet: s1_row, q_src, q_row (one row per true pair)
       work/train/folds.npy : fold id (0..4) per S1 row, from a hash of the S1 ID (deterministic)
Membership tests use sorted integer arrays (np.searchsorted), never Python sets of strings:
that was the cause of the original MemoryError.
"""
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import DATA, log, read_tsv, wpath


def id_index(split, src):
    """Sorted id_num array and matching row positions for fast ID -> row lookup."""
    num = pq.read_table(wpath(split, f"s{src}.parquet"), columns=["id_num"])["id_num"].to_numpy()
    order = np.argsort(num)
    return num[order], order


def lookup(index, nums):
    keys, rows = index
    pos = np.searchsorted(keys, nums).clip(0, len(keys) - 1)
    assert (keys[pos] == nums).all(), "ID not found"
    return rows[pos]


if __name__ == "__main__":
    dst = wpath("train", "gt.parquet")
    if os.path.exists(dst):
        log(f"skip {dst}"); raise SystemExit
    gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
    s1num = pc.cast(pc.utf8_slice_codeunits(gt["source1_entity_id"], 3), pa.int64()).to_numpy()
    s1row = lookup(id_index("train", 1), s1num)
    lists = pc.split_pattern(gt["matched_entity_ids"], ",")
    flat = pc.list_flatten(lists)
    parent = pc.list_parent_indices(lists).to_numpy()
    nonempty = pc.greater(pc.utf8_length(flat), 0).to_numpy(zero_copy_only=False)
    flat = flat.filter(pa.array(nonempty)); parent = parent[nonempty]
    src = pc.cast(pc.utf8_slice_codeunits(flat, 1, 2), pa.int8()).to_numpy()
    num = pc.cast(pc.utf8_slice_codeunits(flat, 3), pa.int64()).to_numpy()
    qrow = np.empty(len(num), np.int32)
    for s in (2, 3):
        m = src == s
        qrow[m] = lookup(id_index("train", s), num[m])
    pq.write_table(pa.table({"s1_row": s1row[parent].astype(np.int32), "q_src": src, "q_row": qrow}), dst)
    n1 = pq.read_metadata(wpath("train", "s1.parquet")).num_rows
    folds = np.empty(n1, np.int8)
    folds[s1row] = (s1num * 2654435761 % 2**32) % 5  # deterministic multiplicative hash of the ID
    np.save(wpath("train", "folds.npy"), folds)
    log(f"gt pairs={len(qrow)} S1={len(s1row)} fold sizes={np.bincount(folds).tolist()}")
