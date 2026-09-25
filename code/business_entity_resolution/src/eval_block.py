"""Blocking diagnostics on train: candidate recall vs k, per country, pairs per S1, reduction ratio.

Usage: python src/eval_block.py [--cand cand.parquet]
"""
import argparse

import numpy as np
import pyarrow.parquet as pq

from common import log, wpath
from metric import pair_key

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="cand.parquet")
    a = ap.parse_args()
    c = pq.read_table(wpath("train", a.cand))
    g = pq.read_table(wpath("train", "gt.parquet"))
    ctry = pq.read_table(wpath("train", "s1.parquet"), columns=["country"])["country"].to_numpy(zero_copy_only=False).astype(str)
    ck = pair_key(c["s1_row"].to_numpy(), c["q_src"].to_numpy(), c["q_row"].to_numpy())
    gk = pair_key(g["s1_row"].to_numpy(), g["q_src"].to_numpy(), g["q_row"].to_numpy())
    rank = c["brank"].to_numpy()
    order = np.argsort(ck); ck, rank = ck[order], rank[order]
    pos = np.searchsorted(ck, gk).clip(0, len(ck) - 1)
    found = ck[pos] == gk
    grank = np.where(found, rank[pos], 99)
    n1 = len(ctry); nq = 0
    for s in (2, 3):
        nq += pq.read_metadata(wpath("train", f"s{s}.parquet")).num_rows
    log(f"pairs={len(ck)} pairs/S1={len(ck)/n1:.1f} reduction_ratio={1 - len(ck) / (n1 * nq):.8f}")
    for k in (1, 2, 3, 5, 10):
        log(f"  recall@{k}: {(grank < k).mean():.4f}")
    gc = ctry[g["s1_row"].to_numpy()]
    for cc in sorted(set(ctry)):
        m = gc == cc
        log(f"  {cc}: recall(all k)={found[m].mean():.4f}  S2={found[m & (g['q_src'].to_numpy() == 2)].mean():.4f} "
            f"S3={found[m & (g['q_src'].to_numpy() == 3)].mean():.4f}")
    np.save(wpath("train", "gt_found.npy"), found)
