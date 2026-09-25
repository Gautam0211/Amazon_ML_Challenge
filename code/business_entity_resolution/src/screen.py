"""Cheap blocking screen on the deterministic query subsample (same hash as block.py --qfrac).

For each candidate file (and the union of all given files, and the union with the main cand)
reports dev-fold (0-2) recall on sampled queries, pairs per query and the gain over the main file.

Usage: python src/screen.py --qfrac 0.02 ret_name_k10_c5000_r0.4_q0.02.parquet ...
"""
import argparse

import numpy as np
import pyarrow.parquet as pq

from common import log, wpath
from metric import pair_key


def sampled(q_row, qfrac):
    return (q_row.astype(np.int64) * 2654435761 % 1000) < qfrac * 1000


def keys(name, qfrac, kmax=99):
    t = pq.read_table(wpath("train", name), columns=["s1_row", "q_src", "q_row", "brank"])
    s1, qs, qr = (t[c].to_numpy() for c in ("s1_row", "q_src", "q_row"))
    m = sampled(qr, qfrac) & (t["brank"].to_numpy() < kmax)
    return np.unique(pair_key(s1[m], qs[m], qr[m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qfrac", type=float, default=0.02)
    ap.add_argument("--main", default="cand.parquet")
    ap.add_argument("--kmax", default="99", help="comma list: per-retriever rank limits to try")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    folds = np.load(wpath("train", "folds.npy"))
    g = pq.read_table(wpath("train", "gt.parquet"))
    gs1, gqs, gqr = (g[c].to_numpy() for c in ("s1_row", "q_src", "q_row"))
    m = sampled(gqr, a.qfrac) & (folds[gs1] <= 2)
    gk = pair_key(gs1[m], gqs[m], gqr[m])
    nq = sum(sampled(np.arange(pq.read_metadata(wpath("train", f"s{s}.parquet")).num_rows), a.qfrac).sum() for s in (2, 3))
    main = keys(a.main, a.qfrac)
    base = np.isin(gk, main)
    log(f"sample queries={nq} dev true pairs={len(gk)} | main {a.main}: recall={base.mean():.4f} pairs/q={len(main) / nq:.2f}")
    for km in map(int, a.kmax.split(",")):
        allk = main
        for f in a.files:
            k = keys(f, a.qfrac, km)
            hit = np.isin(gk, k)
            u = np.union1d(main, k)
            uh = np.isin(gk, u)
            log(f"  k<{km} {f}: alone recall={hit.mean():.4f} pairs/q={len(k) / nq:.2f} | union w/ main recall={uh.mean():.4f} "
                f"(+{uh.mean() - base.mean():.4f}) pairs/q={len(u) / nq:.2f} (x{len(u) / len(main):.2f})")
            allk = np.union1d(allk, k)
        if len(a.files) > 1:
            uh = np.isin(gk, allk)
            log(f"  k<{km} UNION all: recall={uh.mean():.4f} (+{uh.mean() - base.mean():.4f}) pairs/q={len(allk) / nq:.2f} (x{len(allk) / len(main):.2f})")


if __name__ == "__main__":
    main()
