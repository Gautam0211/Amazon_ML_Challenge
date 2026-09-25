"""Phase 2/3: simulate adaptive-K / relative-cutoff policies offline.

A single generous blocking run on the query subsample (e.g. k=50, rel=0.2) contains every pair any
stricter per-query (k, rel) policy would keep, so policies are evaluated here without re-blocking:
dev-fold recall on sampled queries vs pairs per query.

Usage: python src/sim_block.py --sup cand_k50_r0.2_q0.05.parquet --qfrac 0.05
"""
import argparse

import numpy as np
import pyarrow.parquet as pq

from common import log, wpath
from metric import pair_key
from screen import sampled


def query_signals(c):
    """Per-query signals from the superset: best score, near-tie count, count above 0.4*best."""
    qk = (c["q_src"].astype(np.int64) << 32) | c["q_row"]
    start = np.r_[0, np.nonzero(qk[1:] != qk[:-1])[0] + 1]
    gid = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(qk)]))
    best = c["bscore"][start]
    rel = c["bscore"] / best[gid]
    ties = np.bincount(gid, rel >= 0.9)
    n04 = np.bincount(gid, rel >= 0.4)
    return gid, start, best, ties, n04, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sup", required=True)
    ap.add_argument("--qfrac", type=float, default=0.05)
    ap.add_argument("--tiers_only", action="store_true")
    a = ap.parse_args()
    t = pq.read_table(wpath("train", a.sup))
    c = {k: t[k].to_numpy() for k in t.column_names}
    del t
    folds = np.load(wpath("train", "folds.npy"))
    g = pq.read_table(wpath("train", "gt.parquet"))
    gs1, gqs, gqr = (g[k].to_numpy() for k in ("s1_row", "q_src", "q_row"))
    m = sampled(gqr, a.qfrac) & (folds[gs1] <= 2)
    gk = np.sort(pair_key(gs1[m], gqs[m], gqr[m]))
    ck = pair_key(c["s1_row"], c["q_src"], c["q_row"])
    pos = np.searchsorted(gk, ck).clip(0, len(gk) - 1)
    y = gk[pos] == ck
    nq = sum(sampled(np.arange(pq.read_metadata(wpath("train", f"s{s}.parquet")).num_rows), a.qfrac).sum() for s in (2, 3))
    gid, start, best, ties, n04, rel = query_signals(c)
    # query attributes
    attr = {}
    for col in ("nonascii", "addr_norm"):
        v = np.zeros(len(start), bool)
        for s in (2, 3):
            ms = c["q_src"][start] == s
            col_arr = pq.read_table(wpath("train", f"s{s}.parquet"), columns=[col])[col]
            x = col_arr.take(c["q_row"][start][ms]).to_numpy(zero_copy_only=False)
            v[ms] = x.astype(bool) if col == "nonascii" else np.array([not s_ for s_ in x])
        attr[col] = v
    na, ae = attr["nonascii"], attr["addr_norm"]  # ae = address empty
    rk = c["brank"].astype(np.int32)
    total_true = len(gk)

    def ev(name, kq, rq):
        keep = (rk < kq[gid]) & (rel >= rq[gid] - 1e-6)
        log(f"{name:58s} recall={y[keep].sum() / total_true:.4f} pairs/q={keep.sum() / nq:.2f}")

    Q = len(start)
    K = lambda v: np.full(Q, v, np.int32)
    R = lambda v: np.full(Q, v, np.float32)
    log(f"superset {a.sup}: queries with cands={Q} sampled queries={nq} dev true pairs={total_true} superset recall={y.sum() / total_true:.4f}")
    ev("baseline k10 r0.4", K(10), R(0.4))
    for k, r in () if a.tiers_only else ((10, 0.3), (10, 0.2), (15, 0.4), (20, 0.4), (20, 0.3), (30, 0.3), (50, 0.2)):
        ev(f"global k{k} r{r}", K(k), R(r))
    hard_sets = {} if a.tiers_only else {"addr_empty": ae, "nonascii": na, "ae|na": ae | na,
                 "ties>=5": ties >= 5, "ties>=10": ties >= 10, "n04>=10": n04 >= 10,
                 "ae|na|ties>=5": ae | na | (ties >= 5), "best<0.3": best < 0.3}
    for hn, h in hard_sets.items():
        for kh, rh in ((20, 0.4), (30, 0.3), (50, 0.3), (50, 0.2)):
            ev(f"adaptive hard={hn} ({h.mean():.2f}) -> k{kh} r{rh}", np.where(h, kh, 10).astype(np.int32), np.where(h, rh, 0.4).astype(np.float32))
    # tiered: clear 10 / ambiguous 20 / very ambiguous 50
    tiers = {"A": ((ties >= 5) | na, ae | (ties >= 10)),
             "B": ((ties >= 3) | na, ae | (ties >= 8)),
             "C": ((ties >= 5) | na | (best < 0.3), ae | (ties >= 10)),
             "D": ((ties >= 3) | na | (best < 0.3), ae | (ties >= 5))}
    for tn, (amb, very) in tiers.items():
        for k1, k2, k3, r in ((10, 20, 50, 0.3), (10, 15, 50, 0.3), (10, 20, 40, 0.3), (10, 25, 50, 0.3)):
            kq = np.where(very, k3, np.where(amb, k2, k1)).astype(np.int32)
            rq = np.where(very | amb, r, 0.4).astype(np.float32)
            ev(f"tiered{tn} {k1}/{k2}/{k3} r{r} (amb {amb.mean():.2f}, very {very.mean():.2f})", kq, rq)


if __name__ == "__main__":
    main()
