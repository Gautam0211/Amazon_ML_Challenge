"""Per-field IDF-cosine scores for every candidate pair (features; the rare-token signal per field).

For each field retriever of retrieve.py (name, skel, addr, cng3, num) builds the same per-country
IDF-weighted, L2-normalized token vectors as blocking (no DF cap: a score, not a generator) and takes
the row-wise dot product for each candidate pair. A missing field on either side gives NaN
(missing != mismatch), not 0.

Output: a new segment of the "ps" feature-store group holding only pairs not scored before.

Usage: python src/pairscore.py --split train --cand cand.parquet
"""
import argparse
import os

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp

from common import guard, log, wpath
from fstore import finish_segment, missing_pairs, new_segment
from metric import pair_key
from retrieve import tokens

FIELDS = ("name", "skel", "addr", "cng3", "num")


def normed(offs, flat, vocab, idf, idf_miss):
    """L2-normalized IDF rows over `vocab` (tokens outside vocab still count in the norm)."""
    pos = np.searchsorted(vocab, flat).clip(0, max(len(vocab) - 1, 0))
    found = vocab[pos] == flat if len(vocab) else np.zeros(len(flat), bool)
    w = np.where(found, idf[pos], idf_miss).astype(np.float32)
    row = np.repeat(np.arange(len(offs) - 1), np.diff(offs))
    norm = np.sqrt(np.bincount(row, w * w, minlength=len(offs) - 1)) + 1e-9
    keep = found
    indptr = np.r_[0, np.cumsum(np.bincount(row[keep], minlength=len(offs) - 1))]
    M = sp.csr_matrix(((w / norm[row])[keep], pos[keep], indptr), shape=(len(offs) - 1, max(len(vocab), 1)))
    return M, np.diff(offs) > 0


def rowdot(A, B, ia, ib, chunk=1_000_000):
    out = np.empty(len(ia), np.float32)
    for i in range(0, len(ia), chunk):
        out[i:i + chunk] = np.asarray(A[ia[i:i + chunk]].multiply(B[ib[i:i + chunk]]).sum(1)).ravel()
    return out


def run(split, cand):
    """Score the pairs of `cand` that the ps store does not hold yet (resumable per field)."""
    s1r, qs, qr = missing_pairs(split, cand, "ps")
    if not len(s1r):
        log("pairscore: all pairs cached"); return
    log(f"pairscore: scoring {len(s1r)} new pairs")
    tag = f"{len(s1r)}_{int(pair_key(s1r[:1], qs[:1], qr[:1])[0])}_{int(pair_key(s1r[-1:], qs[-1:], qr[-1:])[0])}"
    pend = wpath(split, "fstore", "ps", f"pending_{tag}", "x")[:-2]
    colf = lambda f: os.path.join(pend, f"{f}.npy")
    for f in FIELDS:
        if os.path.exists(colf(f)):
            continue
        res = np.full(len(s1r), np.nan, np.float32)
        c1, o1, h1 = tokens(split, 1, f)
        per = {}  # country -> (S1 matrix, has-field flags, vocab, idf, idf_miss, S1 row -> local index)
        for ctry in sorted(set(c1)):
            r1 = np.nonzero(c1 == ctry)[0]
            lens = o1[r1 + 1] - o1[r1]
            sf = h1[np.repeat(o1[r1], lens) + np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)]
            vocab, inv = np.unique(sf, return_inverse=True)
            df = np.bincount(inv, minlength=len(vocab)).astype(np.float32)
            idf = np.log((len(r1) + 1) / df).astype(np.float32)
            idf_miss = np.float32(np.log(len(r1) + 1))
            A, a_has = normed(np.r_[0, np.cumsum(lens)], sf, vocab, idf, idf_miss)
            del sf, inv, df
            loc1 = np.full(len(c1), -1, np.int32); loc1[r1] = np.arange(len(r1))
            per[ctry] = (A, a_has, vocab, idf, idf_miss, loc1)
        del c1, o1, h1
        for src in (2, 3):
            cq, oq, hq = tokens(split, src, f)
            del cq
            for ctry, (A, a_has, vocab, idf, idf_miss, loc1) in per.items():
                B, b_has = normed(oq, hq, vocab, idf, idf_miss)
                m = np.nonzero(qs == src)[0]
                m = m[loc1[s1r[m]] >= 0]
                ia, ib = loc1[s1r[m]], qr[m]
                v = rowdot(A, B, ia, ib)
                v[~(a_has[ia] & b_has[ib])] = np.nan
                res[m] = v
                del B, m, ia, ib, v
                guard("pairscore")
            del oq, hq
        del per
        np.save(colf(f) + ".tmp.npy", res); os.replace(colf(f) + ".tmp.npy", colf(f))
        log(f"  {f}: done, nan={np.isnan(res).mean():.3f} mean={np.nanmean(res):.3f}")
        del res
    cols = [f"ps_{f}" for f in FIELDS]
    d, M = new_segment(split, "ps", len(s1r), cols)
    for j, f in enumerate(FIELDS):
        M[:, j] = np.load(colf(f), mmap_mode="r")
    M.flush(); del M
    finish_segment(d, len(s1r), pair_key(s1r, qs, qr), cols)
    import shutil
    shutil.rmtree(pend)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--cand", default="cand.parquet")
    a = ap.parse_args()
    run(a.split, a.cand)
