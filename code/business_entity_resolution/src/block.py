"""Stage 3: Blocking (candidate generation).

For every S2/S3 record ("query") find the K most similar S1 records using an IDF-weighted
cosine over hashed blocking tokens (name words, name concatenation, phonetic skeletons,
address words/numbers). Runs per country label found in the data (no hard-coded countries),
so candidates never cross countries.

Memory safety:
  * vocabulary = S1 tokens of one country only (a query token absent from S1 cannot create a pair)
  * tokens whose S1 document frequency exceeds --cap are not used to generate pairs
    (they still count in the cosine norm), so a common word cannot explode a block
  * queries are processed in adaptive chunks whose worst-case product size is bounded
    by --budget non-zeros; the budget halves automatically when free RAM runs low

Output: work/<split>/cand.parquet with columns
  q_src (2/3), q_row (row in s<src>.parquet), s1_row, bscore (cosine), brank (0 = best)

Usage: python src/block.py --split train [--k 10 --cap 2000]
"""
import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

from common import free_gb, guard, log, wpath


def load_tokens(split, src):
    t = pq.read_table(wpath(split, f"s{src}.parquet"), columns=["country", "btok"])
    lst = t["btok"].combine_chunks()
    return (t["country"].to_numpy(zero_copy_only=False).astype(str),
            np.asarray(lst.offsets), np.asarray(lst.values))


def rows_subset(offs, flat, rows):
    """Token lists (offsets, values) for a subset of rows."""
    lens = offs[rows + 1] - offs[rows]
    o = np.zeros(len(rows) + 1, np.int64); np.cumsum(lens, out=o[1:])
    idx = np.repeat(offs[rows] - o[:-1], lens) + np.arange(o[-1])
    return o, flat[idx]


def topk_rows(C, k, rel=0.25):
    """Per-row top-k of a CSR matrix -> (row, col, val) arrays, sorted by row then score desc."""
    n = C.shape[0]
    rows = np.repeat(np.arange(n), np.diff(C.indptr))
    if C.nnz == 0:
        return rows, C.indices, C.data, rows
    # cheap pre-filter: entries far below the row maximum can never reach the top-k
    nz = np.diff(C.indptr) > 0
    rmax = np.zeros(n, C.data.dtype)
    rmax[nz] = np.maximum.reduceat(C.data, C.indptr[:-1][nz])
    m = C.data >= rel * rmax[rows]
    rows, cols, vals = rows[m], C.indices[m], C.data[m]
    order = np.lexsort((-vals, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    first = np.r_[0, np.nonzero(np.diff(rows))[0] + 1]
    rank = np.arange(len(rows)) - np.repeat(first, np.diff(np.r_[first, len(rows)]))
    m = rank < k
    return rows[m], cols[m], vals[m], rank[m]


def block_country(ctry, s1, qs, k, cap, budget, writer, rel=0.25, qfrac=1.0):
    c1, o1, f1 = s1
    r1 = np.nonzero(c1 == ctry)[0]
    so, sf = rows_subset(o1, f1, r1)
    vocab, inv = np.unique(sf, return_inverse=True)
    df = np.bincount(inv, minlength=len(vocab)).astype(np.float32)
    idf = np.log((len(r1) + 1) / df).astype(np.float32)
    A = sp.csr_matrix((idf[inv], inv, so), shape=(len(r1), len(vocab)))
    anorm = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()) + 1e-9
    keep = (df <= cap).astype(np.float32)  # tokens allowed to generate pairs
    AT = (sp.diags(1 / anorm) @ A @ sp.diags(keep)).T.tocsr()  # vocab x S1, cosine-ready
    idf_miss = np.float32(np.log(len(r1) + 1))
    log(f"  [{ctry}] S1={len(r1)} vocab={len(vocab)} capped_tokens={(df > cap).sum()}")
    total, hit_rows = 0, 0
    for src, (cq, oq, fq) in qs.items():
        rq = np.nonzero(cq == ctry)[0]
        if qfrac < 1:  # diagnostic subsample of queries (deterministic)
            rq = rq[(rq * 2654435761 % 1000) < qfrac * 1000]
        qo, qf = rows_subset(oq, fq, rq)
        pos = np.searchsorted(vocab, qf).clip(0, len(vocab) - 1)
        found = vocab[pos] == qf
        w = np.where(found, idf[pos], idf_miss)
        qrow = np.repeat(np.arange(len(rq)), np.diff(qo))
        qnorm = np.sqrt(np.bincount(qrow, w * w, minlength=len(rq))) + 1e-9
        # worst-case product size per query row = sum of capped df of its tokens
        cost = np.bincount(qrow[found], (df * keep)[pos[found]], minlength=len(rq))
        B = sp.csr_matrix(((w / qnorm[qrow])[found], pos[found], np.r_[0, np.cumsum(np.bincount(qrow[found], minlength=len(rq)))]),
                          shape=(len(rq), len(vocab)))
        cc = np.cumsum(cost)
        start = nchunk = 0
        while start < len(rq):
            guard(f"block {ctry} s{src}")
            if free_gb() < 3:
                budget = max(2_000_000, budget // 2)
            base = cc[start - 1] if start else 0.0
            end = max(start + 1, int(np.searchsorted(cc, base + budget)))
            C = (B[start:end] @ AT).tocsr()
            qr, s1c, val, rk = topk_rows(C, k, rel)
            writer.write_table(pa.table({
                "q_src": np.full(len(qr), src, np.int8), "q_row": rq[start + qr].astype(np.int32),
                "s1_row": r1[s1c].astype(np.int32), "bscore": val.astype(np.float32), "brank": rk.astype(np.int8)}))
            total += len(qr); hit_rows += len(np.unique(qr))
            nchunk += 1
            if nchunk % 25 == 0:
                log(f"    [{ctry}] s{src}: {end}/{len(rq)} queries, pairs={total}, budget={budget}")
            start = end
        log(f"  [{ctry}] s{src}: queries={len(rq)} with_candidates={hit_rows} pairs_so_far={total}")
        hit_rows = 0
    return budget


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--budget", type=int, default=20_000_000)
    ap.add_argument("--out", default="cand.parquet")
    ap.add_argument("--rel", type=float, default=0.4)
    ap.add_argument("--qfrac", type=float, default=1.0)
    a = ap.parse_args()
    dst = wpath(a.split, a.out)
    if os.path.exists(dst):
        log(f"skip {dst} (exists)")
        raise SystemExit
    s1 = load_tokens(a.split, 1)
    qs = {2: load_tokens(a.split, 2), 3: load_tokens(a.split, 3)}
    log(f"blocking {a.split}: k={a.k} cap={a.cap}")
    schema = pa.schema([("q_src", pa.int8()), ("q_row", pa.int32()), ("s1_row", pa.int32()), ("bscore", pa.float32()), ("brank", pa.int8())])
    w = pq.ParquetWriter(dst + ".tmp", schema, compression="zstd")
    budget = a.budget
    for ctry in sorted(set(s1[0])):
        budget = block_country(ctry, s1, qs, a.k, a.cap, budget, w, a.rel, a.qfrac)
    w.close()
    os.replace(dst + ".tmp", dst)
    log(f"wrote {dst}")
