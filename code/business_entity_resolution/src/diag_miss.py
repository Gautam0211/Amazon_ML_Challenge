"""Phase 0 diagnosis on dev folds 0-2: why are true pairs missed?

Blocking misses (true pair not in cand): each gets one cause, re-computing the blocking score
exactly as block.py does (same vocab, IDF, DF cap):
  zero_overlap   - no shared blocking token at all
  only_capped    - shared tokens exist but all have S1 df > cap (cannot create the pair)
  cutoff         - true score < rel * query's best score (dropped by the relative cutoff)
  not_topk       - survives the cutoff but ranks >= k
Plus the rank the true S1 would have with k unlimited (cap kept, cutoff off).

Classifier misses (true pair in cand, not predicted): lost_to_other_s1 / below_threshold /
lost_on_margin, with probability buckets. Uses cached OOF predictions (E000).

Usage: python src/diag_miss.py [--cand cand.parquet --oof oof.parquet]
Writes experiments/diag_<tag>.md and work/train/diag_miss.parquet
"""
import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

from block import load_tokens, rows_subset
from common import ROOT, log, wpath
from exp import load_gt
from metric import pair_key
from model import best_per_query, gt_keys, is_pos


def missed_pairs(cand_path, folds):
    g = load_gt()
    c = pq.read_table(wpath("train", cand_path), columns=["s1_row", "q_src", "q_row"])
    ck = np.sort(pair_key(c["s1_row"].to_numpy(), c["q_src"].to_numpy(), c["q_row"].to_numpy()))
    del c
    dev = folds[g["s1_row"]] <= 2
    g = {k: v[dev] for k, v in g.items()}
    found = is_pos(pair_key(g["s1_row"], g["q_src"], g["q_row"]), ck)
    return g, found


def blocking_causes(miss, cap, rel, k, cand_path):
    """Per missed pair: n shared tokens, n shared usable (df<=cap), true score, query best score, rank."""
    n = len(miss["s1_row"])
    out = {c: np.zeros(n, np.float32) for c in ("n_shared", "n_usable", "score", "best", "rank")}
    c1, o1, f1 = load_tokens("train", 1)
    cb = pq.read_table(wpath("train", cand_path), columns=["q_src", "q_row", "bscore"], filters=[("brank", "==", 0)])
    cb = {c: cb[c].to_numpy() for c in cb.column_names}
    qtok = {}
    for src in (2, 3):  # tokens of the missed queries only (full S2/S3 token lists do not fit next to S1)
        m = miss["q_src"] == src
        uq = np.unique(miss["q_row"][m])
        t = pq.read_table(wpath("train", f"s{src}.parquet"), columns=["btok"])["btok"].take(pa.array(uq)).combine_chunks()
        best_q = np.zeros(int(uq.max()) + 1, np.float32)
        mm = cb["q_src"] == src
        sel = cb["q_row"][mm] <= uq.max()
        best_q[cb["q_row"][mm][sel]] = cb["bscore"][mm][sel]
        qtok[src] = (uq, np.asarray(t.offsets), np.asarray(t.values), best_q)
    del cb
    for ctry in sorted(set(c1)):
        r1 = np.nonzero(c1 == ctry)[0]
        so, sf = rows_subset(o1, f1, r1)
        vocab, inv = np.unique(sf, return_inverse=True)
        del sf
        df = np.bincount(inv, minlength=len(vocab)).astype(np.float32)
        idf = np.log((len(r1) + 1) / df).astype(np.float32)
        A = sp.csr_matrix((idf[inv], inv, so), shape=(len(r1), len(vocab)))
        del inv, so
        anorm = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()) + 1e-9
        keep = (df <= cap).astype(np.float32)
        An = (sp.diags(1 / anorm) @ A @ sp.diags(keep)).tocsr()
        AT = An.T.tocsr()
        Arow = (A > 0).astype(np.float32).tocsr()
        del A
        idf_miss = np.float32(np.log(len(r1) + 1))
        for src in (2, 3):
            idx = np.nonzero((miss["q_src"] == src) & (c1[miss["s1_row"]] == ctry))[0]
            if not len(idx):
                continue
            uq_all, oq, fq, best_q = qtok[src]
            uq, qinv = np.unique(miss["q_row"][idx], return_inverse=True)
            qo, qf = rows_subset(oq, fq, np.searchsorted(uq_all, uq))
            pos = np.searchsorted(vocab, qf).clip(0, len(vocab) - 1)
            found = vocab[pos] == qf
            w = np.where(found, idf[pos], idf_miss)
            qrow = np.repeat(np.arange(len(uq)), np.diff(qo))
            qnorm = np.sqrt(np.bincount(qrow, w * w, minlength=len(uq))) + 1e-9
            B = sp.csr_matrix(((w / qnorm[qrow])[found], pos[found], np.r_[0, np.cumsum(np.bincount(qrow[found], minlength=len(uq)))]),
                              shape=(len(uq), len(vocab)))
            Bu = (B > 0).astype(np.float32)
            loc = np.searchsorted(r1, miss["s1_row"][idx])
            for s0 in range(0, len(idx), 50_000):
                sl = slice(s0, s0 + 50_000)
                qi, si = qinv[sl], loc[sl]
                out["n_shared"][idx[sl]] = np.asarray(Bu[qi].multiply(Arow[si]).sum(1)).ravel()
                out["n_usable"][idx[sl]] = np.asarray(Bu[qi].multiply(Arow[si] @ sp.diags(keep)).sum(1)).ravel()
                out["score"][idx[sl]] = np.asarray(B[qi].multiply(An[si]).sum(1)).ravel()
            out["best"][idx] = best_q[miss["q_row"][idx]]
            # rank with k unlimited (cap on, cutoff off). A query belongs to at most one S1 -> one missed pair per query.
            thr = np.zeros(len(uq), np.float32)
            np.maximum.at(thr, qinv, out["score"][idx])
            rank_q = np.zeros(len(uq), np.float32)
            step = 300
            for q0 in range(0, len(uq), step):
                C = (B[q0:q0 + step] @ AT).tocsr()
                rows = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
                rank_q[q0:q0 + step] = np.bincount(rows[C.data > thr[q0 + rows] + 1e-6], minlength=C.shape[0])
            out["rank"][idx] = rank_q[qinv]
            log(f"  s{src} {ctry}: missed pairs={len(idx)} queries={len(uq)}")
        del An, AT, Arow
    cause = np.where(out["n_shared"] == 0, "zero_overlap",
             np.where(out["n_usable"] == 0, "only_capped",
             np.where(out["score"] < rel * out["best"] - 1e-6, "cutoff", "not_topk")))
    return out, cause


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="cand.parquet")
    ap.add_argument("--oof", default="oof.parquet")
    ap.add_argument("--policy_t", type=float, default=0.6)
    ap.add_argument("--policy_margin", type=float, default=0.2)
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--rel", type=float, default=0.4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tag", default="E000")
    a = ap.parse_args()
    folds = np.load(wpath("train", "folds.npy"))
    g, found = missed_pairs(a.cand, folds)
    miss = {k: v[~found] for k, v in g.items()}
    log(f"dev true pairs={len(found)} found={found.mean():.4f} missed={len(miss['s1_row'])}")
    st, cause = blocking_causes(miss, a.cap, a.rel, a.k, a.cand)

    # slices: need record attributes
    recs = {s: pq.read_table(wpath("train", f"s{s}.parquet"), columns=["name_norm", "name_core", "addr_norm", "nonascii", "country"])
            for s in (1, 2, 3)}
    def qattr(d, col):
        out = np.empty(len(d["q_src"]), object)
        for s in (2, 3):
            m = d["q_src"] == s
            out[m] = recs[s][col].take(pa.array(d["q_row"][m])).to_numpy(zero_copy_only=False)
        return out
    def slices(d):
        s1n = recs[1]["name_norm"].take(pa.array(d["s1_row"])).to_numpy(zero_copy_only=False)
        s1c = recs[1]["name_core"].take(pa.array(d["s1_row"])).to_numpy(zero_copy_only=False)
        s1a = recs[1]["addr_norm"].take(pa.array(d["s1_row"])).to_numpy(zero_copy_only=False)
        qn, qc, qa = qattr(d, "name_norm"), qattr(d, "name_core"), qattr(d, "addr_norm")
        jac = lambda x, y: np.array([len(set(a.split()) & set(b.split())) / max(1, len(set(a.split()) | set(b.split()))) for a, b in zip(x, y)])
        cj, aj = jac(s1c, qc), jac(s1a, qa)
        return {
            "nonascii": qattr(d, "nonascii").astype(bool),
            "country": recs[1]["country"].take(pa.array(d["s1_row"])).to_numpy(zero_copy_only=False),
            "q_addr_empty": np.array([not x for x in qa]),
            "q_name_empty": np.array([not x for x in qc]),
            "exact_name": s1n == qn,
            "near_name": (cj >= 0.5) & (s1n != qn),
            "weak_name_strong_addr": (cj == 0) & (aj >= 0.5),
            "no_name_no_addr_overlap": (cj == 0) & (aj < 0.5),
            "src": np.where(d["q_src"] == 2, "S2", "S3"),
        }
    rng = np.random.default_rng(0)
    samp = rng.choice(len(found), 400_000, replace=False)  # slice rates for all dev pairs (sample)
    allsl = slices({k: v[samp] for k, v in g.items()})
    fs = found[samp]
    msl = slices(miss)

    L = [f"# Phase 0 diagnosis ({a.tag}), dev folds 0-2\n",
         f"Dev true pairs: {len(found):,}; in candidates: {found.mean():.4f}; missed by blocking: {len(cause):,}\n",
         "## Blocking miss causes\n", "| cause | pairs | share of misses | share of all true pairs |", "|---|---|---|---|"]
    for c in ("zero_overlap", "only_capped", "cutoff", "not_topk"):
        n = int((cause == c).sum())
        L.append(f"| {c} | {n:,} | {n / len(cause):.3f} | {n / len(found):.4f} |")
    r = st["rank"]
    nt = cause == "not_topk"
    L += ["", "Rank of true S1 (cap on, cutoff off, k unlimited) for `not_topk` misses:",
          "| rank bucket | pairs |", "|---|---|"]
    for lo, hi in ((10, 20), (20, 50), (50, 100), (100, 10**9)):
        L.append(f"| {lo}-{hi if hi < 10**9 else 'inf'} | {int((nt & (r >= lo) & (r < hi)).sum()):,} |")
    ct = cause == "cutoff"
    ratio = st["score"][ct] / np.maximum(st["best"][ct], 1e-9)
    L += ["", "True-score / best-score for `cutoff` misses:", "| ratio | pairs |", "|---|---|"]
    for lo, hi in ((0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (0, 0.1)):
        L.append(f"| {lo}-{hi} | {int(((ratio >= lo) & (ratio < hi)).sum()):,} |")
    L += ["", "## Slices: blocking recall and cause mix", "| slice | value | share of true pairs | blocking recall | misses | zero_overlap | only_capped | cutoff | not_topk |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name in allsl:
        vals = np.unique(allsl[name]) if allsl[name].dtype != bool else [True]
        for v in vals:
            m_all = allsl[name] == v
            m_miss = msl[name] == v
            mix = [f"{((cause == c) & m_miss).sum() / max(m_miss.sum(), 1):.2f}" for c in ("zero_overlap", "only_capped", "cutoff", "not_topk")]
            L.append(f"| {name} | {v} | {m_all.mean():.3f} | {fs[m_all].mean():.4f} | {int(m_miss.sum()):,} | " + " | ".join(mix) + " |")

    # classifier misses (E000 OOF)
    oof = pq.read_table(wpath("train", a.oof))
    s1, qs, qr, p = (oof[c].to_numpy() for c in ("s1_row", "q_src", "q_row", "p"))
    del oof
    bq = best_per_query(s1, qs, qr, p)
    del s1, qs, qr, p
    cf = {k: v[found] for k, v in g.items()}
    qk_b = (bq["q_src"].astype(np.int64) << 32) | bq["q_row"]
    o = np.argsort(qk_b); qk_b = qk_b[o]
    pos = o[np.searchsorted(qk_b, (cf["q_src"].astype(np.int64) << 32) | cf["q_row"])]
    best_is_true = bq["s1_row"][pos] == cf["s1_row"]
    pb, p2 = bq["p"][pos], bq["p2"][pos]
    pred = best_is_true & (pb >= a.policy_t) & (pb - p2 >= a.policy_margin)
    L += ["", f"## Classifier misses (true pair in candidates, policy t={a.policy_t} margin={a.policy_margin})",
          f"In-candidate true pairs: {len(pred):,}; predicted: {pred.mean():.4f}; missed: {(~pred).sum():,}\n",
          "| reason | pairs | share of all true pairs |", "|---|---|---|"]
    lost = ~best_is_true
    below = best_is_true & (pb < a.policy_t)
    marg = best_is_true & (pb >= a.policy_t) & (pb - p2 < a.policy_margin)
    for nm, m in (("lost_to_other_s1", lost), ("below_threshold", below), ("lost_on_margin", marg)):
        L.append(f"| {nm} | {int(m.sum()):,} | {m.sum() / len(found):.4f} |")
    L += ["", "below_threshold by best-probability bucket:", "| p | pairs |", "|---|---|"]
    for lo, hi in ((0.5, 0.6), (0.4, 0.5), (0.3, 0.4), (0.1, 0.3), (0, 0.1)):
        L.append(f"| {lo}-{hi} | {int((below & (pb >= lo) & (pb < hi)).sum()):,} |")
    path = os.path.join(ROOT, "experiments", f"diag_{a.tag}.md")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    pq.write_table(pa.table({**{k: v for k, v in miss.items()}, **st, "cause": cause.astype(str)}), wpath("train", "diag_miss.parquet"))
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
