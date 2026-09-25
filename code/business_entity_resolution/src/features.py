"""Stage 4: Feature Engineering for candidate pairs.

Each candidate pair (S1 record, S2/S3 record) gets string-similarity, number, missingness and
blocking-context features. Work is split into batches written as part files
(work/<split>/feat/part_00000.parquet ...): a crash resumes at the first missing part.

Usage: python src/features.py --split train [--batch 500000 --workers 6]
"""
import argparse
import glob
import os
import re
from multiprocessing import Pool

import numpy as np

_NUM = re.compile(r"\b\d+\b")


def _nums(s):
    return set(_NUM.findall(s))


def pair_feats(args):
    """Worker: string features for one batch. Inputs are aligned lists of strings."""
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    from rapidfuzz.process import cpdist

    (n1, c1, a1, k1, n2, c2, a2, k2) = args
    f = {}
    for nm, (x, y) in {"name": (n1, n2), "core": (c1, c2), "addr": (a1, a2)}.items():
        f[f"{nm}_ratio"] = cpdist(x, y, scorer=fuzz.ratio, workers=1)
        f[f"{nm}_tset"] = cpdist(x, y, scorer=fuzz.token_set_ratio, workers=1)
        f[f"{nm}_tsort"] = cpdist(x, y, scorer=fuzz.token_sort_ratio, workers=1)
        f[f"{nm}_partial"] = cpdist(x, y, scorer=fuzz.partial_ratio, workers=1)
    f["core_jw"] = cpdist(c1, c2, scorer=JaroWinkler.normalized_similarity, workers=1)
    f["skel_tset"] = cpdist(k1, k2, scorer=fuzz.token_set_ratio, workers=1)
    # core name of one side appearing inside the other side's full address/name (DBA, websites)
    f["core_nospace_ratio"] = cpdist([s.replace(" ", "") for s in c1], [s.replace(" ", "") for s in c2], scorer=fuzz.ratio, workers=1)
    num_j, num_first, num_n1, num_n2, tok_j, core_j, skel_j = [], [], [], [], [], [], []
    for i in range(len(a1)):
        s, t = _nums(a1[i]), _nums(a2[i])
        num_n1.append(len(s)); num_n2.append(len(t))
        num_j.append(len(s & t) / len(s | t) if s and t else -1.0)
        m1, m2 = _NUM.search(a1[i]), _NUM.search(a2[i])
        num_first.append(-1 if not (m1 and m2) else float(m1.group() == m2.group()))
        s, t = set(a1[i].split()), set(a2[i].split())
        tok_j.append(len(s & t) / len(s | t) if s and t else -1.0)
        s, t = set(c1[i].split()), set(c2[i].split())
        core_j.append(len(s & t) / len(s | t) if s and t else -1.0)
        s, t = set(k1[i].split()), set(k2[i].split())
        skel_j.append(len(s & t) / len(s | t) if s and t else -1.0)
    f.update(num_jacc=num_j, num_first_eq=num_first, num_n1=num_n1, num_n2=num_n2, addr_jacc=tok_j, core_jacc=core_j, skel_jacc=skel_j,
             core_len1=[len(s.split()) for s in c1], core_len2=[len(s.split()) for s in c2],
             addr_empty2=[not s for s in a2])
    return {k: np.asarray(v, np.float32) for k, v in f.items()}


class Context:
    """Blocking-context features. Keeps only compact per-query / per-S1 arrays in memory and
    expands them per batch (a full 93M x 9 float matrix would not fit in RAM)."""

    def __init__(self, cand):
        q_src, q_row = cand["q_src"], cand["q_row"]
        self.s1, self.sc, self.rk = cand["s1_row"], cand["bscore"], cand["brank"]
        # rows of one query are contiguous and sorted by rank (block.py writes them that way)
        qkey = (q_src.astype(np.int64) << 32) | q_row
        start = np.r_[0, np.nonzero(qkey[1:] != qkey[:-1])[0] + 1]
        del qkey
        cnt = np.diff(np.r_[start, len(self.sc)])
        self.gid = np.repeat(np.arange(len(start), dtype=np.int32), cnt)
        self.top1 = self.sc[start]
        self.second = np.where(cnt > 1, self.sc[np.minimum(start + 1, len(self.sc) - 1)], 0).astype(np.float32)
        self.qn = cnt.astype(np.float32)
        n1 = int(self.s1.max()) + 1
        self.s1_max = np.zeros(n1, np.float32); np.maximum.at(self.s1_max, self.s1, self.sc)
        self.s1_cnt = np.bincount(self.s1, minlength=n1).astype(np.float32)
        self.s1_top = np.bincount(self.s1, weights=(self.rk == 0), minlength=n1).astype(np.float32)  # queries where this S1 ranks first

    def batch(self, sl):
        s1, sc, g = self.s1[sl], self.sc[sl], self.gid[sl]
        return {"bscore": sc, "brank": self.rk[sl].astype(np.float32), "gap_top1": self.top1[g] - sc,
                "gap_second": sc - self.second[g], "q_ncand": self.qn[g],
                "s1_maxscore_gap": self.s1_max[s1] - sc, "s1_ncand": self.s1_cnt[s1], "s1_ntop": self.s1_top[s1]}


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from common import guard, log, wpath

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--cand", default="cand.parquet")
    ap.add_argument("--out", default="feat")
    ap.add_argument("--batch", type=int, default=250_000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    outdir = wpath(a.split, a.out, "x")[:-2]
    t = pq.read_table(wpath(a.split, a.cand))
    cand = {c: t[c].to_numpy() for c in t.column_names}  # one numpy copy; drop the Arrow table
    n_pairs = t.num_rows
    del t
    ctx = Context(cand)
    cols = ["name_norm", "name_core", "addr_norm", "name_skel"]
    recs = {s: pq.read_table(wpath(a.split, f"s{s}.parquet"), columns=cols + ["nonascii"]) for s in (1, 2, 3)}
    log(f"features {a.split}: pairs={n_pairs}")
    s1r, qs, qr = cand["s1_row"], cand["q_src"], cand["q_row"]
    nb = (n_pairs + a.batch - 1) // a.batch

    def job(b):
        sl = slice(b * a.batch, (b + 1) * a.batch)
        L = [recs[1][c].take(s1r[sl]).to_pylist() for c in cols]
        qsb, qrb = qs[sl], qr[sl]
        R = [[None] * len(qsb) for _ in cols]
        for s in (2, 3):
            m = np.nonzero(qsb == s)[0]
            for j, c in enumerate(cols):
                vals = recs[s][c].take(qrb[m]).to_pylist()
                for i, v in zip(m, vals):
                    R[j][i] = v
        return tuple(L + R)

    todo = [b for b in range(nb) if not os.path.exists(os.path.join(outdir, f"part_{b:05d}.parquet"))]
    log(f"  batches total={nb} todo={len(todo)}")
    with Pool(a.workers, maxtasksperchild=10) as pool:  # recycle workers: stops slow memory growth
        for w0 in range(0, len(todo), a.workers):
            guard("features")
            win = todo[w0:w0 + a.workers]
            for b, f in zip(win, pool.imap(pair_feats, [job(b) for b in win])):
                sl = slice(b * a.batch, (b + 1) * a.batch)
                nonasc = np.zeros(len(f["name_ratio"]), np.float32)
                for s in (2, 3):
                    m = qs[sl] == s
                    nonasc[m] = recs[s]["nonascii"].take(qr[sl][m]).to_numpy(zero_copy_only=False)
                t = pa.table({"s1_row": s1r[sl], "q_src": qs[sl], "q_row": qr[sl], "q_src_f": qs[sl].astype(np.float32),
                              **{f"ctx_{k}": v for k, v in ctx.batch(sl).items()}, **f, "nonascii2": nonasc})
                pq.write_table(t, os.path.join(outdir, f"part_{b:05d}.parquet.tmp"))
                os.replace(os.path.join(outdir, f"part_{b:05d}.parquet.tmp"), os.path.join(outdir, f"part_{b:05d}.parquet"))
            log(f"  done {min(w0 + a.workers, len(todo))}/{len(todo)}")
    log(f"features {a.split} complete: {len(glob.glob(os.path.join(outdir, 'part_*.parquet')))} parts")


if __name__ == "__main__":
    main()
