"""Pair feature cache: string features are computed once per (S1, S2/S3) pair and reused by every
experiment whose candidate set contains that pair. Only context features (which depend on the
whole candidate set) are rebuilt per experiment.

Groups: "str" (string similarities, features.pair_feats) and "ps" (per-field IDF cosines, pairscore.py).
Layout work/<split>/fstore/<group>/seg_NNN/: keys.npy (sorted pair keys), rows.npy (row of each sorted key
in X), X.f32 (float32 memmap, n x len(STR_COLS)). Segment 0 is imported from the v2 feature parts;
later segments hold only pairs that no earlier segment had.

Usage: python src/fstore.py --split train --init          (import v2 parts as segment 0)
       python src/fstore.py --split train --cand cand_x.parquet  (featurize missing pairs)
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

from features import pair_feats

STR_COLS = ["name_ratio", "name_tset", "name_tsort", "name_partial", "core_ratio", "core_tset", "core_tsort", "core_partial",
            "addr_ratio", "addr_tset", "addr_tsort", "addr_partial", "core_jw", "skel_tset", "core_nospace_ratio", "num_jacc",
            "num_first_eq", "num_n1", "num_n2", "addr_jacc", "core_jacc", "skel_jacc", "core_len1", "core_len2", "addr_empty2", "nonascii2"]


def seg_dirs(split, group="str"):
    from common import wpath
    # a segment counts only once meta.json exists (written last); incomplete dirs are replaced
    d = sorted(glob.glob(os.path.join(wpath(split, "fstore", group, "x")[:-2], "seg_*")))
    return [x for x in d if os.path.exists(os.path.join(x, "meta.json"))]


class Store:
    def __init__(self, split, group="str"):
        self.segs, self.cols = [], STR_COLS
        for d in seg_dirs(split, group):
            meta = json.load(open(os.path.join(d, "meta.json")))
            n, self.cols = meta["n"], meta["cols"]
            self.segs.append((np.load(os.path.join(d, "keys.npy")), np.load(os.path.join(d, "rows.npy")),
                              (os.path.join(d, "X.f32"), (n, len(self.cols)))))

    def locate(self, keys, chunk=4_000_000):
        """(segment id, row) per key; segment -1 = not stored. Chunked: bounded temporaries."""
        seg = np.full(len(keys), -1, np.int8); row = np.zeros(len(keys), np.int64)
        for c0 in range(0, len(keys), chunk):
            kc = keys[c0:c0 + chunk]
            sc, rc = seg[c0:c0 + chunk], row[c0:c0 + chunk]
            for i, (k, r, _) in enumerate(self.segs):
                todo = np.nonzero(sc < 0)[0]
                pos = np.searchsorted(k, kc[todo]).clip(0, len(k) - 1)
                hit = k[pos] == kc[todo]
                sc[todo[hit]] = i; rc[todo[hit]] = r[pos[hit]]
        return seg, row

    def gather(self, keys):
        seg, row = self.locate(keys)
        assert (seg >= 0).all(), f"{(seg < 0).sum()} pairs missing from feature store"
        X = np.empty((len(keys), len(self.cols)), np.float32)
        for i, (_, _, (path, shape)) in enumerate(self.segs):
            m = np.nonzero(seg == i)[0]
            if not len(m):
                continue
            o = np.argsort(row[m])  # sequential memmap reads
            M = np.memmap(path, np.float32, "r", shape=shape)  # mapped only for this call: pages return to standby
            X[m[o]] = M[row[m][o]]
            del M
        return X


def new_segment(split, group, n, cols):
    """Create the next segment dir with an (n, cols) float32 memmap to fill; returns (dir, memmap)."""
    from common import wpath
    d = wpath(split, "fstore", group, f"seg_{len(seg_dirs(split, group)):03d}", "x")[:-2]
    return d, np.memmap(os.path.join(d, "X.f32.tmp"), np.float32, "w+", shape=(n, len(cols)))


def finish_segment(d, n, keys_unsorted, cols):
    """Call after the caller has flushed and dropped every reference to the memmap (Windows file lock)."""
    from common import log
    os.replace(os.path.join(d, "X.f32.tmp"), os.path.join(d, "X.f32"))
    order = np.argsort(keys_unsorted, kind="stable")
    np.save(os.path.join(d, "keys.npy"), keys_unsorted[order]); np.save(os.path.join(d, "rows.npy"), order.astype(np.int32))
    json.dump({"n": n, "cols": list(cols)}, open(os.path.join(d, "meta.json"), "w"))
    log(f"feature store segment written: {d} pairs={n}")


def write_segment(split, keys_unsorted, X_parts, n, group="str", cols=STR_COLS):
    """keys_unsorted aligned with rows of the concatenated X_parts (iterator of float32 blocks)."""
    d, M = new_segment(split, group, n, cols)
    i = 0
    for blk in X_parts:
        M[i:i + len(blk)] = blk; i += len(blk)
    assert i == n
    M.flush(); del M
    finish_segment(d, n, keys_unsorted, cols)


def init_from_parts(split):
    import pyarrow.parquet as pq
    from common import wpath
    from metric import pair_key
    parts = sorted(glob.glob(wpath(split, "feat", "part_*.parquet")))
    keys = []
    for p in parts:
        t = pq.read_table(p, columns=["s1_row", "q_src", "q_row"])
        keys.append(pair_key(t["s1_row"].to_numpy(), t["q_src"].to_numpy(), t["q_row"].to_numpy()))
    keys = np.concatenate(keys)

    def blocks():
        for p in parts:
            t = pq.read_table(p, columns=STR_COLS)
            yield np.column_stack([t[c].to_numpy() for c in STR_COLS]).astype(np.float32)
    write_segment(split, keys, blocks(), len(keys))


def missing_pairs(split, cand_name, group="str"):
    """(s1_row, q_src, q_row) of candidate pairs that no segment of `group` holds yet.
    Only these small arrays survive the call (the full candidate table is released)."""
    import pyarrow.parquet as pq
    from common import wpath
    from metric import pair_key
    t = pq.read_table(wpath(split, cand_name), columns=["s1_row", "q_src", "q_row"])
    s1r, qs, qr = (t[c].to_numpy().copy() for c in ("s1_row", "q_src", "q_row"))
    del t
    new = np.zeros(len(s1r), bool)
    st = Store(split, group)
    for c0 in range(0, len(s1r), 8_000_000):
        sl = slice(c0, c0 + 8_000_000)
        new[sl] = st.locate(pair_key(s1r[sl], qs[sl], qr[sl]))[0] < 0
    del st
    return s1r[new], qs[new], qr[new]


def add_missing(split, cand_name, workers=3, batch=150_000):
    """Featurize the pairs of candidate file `cand_name` that no segment holds yet."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from common import guard, log, wpath
    from metric import pair_key
    s1r, qs, qr = missing_pairs(split, cand_name)
    if not len(s1r):
        log("feature store: all pairs cached"); return
    order = np.lexsort((qr, qs))  # group by query side for locality
    s1r, qs, qr = s1r[order], qs[order], qr[order]
    log(f"feature store: featurizing {len(s1r)} new pairs")
    cols = ["name_norm", "name_core", "addr_norm", "name_skel"]
    recs = {s: pq.read_table(wpath(split, f"s{s}.parquet"), columns=cols + ["nonascii"]) for s in (1, 2, 3)}

    def job(sl):
        L = [recs[1][c].take(pa.array(s1r[sl])).to_pylist() for c in cols]
        qsb, qrb = qs[sl], qr[sl]
        R = [[None] * len(qsb) for _ in cols]
        for s in (2, 3):
            m = np.nonzero(qsb == s)[0]
            for j, c in enumerate(cols):
                for i, v in zip(m, recs[s][c].take(pa.array(qrb[m])).to_pylist()):
                    R[j][i] = v
        return tuple(L + R)

    # resumable: each batch lands in its own .npy part; the segment is assembled once all exist
    tag = f"{len(s1r)}_{int(pair_key(s1r[:1], qs[:1], qr[:1])[0])}_{int(pair_key(s1r[-1:], qs[-1:], qr[-1:])[0])}"
    pend = wpath(split, "fstore", "str", f"pending_{tag}", "x")[:-2]
    sls = [slice(b, b + batch) for b in range(0, len(s1r), batch)]
    part = lambda i: os.path.join(pend, f"part_{i:05d}.npy")
    todo = [i for i in range(len(sls)) if not os.path.exists(part(i))]
    log(f"  batches total={len(sls)} todo={len(todo)}")
    with Pool(workers, maxtasksperchild=10) as pool:
        for w0 in range(0, len(todo), workers):
            guard("fstore")
            win = todo[w0:w0 + workers]
            for i, f in zip(win, pool.imap(pair_feats, [job(sls[i]) for i in win])):
                sl = sls[i]
                na = np.zeros(len(f["name_ratio"]), np.float32)
                for s in (2, 3):
                    m = qs[sl] == s
                    na[m] = recs[s]["nonascii"].take(pa.array(qr[sl][m])).to_numpy(zero_copy_only=False)
                f["nonascii2"] = na
                np.save(part(i) + ".tmp.npy", np.column_stack([f[c] for c in STR_COLS]).astype(np.float32))
                os.replace(part(i) + ".tmp.npy", part(i))
            if (w0 // workers) % 10 == 0:
                log(f"  fstore {min(w0 + workers, len(todo))}/{len(todo)} batches")
    del recs
    write_segment(split, pair_key(s1r, qs, qr), (np.load(part(i)) for i in range(len(sls))), len(s1r))
    import shutil
    shutil.rmtree(pend)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--cand")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.init:
        init_from_parts(a.split)
    else:
        add_missing(a.split, a.cand, a.workers)
