"""Phase 1: single-field retrievers for multi-retriever blocking.

Each retriever builds its own tokens from the normalized columns (vectorized: pyarrow split +
pandas hashing, char n-grams from the UTF-8 byte buffer) and reuses block.block_country for the
IDF-cosine top-k, so every retriever gets the same memory-safe chunking, country partitioning,
DF cap and relative cutoff.

Retrievers:
  name   core-name words + joined core name
  skel   phonetic skeletons of core-name words
  addr   address words and numbers
  cng3   character 3-grams of the joined core name (typos, transliteration variants)
  cng4   character 4-grams of the joined core name

Output: work/train/ret_<name>_k<k>_c<cap>_r<rel>.parquet (same schema as cand.parquet)

Usage: python src/retrieve.py --split train --ret name --k 10 [--cap 5000 --rel 0.4 --qfrac 1]
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from block import block_country
from common import log, wpath

SALT = {"name": 1, "skel": 2, "addr": 3, "cng3": 4, "cng4": 5, "num": 6}


def _hash_words(arr, salt, minlen=2, digits=True, only_digits=False):
    """(parent row, token hash) for space-separated words of a string array."""
    lst = pc.split_pattern(arr, " ")
    vals = pc.list_flatten(lst)
    parent = pc.list_parent_indices(lst).to_numpy()
    ln = pc.utf8_length(vals).to_numpy()
    isd = pc.utf8_is_digit(vals).to_numpy(zero_copy_only=False)
    keep = (ln >= minlen) | (isd & (ln >= 1)) if digits else (ln >= minlen) & ~isd
    if only_digits:
        keep = isd & (ln >= 1)
    v = vals.filter(pa.array(keep)).to_numpy(zero_copy_only=False)
    h = pd.util.hash_array(v, categorize=False).view(np.int64) ^ np.int64(salt * 0x9E3779B97F4A7C15 - 2**63)
    return parent[keep], h


def _ngrams(arr, n, salt):
    """(parent row, hash) of character n-grams of each string with spaces removed."""
    s = pc.replace_substring(arr, " ", "").combine_chunks() if isinstance(arr, pa.ChunkedArray) else pc.replace_substring(arr, " ", "")
    s = s.cast(pa.large_string())
    offs = np.frombuffer(s.buffers()[1], np.int64)[s.offset:s.offset + len(s) + 1]
    buf = np.frombuffer(s.buffers()[2], np.uint8)
    lens = np.diff(offs)
    cnt = np.maximum(lens - n + 1, 0)
    parent = np.repeat(np.arange(len(s)), cnt)
    start = np.repeat(offs[:-1], cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    code = np.zeros(len(start), np.int64)
    for j in range(n):
        code = (code << 8) | buf[start + j]
    return parent, code * np.int64(1_000_003) + salt


def tokens(split, src, ret):
    t = pq.read_table(wpath(split, f"s{src}.parquet"), columns=["country", "name_core", "name_skel", "addr_norm"])
    n = t.num_rows
    if ret == "name":
        p1, h1 = _hash_words(t["name_core"], SALT["name"], digits=False)
        joined = pc.replace_substring(t["name_core"], " ", "")
        multi = pc.greater(pc.count_substring(t["name_core"], " "), 0).to_numpy(zero_copy_only=False)
        p2, h2 = _hash_words(joined, SALT["name"], minlen=2, digits=False)
        k2 = multi[p2]
        parent, h = np.r_[p1, p2[k2]], np.r_[h1, h2[k2]]
    elif ret == "skel":
        parent, h = _hash_words(t["name_skel"], SALT["skel"], minlen=1)
    elif ret == "addr":
        parent, h = _hash_words(t["addr_norm"], SALT["addr"])
    elif ret == "num":
        parent, h = _hash_words(t["addr_norm"], SALT["num"], only_digits=True)
    elif ret in ("cng3", "cng4"):
        parent, h = _ngrams(t["name_core"], int(ret[-1]), SALT[ret])
    else:
        raise ValueError(ret)
    # de-duplicate tokens within a record, sort by record
    o = np.lexsort((h, parent))
    parent, h = parent[o].astype(np.int64), h[o]
    d = np.r_[True, (parent[1:] != parent[:-1]) | (h[1:] != h[:-1])]
    parent, h = parent[d], h[d]
    offs = np.zeros(n + 1, np.int64); np.cumsum(np.bincount(parent, minlength=n), out=offs[1:])
    return t["country"].to_numpy(zero_copy_only=False).astype(str), offs, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--ret", required=True, choices=list(SALT))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cap", type=int, default=5000)
    ap.add_argument("--rel", type=float, default=0.4)
    ap.add_argument("--qfrac", type=float, default=1.0)
    ap.add_argument("--budget", type=int, default=20_000_000)
    a = ap.parse_args()
    tag = f"ret_{a.ret}_k{a.k}_c{a.cap}_r{a.rel}" + (f"_q{a.qfrac}" if a.qfrac < 1 else "")
    dst = wpath(a.split, tag + ".parquet")
    if os.path.exists(dst):
        log(f"skip {dst} (exists)"); return
    s1 = tokens(a.split, 1, a.ret)
    qs = {2: tokens(a.split, 2, a.ret), 3: tokens(a.split, 3, a.ret)}
    log(f"retriever {a.ret}: tokens S1={len(s1[2])} S2={len(qs[2][2])} S3={len(qs[3][2])}")
    schema = pa.schema([("q_src", pa.int8()), ("q_row", pa.int32()), ("s1_row", pa.int32()), ("bscore", pa.float32()), ("brank", pa.int8())])
    w = pq.ParquetWriter(dst + ".tmp", schema, compression="zstd")
    budget = a.budget
    for ctry in sorted(set(s1[0])):
        budget = block_country(ctry, s1, qs, a.k, a.cap, budget, w, a.rel, a.qfrac)
    w.close()
    os.replace(dst + ".tmp", dst)
    log(f"wrote {dst}")


if __name__ == "__main__":
    main()
