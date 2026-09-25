"""Stage 1+2 driver: read raw TSVs in chunks, normalize + transliterate, emit blocking tokens.

Output per (split, source): work/<split>/s<src>.parquet with
  id_num (int64, lossless ID part), country, raw fields, normalized fields,
  btok (list<int64>): hashed blocking tokens, namespaced by country and field.

Usage: python src/prep.py --split train   (resumable: finished files are skipped)
"""
import argparse
import os
from multiprocessing import Pool

from prep_worker import work  # workers re-import this module on Windows: keep heavy imports inside functions

CHUNK = 200_000


def prep_one(split, src, pool, workers):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from common import guard, log, read_tsv, src_file, wpath
    dst = wpath(split, f"s{src}.parquet")
    if os.path.exists(dst):
        log(f"skip {dst} (exists)")
        return
    guard(f"prep {split} s{src}")
    t = read_tsv(src_file(split, src))
    ids = t["entity_id"]
    num = pc.cast(pc.utf8_slice_codeunits(ids, 3), pa.int64())
    rebuilt = pc.binary_join_element_wise(pa.scalar(f"S{src}-"), pc.cast(num, pa.string()), "")
    assert pc.all(pc.equal(rebuilt, ids)).as_py(), "ID round-trip failed"
    assert len(pc.unique(num)) == len(num), "duplicate IDs"
    log(f"{split} s{src}: {t.num_rows} rows, ID round-trip verified")
    def job(i):
        s = t.slice(i * CHUNK, CHUNK)
        return s["business_name"].to_pylist(), s["business_address"].to_pylist(), s["country"].to_pylist()

    def results():
        # Bounded windows: only ~2x workers chunks exist as Python objects at any time.
        n = (t.num_rows + CHUNK - 1) // CHUNK
        for b in range(0, n, 2 * workers):
            guard(f"prep {split} s{src} chunk {b}")
            yield from pool.imap(work, [job(i) for i in range(b, min(n, b + 2 * workers))])

    tmp = dst + ".tmp"
    w = None
    for k, (o, lens, h) in enumerate(results()):
        s = t.slice(k * CHUNK, CHUNK)
        offs = np.zeros(len(lens) + 1, np.int32); np.cumsum(lens, out=offs[1:])
        tb = pa.table({
            "id_num": num.slice(k * CHUNK, CHUNK),
            "country": s["country"],
            "name_raw": s["business_name"], "addr_raw": s["business_address"],
            **{c: pa.array(v, pa.bool_() if c == "nonascii" else pa.string()) for c, v in o.items()},
            "btok": pa.ListArray.from_arrays(pa.array(offs), pa.array(h)),
        })
        if w is None:
            w = pq.ParquetWriter(tmp, tb.schema, compression="zstd")
        w.write_table(tb)
    w.close()
    os.replace(tmp, dst)
    log(f"wrote {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        for src in (1, 2, 3):
            prep_one(a.split, src, pool, a.workers)
