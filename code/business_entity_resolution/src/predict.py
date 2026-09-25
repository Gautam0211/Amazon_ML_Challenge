"""Stage 8: test inference and official Output files.

Needs: work/test/{s1,s2,s3}.parquet (prep.py), work/test/cand.parquet (block.py),
       work/test/feat/ (features.py), work/models/final.txt + policy.json (model.py).
Writes output/matching_results.tsv and output/candidate_pairs.tsv: one row per test S1,
exact original IDs, empty list when nothing is found. candidate_pairs = every pair the model scored.

Usage: python src/predict.py
"""
import glob
import json
import os

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import OUT, guard, log, wpath
from model import assign, best_per_query


def id_strings(src, nums):
    return pc.binary_join_element_wise(pa.scalar(f"S{src}-"), pc.cast(pa.array(nums), pa.string()), "")


def write_lists(path, col, s1_ids, s1_row, q_src, q_row, qnum):
    """One line per S1 (in test_source1 order) with a comma-joined, de-duplicated ID list.
    Streams S1 in sorted order, so ID strings exist only for one line at a time
    (91M candidate ID strings at once would need ~5 GB)."""
    order = np.lexsort((q_row, q_src, s1_row))
    s1o, qso, qro = s1_row[order], q_src[order], q_row[order]
    del order
    keep = np.r_[True, (s1o[1:] != s1o[:-1]) | (qso[1:] != qso[:-1]) | (qro[1:] != qro[:-1])]  # de-duplicate
    s1o, qso, qro = s1o[keep], qso[keep], qro[keep]
    idn = np.where(qso == 2, qnum[2][np.minimum(qro, len(qnum[2]) - 1)], qnum[3][np.minimum(qro, len(qnum[3]) - 1)])
    bounds = np.searchsorted(s1o, np.arange(len(s1_ids) + 1))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for i, sid in enumerate(s1_ids):
            lo, hi = bounds[i], bounds[i + 1]
            f.write(sid + "\t" + ",".join(f"S{s}-{n}" for s, n in zip(qso[lo:hi].tolist(), idn[lo:hi].tolist())) + "\n")
    os.replace(tmp, path)


def main():
    pol = json.load(open(wpath("models", "policy.json")))
    feats = pol["features"]
    model = lgb.Booster(model_file=wpath("models", "final.txt"))
    scores = wpath("test", "scores.parquet")
    if os.path.exists(scores):
        log(f"reusing {scores}")
    else:
        score_all(feats, model, scores)
    sc = pq.read_table(scores)
    s1, qs, qr, p = (sc[c].to_numpy() for c in ("s1_row", "q_src", "q_row", "p"))
    del sc
    log(f"scored test pairs: {len(p)}")
    m = assign(best_per_query(s1, qs, qr, p), pol["t"], pol["margin"])
    del p
    s1num = pq.read_table(wpath("test", "s1.parquet"), columns=["id_num"])["id_num"].to_numpy()
    s1_ids = id_strings(1, s1num).to_pylist()
    qnum = {s: pq.read_table(wpath("test", f"s{s}.parquet"), columns=["id_num"])["id_num"].to_numpy() for s in (2, 3)}
    os.makedirs(OUT, exist_ok=True)
    write_lists(os.path.join(OUT, "candidate_pairs.tsv"), "candidate_entity_ids", s1_ids, s1, qs, qr, qnum)
    write_lists(os.path.join(OUT, "matching_results.tsv"), "matched_entity_ids", s1_ids, m["s1_row"], m["q_src"], m["q_row"], qnum)
    n_match = len(m["s1_row"]); n_s1_hit = len(np.unique(m["s1_row"]))
    log(f"wrote outputs: test S1={len(s1_ids)} matched pairs={n_match} S1 with >=1 match={n_s1_hit} "
        f"({n_s1_hit / len(s1_ids):.3f}); candidates/S1={len(s1) / len(s1_ids):.2f}")


def score_all(feats, model, dst):
    """Score every test candidate pair with the final model (checkpointed to dst)."""
    outs = []
    for p in sorted(glob.glob(wpath("test", "feat", "part_*.parquet"))):
        t = pq.read_table(p)
        X = np.column_stack([t[c].to_numpy() for c in feats]).astype(np.float32)
        outs.append(pa.table({"s1_row": t["s1_row"], "q_src": t["q_src"], "q_row": t["q_row"],
                              "p": model.predict(X, num_threads=12).astype(np.float32)}))
        guard("predict")
    pq.write_table(pa.concat_tables(outs), dst)


if __name__ == "__main__":
    main()
