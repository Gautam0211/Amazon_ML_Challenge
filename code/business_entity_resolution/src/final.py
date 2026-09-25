"""Final path: one model on all labelled folds, then test blocking, features, scoring and output files.

Uses the same stages and sampling as dev.py, so a dev experiment's config reproduces here exactly:
  model    work/models/<tag>/final.txt   (all folds 0-4, sample_rows with max_fold=4)
  blocking work/test/<cand>              (block.py, same flags as the train candidate file)
  features work/test/fstore/             (fstore.py / pairscore.py top-ups)
  scores   work/test/scores_<tag>.parquet
  outputs  output/{matching_results,candidate_pairs}.tsv via predict.write_lists
Policy (t, margin, S3 t) comes from work/exp/<policy_exp>/policy.json (policy.py, tuned on dev folds for F0.5).

Usage: python src/final.py --tag E005 --policy_exp E005 --cand cand_adB.parquet --sample_frac 0.4 --neg_keep 0.3
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import OUT, guard, log, peak_gb, wpath
from dev import Assembler, RowSeq, load_cand, sample_rows
from metric import pair_key
from model import PARAMS, assign, best_per_query, gt_keys, is_pos
from predict import id_strings, write_lists

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTIVE = "10,20,50,0.3,3,8"  # cand_adB blocking config (Phase 2)


def run(*args):
    subprocess.run([sys.executable, *args], check=True, cwd=HERE)


def train_final(a, params, path):
    folds = np.load(wpath("train", "folds.npy"))
    cand = load_cand(a.cand)
    y = is_pos(pair_key(cand["s1_row"], cand["q_src"], cand["q_row"]), gt_keys())
    asm = Assembler(cand, groups=a.groups.split(","))
    s1_in = np.random.default_rng(0).random(len(folds)) < a.sample_frac  # same draw as dev.py
    samp, wt = sample_rows(asm, y, folds, s1_in, a.neg_keep, a.hard_rel, max_fold=int(folds.max()))
    log(f"final train rows={len(samp)} pos_rate={y[samp].mean():.4f} features={len(asm.names)}")
    ds = lgb.Dataset(RowSeq(asm, samp), y[samp], weight=wt, feature_name=asm.names, free_raw_data=True,
                     params={"verbose": -1})
    m = lgb.train(params, ds, a.rounds, callbacks=[lgb.log_evaluation(0)])
    guard("final-train")
    m.save_model(path)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="name of this final run (model dir, scores file)")
    ap.add_argument("--policy_exp", required=True, help="dev experiment whose policy.json to use")
    ap.add_argument("--cand", default="cand_adB.parquet")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--sample_frac", type=float, default=0.4)
    ap.add_argument("--neg_keep", type=float, default=0.3)
    ap.add_argument("--hard_rel", type=float, default=0.6)
    ap.add_argument("--groups", default="str")
    a = ap.parse_args()
    t0, times = time.time(), {}
    params = {**PARAMS, **json.loads(a.params)}
    mpath = wpath("models", a.tag, "final.txt")
    os.makedirs(os.path.dirname(mpath), exist_ok=True)

    t = time.time()
    model = lgb.Booster(model_file=mpath) if os.path.exists(mpath) else train_final(a, params, mpath)
    times["train_s"] = round(time.time() - t)

    t = time.time()
    if not os.path.exists(wpath("test", a.cand)):
        run("block.py", "--split", "test", "--adaptive", ADAPTIVE, "--out", a.cand)
    times["block_s"] = round(time.time() - t)

    t = time.time()
    if not glob.glob(wpath("test", "fstore", "str", "seg_*")):
        run("fstore.py", "--split", "test", "--init")  # seed from the k=10 feature parts
    run("fstore.py", "--split", "test", "--cand", a.cand)
    if "ps" in a.groups.split(","):
        run("pairscore.py", "--split", "test", "--cand", a.cand)
    times["feat_s"] = round(time.time() - t)

    t = time.time()
    scores = wpath("test", f"scores_{a.tag}.parquet")
    cand = pq.read_table(wpath("test", a.cand))
    cand = {c: cand[c].to_numpy() for c in cand.column_names}
    s1, qs, qr = cand["s1_row"], cand["q_src"], cand["q_row"]
    if os.path.exists(scores):
        p = pq.read_table(scores, columns=["p"])["p"].to_numpy()
    else:
        asm = Assembler(cand, groups=a.groups.split(","), split="test")
        assert asm.names == model.feature_name(), "test features differ from the model's"
        p = np.empty(len(s1), np.float32)
        for i in range(0, len(s1), 2_000_000):
            p[i:i + 2_000_000] = model.predict(asm.rows(np.arange(i, min(len(s1), i + 2_000_000))), num_threads=12)
            guard("final-score")
        del asm
        pq.write_table(pa.table({"s1_row": s1, "q_src": qs, "q_row": qr, "p": p}), scores)
    times["score_s"] = round(time.time() - t)

    pol = json.load(open(wpath("exp", a.policy_exp, "policy.json")))  # read late: policy.py may run alongside
    m = assign(best_per_query(s1, qs, qr, p), pol["t"], pol["margin"], pol.get("t3"))
    s1_ids = id_strings(1, pq.read_table(wpath("test", "s1.parquet"), columns=["id_num"])["id_num"].to_numpy()).to_pylist()
    qnum = {s: pq.read_table(wpath("test", f"s{s}.parquet"), columns=["id_num"])["id_num"].to_numpy() for s in (2, 3)}
    os.makedirs(OUT, exist_ok=True)
    write_lists(os.path.join(OUT, "candidate_pairs.tsv"), "candidate_entity_ids", s1_ids, s1, qs, qr, qnum)
    write_lists(os.path.join(OUT, "matching_results.tsv"), "matched_entity_ids", s1_ids, m["s1_row"], m["q_src"], m["q_row"], qnum)
    n_hit = len(np.unique(m["s1_row"]))
    info = {"tag": a.tag, "policy": pol, "args": vars(a), **times, "total_s": round(time.time() - t0),
            "peak_mem_gb": round(peak_gb(), 2), "test_s1": len(s1_ids), "test_pairs": len(s1),
            "matched_pairs": len(m["s1_row"]), "s1_matched_frac": round(n_hit / len(s1_ids), 4)}
    json.dump(info, open(wpath("models", a.tag, "run.json"), "w"), indent=1, default=float)
    log(f"FINAL {json.dumps(info, default=float)}")


if __name__ == "__main__":
    main()
