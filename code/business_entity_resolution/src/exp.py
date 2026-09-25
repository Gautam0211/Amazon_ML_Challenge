"""Experiment bookkeeping: dev-fold evaluation (folds 0-2) and the append-only experiments/results.csv.

Primary dev metric: per-S1 macro F1 under the full policy (Assignment + threshold + margin),
reported per fold and as mean +- std over folds 0-2. Folds 3-4 are never touched here.

Usage: python src/exp.py e000     (baseline: score the cached v2 OOF predictions, no retraining)
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from common import ROOT, log, wpath
from metric import macro_f05, pair_key
from model import ID_COLS, assign, best_per_query, gt_keys, is_pos

DEV_FOLDS = (0, 1, 2)
RESULTS = os.path.join(ROOT, "experiments", "results.csv")
COLS = ["exp_id", "parent", "change", "config_hash", "git_commit", "seed", "folds",
        "block_recall", "ceiling_f1", "ceiling_f05", "cand_per_s1", "total_pairs",
        "f1_fold0", "f1_fold1", "f1_fold2", "f1_mean", "f1_std",
        "f05_fold0", "f05_fold1", "f05_fold2", "f05_mean", "f05_std",
        "precision", "recall", "singleton_acc",
        "t_block_s", "t_feat_s", "t_model_s", "peak_mem_gb", "model_params", "best_iter",
        "threshold", "margin", "decision", "notes", "timestamp"]


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return ""


def config_hash(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def load_gt():
    g = pq.read_table(wpath("train", "gt.parquet"))
    return {c: g[c].to_numpy() for c in ID_COLS}


def dev_eval(s1, qs, qr, p, t, margin, folds, gd, bq=None, t3=None):
    """Per-fold metrics on dev folds for scored pairs (s1, qs, qr, p) and a policy (t, margin, S3 threshold)."""
    bq = best_per_query(s1, qs, qr, p) if bq is None else bq
    pred = assign(bq, t, margin, t3)
    return {k: macro_f05(np.nonzero(folds == k)[0], pred, gd) for k in DEV_FOLDS}


def ceiling(s1, qs, qr, folds, gd, gk):
    """Blocking recall and the perfect-classifier ceiling on dev folds."""
    hit = is_pos(pair_key(s1, qs, qr), gk)
    dev = np.nonzero(folds <= max(DEV_FOLDS))[0]
    ub = macro_f05(dev, {"s1_row": s1[hit], "q_src": qs[hit], "q_row": qr[hit]}, gd)
    return ub["pair_recall"], ub["macro_f1"], ub["macro_f05"]


def summarize(per_fold):
    f1 = np.array([per_fold[k]["macro_f1"] for k in DEV_FOLDS])
    f5 = np.array([per_fold[k]["macro_f05"] for k in DEV_FOLDS])
    tot = lambda c: np.mean([per_fold[k][c] for k in DEV_FOLDS])
    out = {f"f1_fold{k}": round(float(v), 5) for k, v in zip(DEV_FOLDS, f1)}
    out.update({f"f05_fold{k}": round(float(v), 5) for k, v in zip(DEV_FOLDS, f5)})
    out.update(f1_mean=round(float(f1.mean()), 5), f1_std=round(float(f1.std()), 5),
               f05_mean=round(float(f5.mean()), 5), f05_std=round(float(f5.std()), 5),
               precision=round(float(tot("pair_precision")), 5), recall=round(float(tot("pair_recall")), 5),
               singleton_acc=round(float(tot("singleton_acc")), 5))
    return out


def append_result(row):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    new = not os.path.exists(RESULTS)
    row = {**{c: "" for c in COLS}, **row, "timestamp": time.strftime("%Y-%m-%d %H:%M")}
    if not row["git_commit"]:
        row["git_commit"] = git_commit()
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLS)
        if new:
            w.writeheader()
        w.writerow({c: row[c] for c in COLS})
    log("RESULT " + json.dumps({k: row[k] for k in ("exp_id", "f1_mean", "f1_std", "f05_mean", "block_recall", "decision")}))


def e000():
    """Baseline v2: recompute dev-fold metrics from the cached OOF predictions (no retraining)."""
    folds = np.load(wpath("train", "folds.npy"))
    gd, gk = load_gt(), gt_keys()
    pol = json.load(open(wpath("models", "policy.json")))
    oof = pq.read_table(wpath("train", "oof.parquet"))
    s1, qs, qr, p = (oof[c].to_numpy() for c in ("s1_row", "q_src", "q_row", "p"))
    del oof
    rec, c1, c5 = ceiling(s1, qs, qr, folds, gd, gk)
    per = dev_eval(s1, qs, qr, p, pol["t"], pol["margin"], folds, gd)
    for k in DEV_FOLDS:
        log(f"  fold {k}: {per[k]}")
    append_result({
        "exp_id": "E000", "parent": "", "change": "baseline v2 (cached OOF; models trained on 4 folds incl. 3-4)",
        "config_hash": config_hash({"k": 10, "cap": 5000, "rel": 0.4, "sample_frac": 0.08, "rounds": 400}),
        "seed": 7, "folds": "0-2", "block_recall": round(rec, 5), "ceiling_f1": round(c1, 5), "ceiling_f05": round(c5, 5),
        "cand_per_s1": round(len(s1) / len(folds), 2), "total_pairs": len(s1), **summarize(per),
        "t_block_s": 49 * 60, "t_feat_s": 15 * 60, "t_model_s": 19 * 60, "peak_mem_gb": 3.9,
        "model_params": "lr0.08 nl127 mdl100 ff0.8 bf0.8 l2=1 400r", "best_iter": 400,
        "threshold": pol["t"], "margin": pol["margin"], "decision": "baseline"})


if __name__ == "__main__":
    {"e000": e000}[sys.argv[1]]()
