"""Stages 5-7: LightGBM training, Assignment policy search and Validation on train.

1. Sample training pairs by S1 entity (all candidate pairs of a random subset of S1),
   so positives and hard negatives keep their natural proportions.
2. 5-fold out-of-fold (OOF) LightGBM, folds grouped by S1 (work/train/folds.npy), so a model
   never scores pairs of S1 entities it was trained on.
3. Stream all feature parts, attach OOF probabilities -> work/train/oof.parquet.
4. Assignment policy search on folds 0-2 with the official per-S1 macro-F0.5;
   the chosen policy is then reported once on untouched folds 3-4.
5. Train the final model on the sample from all folds -> work/model_final.txt + policy.json.

Usage: python src/model.py [--sample_frac 0.08]
"""
import argparse
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import guard, log, wpath
from metric import macro_f05, pair_key

ID_COLS = ["s1_row", "q_src", "q_row"]
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=12, verbose=-1, seed=7)


def feature_names(split="train", feat="feat"):
    sch = pq.read_schema(sorted(glob.glob(wpath(split, feat, "part_*.parquet")))[0])
    return [c for c in sch.names if c not in ID_COLS]


def gt_keys():
    g = pq.read_table(wpath("train", "gt.parquet"))
    return np.sort(pair_key(g["s1_row"].to_numpy(), g["q_src"].to_numpy(), g["q_row"].to_numpy()))


def is_pos(keys, gk, chunk=4_000_000):
    out = np.empty(len(keys), bool)
    for i in range(0, len(keys), chunk):  # chunked: bounded int64 temporaries
        k = keys[i:i + chunk]
        out[i:i + chunk] = gk[np.searchsorted(gk, k).clip(0, len(gk) - 1)] == k
    return out


def best_per_query(s1, qsrc, qrow, p):
    """Reduce scored pairs to one row per S2/S3 record: its best S1, best and runner-up probability.
    Done once; every threshold / margin setting is then a cheap filter on this compact table."""
    qk = (qsrc.astype(np.int64) << 32) | qrow
    order = np.lexsort((-p, qk))
    qk_o = qk[order]
    idx = np.nonzero(np.r_[True, qk_o[1:] != qk_o[:-1]])[0]
    has2 = np.r_[idx[1:], len(qk_o)] - idx > 1
    second = np.where(has2, p[order[np.minimum(idx + 1, len(order) - 1)]], 0.0).astype(np.float32)
    best = order[idx]
    return {"s1_row": s1[best], "q_src": qsrc[best], "q_row": qrow[best], "p": p[best], "p2": second}


def assign(bq, t, margin=0.0, t3=None):
    """Assignment: each S2/S3 record goes to its highest-probability S1, kept only if the
    probability clears the threshold (t for S2, t3 for S3) and beats the runner-up by `margin`.
    Ties are broken deterministically by lexsort order (lowest S1 row after equal p)."""
    thr = np.where(bq["q_src"] == 3, t if t3 is None else t3, t)
    k = (bq["p"] >= thr) & (bq["p"] - bq["p2"] >= margin)
    return {c: bq[c][k] for c in ("s1_row", "q_src", "q_row")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_frac", type=float, default=0.08)
    ap.add_argument("--rounds", type=int, default=400)
    a = ap.parse_args()
    t0 = time.time()
    folds = np.load(wpath("train", "folds.npy"))
    gk = gt_keys()
    feats = feature_names()
    parts = sorted(glob.glob(wpath("train", "feat", "part_*.parquet")))
    rng = np.random.default_rng(0)
    s1_in = rng.random(len(folds)) < a.sample_frac

    # 1. sample
    X, y, f = [], [], []
    for p in parts:
        t = pq.read_table(p)
        s1 = t["s1_row"].to_numpy()
        m = s1_in[s1]
        X.append(np.column_stack([t[c].to_numpy()[m] for c in feats]).astype(np.float32))
        y.append(is_pos(pair_key(s1[m], t["q_src"].to_numpy()[m], t["q_row"].to_numpy()[m]), gk))
        f.append(folds[s1[m]])
        guard("sample")
    X, y, f = np.vstack(X), np.concatenate(y), np.concatenate(f)
    log(f"train sample: rows={len(y)} pos_rate={y.mean():.4f} features={len(feats)}")

    # 2. OOF models
    models = []
    for k in range(5):
        tr = f != k
        ds = lgb.Dataset(X[tr], y[tr], feature_name=feats, free_raw_data=True)
        va = lgb.Dataset(X[~tr], y[~tr], reference=ds)
        m = lgb.train(PARAMS, ds, a.rounds, valid_sets=[va], callbacks=[lgb.log_evaluation(0)])
        models.append(m)
        m.save_model(wpath("models", f"oof_{k}.txt"))
        log(f"  fold {k} model trained ({m.num_trees()} trees)")
    imp = sorted(zip(models[0].feature_importance("gain"), feats), reverse=True)
    log("  top gain features: " + ", ".join(f"{n}={g:.0f}" for g, n in imp[:15]))

    # 3. OOF predictions for every candidate pair
    outs = []
    for p in parts:
        t = pq.read_table(p)
        s1 = t["s1_row"].to_numpy()
        Xp = np.column_stack([t[c].to_numpy() for c in feats]).astype(np.float32)
        pr = np.zeros(len(s1), np.float32)
        fk = folds[s1]
        for k in range(5):
            m = fk == k
            if m.any():
                pr[m] = models[k].predict(Xp[m], num_threads=12)
        outs.append(pa.table({"s1_row": s1, "q_src": t["q_src"], "q_row": t["q_row"], "p": pr}))
        guard("oof predict")
    oof = pa.concat_tables(outs)
    pq.write_table(oof, wpath("train", "oof.parquet"))
    s1, qs, qr, p = (oof[c].to_numpy() for c in ("s1_row", "q_src", "q_row", "p"))
    del X, outs
    bq = best_per_query(s1, qs, qr, p)
    log(f"OOF predictions: {len(p)} pairs")

    # 4. policy search (tune folds 0-2, report folds 3-4)
    g = pq.read_table(wpath("train", "gt.parquet"))
    gd = {c: g[c].to_numpy() for c in ID_COLS}
    all_s1 = np.arange(len(folds))
    tune_s1, test_s1 = all_s1[folds <= 2], all_s1[folds >= 3]
    ub = macro_f05(tune_s1, {"s1_row": s1[is_pos(pair_key(s1, qs, qr), gk)], "q_src": qs[is_pos(pair_key(s1, qs, qr), gk)],
                             "q_row": qr[is_pos(pair_key(s1, qs, qr), gk)]}, gd)
    log(f"blocking ceiling (perfect classifier on candidates) tune folds: {ub}")
    results = []
    for t in np.r_[np.arange(0.05, 0.95, 0.05), 0.97]:
        r = macro_f05(tune_s1, assign(bq, t), gd)
        results.append({"t": float(t), "margin": 0.0, **r})
    tb = max(results, key=lambda r: r["macro_f05"])["t"]
    for t in np.arange(max(0.01, tb - 0.06), min(0.99, tb + 0.06), 0.01):
        for mg in (0.0, 0.05, 0.1, 0.2):
            r = macro_f05(tune_s1, assign(bq, t, mg), gd)
            results.append({"t": float(t), "margin": mg, **r})
    # no-assignment baseline: keep every pair above the threshold
    for t in (0.3, 0.5, 0.7):
        m = p >= t
        r = macro_f05(tune_s1, {"s1_row": s1[m], "q_src": qs[m], "q_row": qr[m]}, gd)
        results.append({"t": float(t), "margin": -1, "policy": "threshold_only", **r})
    best = max((r for r in results if r["margin"] >= 0), key=lambda r: r["macro_f05"])
    log(f"best policy on tune folds: t={best['t']:.2f} margin={best['margin']} -> {best['macro_f05']:.4f}")
    final = macro_f05(test_s1, assign(bq, best["t"], best["margin"]), gd)
    log(f"HOLDOUT (folds 3-4) with frozen policy: {final}")
    json.dump({"policy": {"t": best["t"], "margin": best["margin"]}, "holdout": final, "ceiling_tune": ub,
               "search": results, "features": feats, "sample_rows": int(len(y))},
              open(wpath("train", "validation.json"), "w"), indent=1)

    # 5. final model on all sampled S1 (all folds)
    X, y = [], []
    for pth in parts:
        t = pq.read_table(pth)
        s1p = t["s1_row"].to_numpy(); m = s1_in[s1p]
        X.append(np.column_stack([t[c].to_numpy()[m] for c in feats]).astype(np.float32))
        y.append(is_pos(pair_key(s1p[m], t["q_src"].to_numpy()[m], t["q_row"].to_numpy()[m]), gk))
    mfin = lgb.train(PARAMS, lgb.Dataset(np.vstack(X), np.concatenate(y), feature_name=feats), a.rounds)
    mfin.save_model(wpath("models", "final.txt"))
    json.dump({"t": best["t"], "margin": best["margin"], "features": feats}, open(wpath("models", "policy.json"), "w"), indent=1)
    log(f"final model + policy saved; model.py total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
