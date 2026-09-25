"""Dev experiment runner (folds 0-2 only; folds 3-4 labels are never read for training or tuning).

Stages, each cached so an experiment recomputes only what its change touches:
  candidates  work/train/<cand>.parquet            (block.py / union.py; one file per blocking config)
  strings     work/train/fstore/                   (fstore.py; features computed once per pair)
  context     rebuilt from the candidate file       (cheap)
  model       work/exp/<id>/oof.parquet            (3 fold models trained on dev folds only)

Scoring: each pair is scored by the model that did not train on its S1's fold (dev folds);
pairs of fold 3-4 S1 (they compete inside queries) are scored by model s1_row % 3.

Usage: python src/dev.py --exp E001 --parent E000 --change "..." [--cand cand.parquet]
       [--params '{"learning_rate":0.05}'] [--rounds 400] [--early_stop 0] [--sample_frac 0.08]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import guard, log, peak_gb, wpath
from exp import DEV_FOLDS, append_result, ceiling, config_hash, dev_eval, load_gt, summarize
from features import Context
from fstore import Store
from metric import pair_key
from model import PARAMS, assign, best_per_query, gt_keys, is_pos

BASE = ("q_src", "q_row", "s1_row", "bscore", "brank")


def load_cand(name):
    t = pq.read_table(wpath("train", name))
    return {c: t[c].to_numpy() for c in t.column_names}


class Assembler:
    """Builds model rows for index sets of a candidate table: q_src, context, extra cand columns, strings."""

    def __init__(self, cand, drop=(), groups=("str",)):
        self.c = cand
        self.ctx = Context(cand)
        self.extra = [c for c in cand if c not in BASE]
        self.stores = [Store("train", g) for g in groups]
        names = ["q_src_f"] + [f"ctx_{k}" for k in self.ctx.batch(slice(0, 1))] + [f"cand_{c}" for c in self.extra]
        names += [n for st in self.stores for n in st.cols]
        self.keep = [i for i, n in enumerate(names) if n not in drop]
        self.names = [names[i] for i in self.keep]

    def rows(self, idx, chunk=2_000_000):
        if len(idx) > chunk:  # fill a preallocated matrix: no 2x temporaries for big samples
            X = np.empty((len(idx), len(self.names)), np.float32)
            for i in range(0, len(idx), chunk):
                X[i:i + chunk] = self.rows(idx[i:i + chunk])
            return X
        c = self.c
        cols = [c["q_src"][idx].astype(np.float32)] + list(self.ctx.batch(idx).values()) + [c[e][idx].astype(np.float32) for e in self.extra]
        keys = pair_key(c["s1_row"][idx], c["q_src"][idx], c["q_row"][idx])
        X = np.hstack([np.column_stack(cols)] + [st.gather(keys) for st in self.stores])
        return X[:, self.keep] if len(self.keep) < X.shape[1] else X


def train_models(asm, y_all, folds, s1_in, params, rounds, early_stop, neg_keep=1.0, hard_rel=0.6):
    c = asm.c
    f = folds[c["s1_row"]]
    samp = np.nonzero(s1_in[c["s1_row"]] & (f <= max(DEV_FOLDS)))[0]
    wt = None
    if neg_keep < 1:  # Phase 7: keep positives and hard negatives, subsample easy negatives (re-weighted)
        hard = (c["brank"][samp] == 0) | (c["bscore"][samp] >= hard_rel * asm.ctx.top1[asm.ctx.gid[samp]])
        easy = ~y_all[samp] & ~hard
        keep = ~easy | (np.random.default_rng(1).random(len(samp)) < neg_keep)
        log(f"hard-negative sampling: rows {len(samp)} -> {keep.sum()} (easy negatives {easy.mean():.3f}, kept at {neg_keep})")
        samp, wt = samp[keep], np.where(easy[keep], 1 / neg_keep, 1.0).astype(np.float32)
    X = asm.rows(samp)
    y, fs = y_all[samp], f[samp]
    log(f"train sample rows={len(y)} pos_rate={y.mean():.4f} features={len(asm.names)}")
    # bin once, then per-fold subsets share the binned data (raw float matrix freed right away)
    full = lgb.Dataset(X, y, weight=wt, feature_name=asm.names, free_raw_data=True, params={"verbose": -1}).construct()
    del X
    models, iters = [], []
    for k in DEV_FOLDS:
        tr = fs != k
        ds = full.subset(np.nonzero(tr)[0])
        cb = [lgb.log_evaluation(0)]
        vs = []
        if early_stop:
            vs = [full.subset(np.nonzero(~tr)[0])]
            cb.append(lgb.early_stopping(early_stop, verbose=False))
        m = lgb.train(params, ds, rounds, valid_sets=vs, callbacks=cb)
        models.append(m); iters.append(m.best_iteration or m.num_trees())
        log(f"  fold {k}: trees={m.num_trees()} best_iter={iters[-1]}")
        guard("train")
    imp = sorted(zip(models[0].feature_importance("gain"), asm.names), reverse=True)
    return models, iters, imp


def score_all(asm, models, iters, chunk=2_000_000):
    c = asm.c
    n = len(c["s1_row"])
    p = np.empty(n, np.float32)
    fold = np.load(wpath("train", "folds.npy"))
    for i in range(0, n, chunk):
        idx = np.arange(i, min(n, i + chunk))
        X = asm.rows(idx)
        s1 = c["s1_row"][idx]
        fk = fold[s1].astype(np.int64)
        which = np.where(fk <= max(DEV_FOLDS), fk, s1 % len(DEV_FOLDS))
        for k, m in enumerate(models):
            sel = which == k
            if sel.any():
                p[idx[sel]] = m.predict(X[sel], num_iteration=iters[k], num_threads=12)
        guard("score")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--parent", default="")
    ap.add_argument("--change", default="")
    ap.add_argument("--cand", default="cand.parquet")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--early_stop", type=int, default=0)
    ap.add_argument("--sample_frac", type=float, default=0.08)
    ap.add_argument("--neg_keep", type=float, default=1.0, help="keep rate of easy negatives (Phase 7)")
    ap.add_argument("--hard_rel", type=float, default=0.6, help="negative is hard if rank 0 or score >= hard_rel x query best")
    ap.add_argument("--drop", default="", help="comma-separated feature names to exclude")
    ap.add_argument("--groups", default="str", help="feature-store groups: str[,ps]")
    ap.add_argument("--t", type=float, default=0.6)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--t_block_s", type=float, default=0)
    ap.add_argument("--no_log", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    out = wpath("exp", a.exp, "oof.parquet")
    params = {**PARAMS, **json.loads(a.params)}
    # feature-store top-ups run as child processes: their memory is returned before the model stage
    tf = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, os.path.join(here, "fstore.py"), "--cand", a.cand], check=True)
    if "ps" in a.groups.split(","):
        subprocess.run([sys.executable, os.path.join(here, "pairscore.py"), "--cand", a.cand], check=True)
    t_feat = time.time() - tf
    folds = np.load(wpath("train", "folds.npy"))
    gd, gk = load_gt(), gt_keys()
    cand = load_cand(a.cand)
    s1, qs, qr = cand["s1_row"], cand["q_src"], cand["q_row"]
    y = is_pos(pair_key(s1, qs, qr), gk)
    tf = time.time()
    iters, imp = [a.rounds] * 3, []
    if os.path.exists(out):
        log(f"reusing {out}")
        p = pq.read_table(out, columns=["p"])["p"].to_numpy()
    else:
        asm = Assembler(cand, drop=set(filter(None, a.drop.split(","))), groups=a.groups.split(","))
        s1_in = np.random.default_rng(0).random(len(folds)) < a.sample_frac
        mp = [wpath("exp", a.exp, f"model_{k}.txt") for k in DEV_FOLDS]
        if all(map(os.path.exists, mp)):  # resume after an abort in scoring
            models = [lgb.Booster(model_file=f) for f in mp]
            iters = [m.best_iteration or m.num_trees() for m in models]
            imp = sorted(zip(models[0].feature_importance("gain"), asm.names), reverse=True)
        else:
            models, iters, imp = train_models(asm, y, folds, s1_in, params, a.rounds, a.early_stop, a.neg_keep, a.hard_rel)
            for m, f in zip(models, mp):
                m.save_model(f)
        log("  top gain: " + ", ".join(f"{n}={g:.0f}" for g, n in imp[:12]))
        p = score_all(asm, models, iters)
        del asm
        pq.write_table(pa.table({"s1_row": s1, "q_src": qs, "q_row": qr, "p": p}), out)
    t_model = time.time() - tf
    rec, c1, c5 = ceiling(s1, qs, qr, folds, gd, gk)
    bq = best_per_query(s1, qs, qr, p)
    per = dev_eval(None, None, None, None, a.t, a.margin, folds, gd, bq=bq)
    summ = summarize(per)
    # informational: best F1 policy on dev folds (Phase 9 tunes this properly)
    grid = [(t, mg) for t in np.arange(0.3, 0.86, 0.05) for mg in (0.0, 0.1, 0.2)]
    best = max(((np.mean([v["macro_f1"] for v in dev_eval(None, None, None, None, t, mg, folds, gd, bq=bq).values()]), t, mg)
                for t, mg in grid))
    log(f"{a.exp}: F1 {summ['f1_mean']} +- {summ['f1_std']} F0.5 {summ['f05_mean']} recall_block {rec:.4f} | best-F1 policy t={best[1]:.2f} m={best[2]} -> {best[0]:.5f}")
    json.dump({"per_fold": per, "importance": [(n, float(g)) for g, n in imp], "iters": iters, "params": params,
               "best_f1_policy": best}, open(wpath("exp", a.exp, "summary.json"), "w"), indent=1, default=float)
    if not a.no_log:
        append_result({
            "exp_id": a.exp, "parent": a.parent, "change": a.change,
            "config_hash": config_hash({"cand": a.cand, "params": params, "rounds": a.rounds, "es": a.early_stop,
                                        "frac": a.sample_frac, "drop": a.drop, "groups": a.groups,
                                        "neg_keep": a.neg_keep, "hard_rel": a.hard_rel}),
            "seed": params.get("seed", 7), "folds": "0-2", "block_recall": round(rec, 5), "ceiling_f1": round(c1, 5),
            "ceiling_f05": round(c5, 5), "cand_per_s1": round(len(s1) / len(folds), 2), "total_pairs": len(s1), **summ,
            "t_block_s": a.t_block_s, "t_feat_s": round(t_feat), "t_model_s": round(t_model), "peak_mem_gb": round(peak_gb(), 2),
            "model_params": json.dumps({k: v for k, v in params.items() if k not in ("verbose", "num_threads")}),
            "best_iter": "/".join(map(str, iters)), "threshold": a.t, "margin": a.margin,
            "notes": f"bestF1policy t={best[1]:.2f} m={best[2]} F1={best[0]:.5f}"})
    log(f"dev.py total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
