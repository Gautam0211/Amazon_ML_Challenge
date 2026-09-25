"""Phase 9: tune the decision rule (threshold, margin, per-source thresholds) on dev folds 0-2.

Reads an experiment's cached OOF scores; the model is untouched. Objective: mean macro-F1 over
folds 0-2 (F0.5 reported alongside).

Usage: python src/policy.py --exp E002 [--log EXXX --parent E002]
"""
import argparse
import json

import numpy as np
import pyarrow.parquet as pq

from common import log, wpath
from exp import DEV_FOLDS, append_result, dev_eval, load_gt, summarize
from model import best_per_query


def mean_f05(per):
    return float(np.mean([per[k]["macro_f05"] for k in DEV_FOLDS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--log", default="", help="experiment id to log the tuned policy under")
    ap.add_argument("--parent", default="")
    a = ap.parse_args()
    folds = np.load(wpath("train", "folds.npy"))
    gd = load_gt()
    t = pq.read_table(wpath("exp", a.exp, "oof.parquet"))
    bq = best_per_query(*(t[c].to_numpy() for c in ("s1_row", "q_src", "q_row", "p")))
    del t
    ev = lambda th, mg, t3=None: dev_eval(None, None, None, None, th, mg, folds, gd, bq=bq, t3=t3)
    res = {}
    for th in np.arange(0.20, 0.81, 0.02):
        for mg in (0.0, 0.05, 0.1, 0.2):
            res[(round(th, 2), mg)] = mean_f05(ev(th, mg))
    (tb, mb), fb = max(res.items(), key=lambda kv: kv[1])
    log(f"coarse best t={tb} margin={mb} F05={fb:.5f}")
    for th in np.arange(tb - 0.02, tb + 0.021, 0.005):
        for mg in sorted({0.0, 0.02, 0.05, 0.08, 0.1, 0.15, mb}):
            res[(round(th, 3), mg)] = mean_f05(ev(th, mg))
    (tb, mb), fb = max(res.items(), key=lambda kv: kv[1])
    # flatness: F0.5 at +-0.05 threshold
    for d in (-0.05, 0.05):
        log(f"  t={tb + d:.3f} m={mb}: F05={mean_f05(ev(tb + d, mb)):.5f}")
    # per-source threshold for S3
    best3 = (None, fb)
    for t3 in np.arange(tb - 0.1, tb + 0.101, 0.02):
        f = mean_f05(ev(tb, mb, t3))
        if f > best3[1] + 1e-5:
            best3 = (round(float(t3), 3), f)
    log(f"fine best t={tb:.3f} margin={mb} F05={fb:.5f}; separate S3 threshold: {best3}")
    per = ev(tb, mb, best3[0])
    summ = summarize(per)
    json.dump({"t": tb, "margin": mb, "t3": best3[0], "f05": summ["f05_mean"], "f1": summ["f1_mean"]}, open(wpath("exp", a.exp, "policy.json"), "w"))
    log(f"policy for {a.exp}: {summ}")
    if a.log:
        append_result({"exp_id": a.log, "parent": a.parent or a.exp, "change": f"policy retune on {a.exp} OOF (t, margin, S3 t)",
                       "folds": "0-2", **summ, "threshold": f"{tb:.3f}" + (f"/S3 {best3[0]}" if best3[0] else ""), "margin": mb})


if __name__ == "__main__":
    main()
