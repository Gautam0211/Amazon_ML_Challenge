"""One-time holdout report: frozen experiment + policy on untouched folds 3-4.

dev.py scores fold 3-4 pairs with its dev-fold models (trained on folds 0-2 only), so the
experiment's oof.parquet already holds unbiased holdout scores.

Usage: python src/holdout.py --exp E007
"""
import argparse
import json

import numpy as np
import pyarrow.parquet as pq

from common import log, peak_gb, wpath
from exp import load_gt
from metric import macro_f05
from model import assign, best_per_query


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    a = ap.parse_args()
    pol = json.load(open(wpath("exp", a.exp, "policy.json")))
    folds = np.load(wpath("train", "folds.npy"))
    t = pq.read_table(wpath("exp", a.exp, "oof.parquet"))
    s1 = t["s1_row"].to_numpy()
    log(f"pairs={len(s1)} holdout-fold pairs={(folds[s1] >= 3).sum()} policy={pol}")
    bq = best_per_query(*(t[c].to_numpy() for c in ("s1_row", "q_src", "q_row", "p")))
    del t
    pred, gd = assign(bq, pol["t"], pol["margin"], pol.get("t3")), load_gt()
    out = {name: macro_f05(np.nonzero(m)[0], pred, gd)
           for name, m in (("fold3", folds == 3), ("fold4", folds == 4), ("folds3-4", folds >= 3))}
    for k, v in out.items():
        log(f"HOLDOUT {k}: F05={v['macro_f05']:.5f} F1={v['macro_f1']:.5f} P={v['pair_precision']:.5f} "
            f"R={v['pair_recall']:.5f} singleton_acc={v['singleton_acc']:.4f} false_merges={v['false_merges']}")
    json.dump({"policy": pol, "holdout": out, "peak_gb": peak_gb()}, open(wpath("exp", a.exp, "holdout.json"), "w"), indent=1)
    log(f"peak={peak_gb():.2f}G")


if __name__ == "__main__":
    main()
