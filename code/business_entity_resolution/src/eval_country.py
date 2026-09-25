"""Per-country Validation breakdown from saved OOF predictions (no retraining).

For each S1 country: holdout (folds 3-4) macro-F0.5 with the frozen global policy,
the blocking ceiling, and the score when the threshold is re-tuned for that country on folds 0-2.
Writes work/train/validation_country.json.

Usage: python src/eval_country.py
"""
import json

import numpy as np
import pyarrow.parquet as pq

from common import log, wpath
from metric import macro_f05, pair_key
from model import ID_COLS, assign, best_per_query, gt_keys, is_pos


def main():
    pol = json.load(open(wpath("models", "policy.json")))
    folds = np.load(wpath("train", "folds.npy"))
    ctry = pq.read_table(wpath("train", "s1.parquet"), columns=["country"])["country"].to_numpy()
    oof = pq.read_table(wpath("train", "oof.parquet"))
    s1, qs, qr, p = (oof[c].to_numpy() for c in ("s1_row", "q_src", "q_row", "p"))
    del oof
    g = pq.read_table(wpath("train", "gt.parquet"))
    gd = {c: g[c].to_numpy() for c in ID_COLS}
    pos = is_pos(pair_key(s1, qs, qr), gt_keys())
    perfect = {"s1_row": s1[pos], "q_src": qs[pos], "q_row": qr[pos]}
    bq = best_per_query(s1, qs, qr, p)
    del s1, qs, qr, p, pos
    log("OOF loaded and reduced per query")

    rows = np.arange(len(folds))
    frozen = assign(bq, pol["t"], pol["margin"])
    out = {"policy": pol["t"], "margin": pol["margin"], "countries": {}}
    for c in ["ALL"] + sorted(set(ctry.tolist())):
        m = np.ones(len(folds), bool) if c == "ALL" else ctry == c
        tune, hold = rows[m & (folds <= 2)], rows[m & (folds >= 3)]
        r = macro_f05(hold, frozen, gd)
        ceil = macro_f05(hold, perfect, gd)["macro_f05"]
        grid = [(t, mg) for t in np.arange(0.30, 0.91, 0.02) for mg in (0.0, 0.1, 0.2, 0.3)]
        bt, bm = max(grid, key=lambda x: macro_f05(tune, assign(bq, x[0], x[1]), gd)["macro_f05"])
        rt = macro_f05(hold, assign(bq, bt, bm), gd)
        out["countries"][c] = {"n_s1_holdout": int(len(hold)), "frozen": r, "ceiling": ceil,
                               "retuned_t": float(bt), "retuned_margin": bm, "retuned_macro_f05": rt["macro_f05"]}
        log(f"{c:>6} n={len(hold):>7} macroF05={r['macro_f05']:.4f} ceil={ceil:.4f} P={r['pair_precision']:.4f} "
            f"R={r['pair_recall']:.4f} single_acc={r['singleton_acc']:.4f} nonsingle={r['nonsingleton_f05']:.4f} "
            f"| retuned t={bt:.2f} m={bm} -> {rt['macro_f05']:.4f}")
    json.dump(out, open(wpath("train", "validation_country.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
