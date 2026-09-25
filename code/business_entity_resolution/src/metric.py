"""Official metric: per-S1 F0.5 (and F1, the dev metric), macro-averaged over all evaluated S1 (singletons included)."""
import numpy as np


def pair_key(s1_row, q_src, q_row):
    """Unique int64 per (S1, S2/S3 record) pair. Built in place: at most two int64 arrays alive."""
    k = np.array(s1_row, np.int64); k <<= 34
    t = np.array(q_src, np.int64); t <<= 32; k |= t; del t
    k |= np.asarray(q_row)
    return k


def macro_f05(eval_s1, pred, gt):
    """eval_s1: S1 rows evaluated; pred/gt: dicts with s1_row, q_src, q_row arrays (any S1).

    Returns dict with macro-F0.5 plus diagnostics. Rules from the problem statement:
    no true matches -> 1.0 if prediction empty else 0.0.
    """
    n = int(eval_s1.max()) + 1 if len(eval_s1) else 0
    ev = np.zeros(max(n, 1), bool); ev[eval_s1] = True

    def sel(d):
        m = d["s1_row"] < n
        m[m] = ev[d["s1_row"][m]]
        return {k: v[m] for k, v in d.items()}

    p, g = sel(pred), sel(gt)
    pk, gk = pair_key(p["s1_row"], p["q_src"], p["q_row"]), pair_key(g["s1_row"], g["q_src"], g["q_row"])
    hit = np.isin(pk, gk)
    npred = np.bincount(p["s1_row"], minlength=n)[eval_s1].astype(float)
    ntrue = np.bincount(g["s1_row"], minlength=n)[eval_s1].astype(float)
    tp = np.bincount(p["s1_row"][hit], minlength=n)[eval_s1].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(npred > 0, tp / npred, 0.0)
        R = np.where(ntrue > 0, tp / ntrue, 0.0)
        f = np.where(tp > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
        f1 = np.where(tp > 0, 2 * P * R / (P + R), 0.0)
    single = ntrue == 0
    f = np.where(single, (npred == 0).astype(float), f)
    f1 = np.where(single, (npred == 0).astype(float), f1)
    return {
        "macro_f05": float(f.mean()), "macro_f1": float(f1.mean()), "n_s1": int(len(eval_s1)),
        "pair_precision": float(tp.sum() / max(npred.sum(), 1)), "pair_recall": float(tp.sum() / max(ntrue.sum(), 1)),
        "false_merges": int(npred.sum() - tp.sum()),
        "singleton_acc": float((npred[single] == 0).mean()) if single.any() else float("nan"),
        "nonsingleton_f05": float(f[~single].mean()) if (~single).any() else float("nan"),
    }


if __name__ == "__main__":  # the worked example from the problem statement: F0.5 = 0.714
    gt = {"s1_row": np.array([0, 0]), "q_src": np.array([2, 3]), "q_row": np.array([47, 812])}
    pr = {"s1_row": np.array([0, 0, 0]), "q_src": np.array([2, 2, 3]), "q_row": np.array([47, 193, 812])}
    r = macro_f05(np.array([0, 1]), pr, gt)  # S1 #1 is a correctly predicted singleton -> 1.0
    assert abs(r["macro_f05"] - (0.7142857 + 1) / 2) < 1e-6, r
    r = macro_f05(np.array([1]), {"s1_row": np.array([1]), "q_src": np.array([2]), "q_row": np.array([5])}, gt)
    assert r["macro_f05"] == 0.0  # predicting anything for a singleton scores 0
    print("metric self-test PASS", r)
