"""Phase 5 Optuna search over LightGBM params, objective = dev macro F0.5 (cheap proxy).

Train: dev folds 1-2, S1 sample --frac (default 0.1 = 25% of the E006 0.4 sample), same hard-negative
sampling as dev.py. Eval: every candidate of a fixed half of fold-0 S1 (held in memory); assignment
uses only those candidates, so this is a screening proxy -- confirm the winner with dev.py on 3 folds.
min_data_in_leaf is searched at this data scale; multiply by 1/scale (~4x) for full data.

Usage: python src/tune.py --study T1 --timeout 9000
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import optuna

from common import log, peak_gb, wpath
from dev import Assembler, load_cand, sample_rows
from exp import load_gt
from metric import macro_f05, pair_key
from model import PARAMS, assign, best_per_query, gt_keys, is_pos

E006 = dict(learning_rate=0.08, num_leaves=127, min_data_in_leaf=25, feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=1.0)
T_GRID = np.round(np.arange(0.40, 0.86, 0.05), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="T1")
    ap.add_argument("--cand", default="cand_adB.parquet")
    ap.add_argument("--groups", default="str,ps")
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--eval_frac", type=float, default=0.25)
    ap.add_argument("--neg_keep", type=float, default=0.3)
    ap.add_argument("--timeout", type=int, default=9000)
    ap.add_argument("--max_rounds", type=int, default=1500)
    a = ap.parse_args()

    folds = np.load(wpath("train", "folds.npy"))
    gd, gk = load_gt(), gt_keys()
    cand = load_cand(a.cand)
    s1, qs, qr = cand["s1_row"], cand["q_src"], cand["q_row"]
    y = is_pos(pair_key(s1, qs, qr), gk)
    asm = Assembler(cand, groups=a.groups.split(","))
    rng = np.random.default_rng(0)
    s1_in = (rng.random(len(folds)) < a.frac) & (folds >= 1)  # train on folds 1-2 only (sample_rows caps at fold 2)
    samp, wt = sample_rows(asm, y, folds, s1_in, a.neg_keep)
    ev_s1 = np.nonzero((folds == 0) & (np.random.default_rng(1).random(len(folds)) < a.eval_frac))[0]
    ev_in = np.zeros(len(folds), bool); ev_in[ev_s1] = True
    ev = np.nonzero(ev_in[s1])[0]
    log(f"train rows={len(samp)} pos={y[samp].mean():.4f} | eval S1={len(ev_s1)} rows={len(ev)}")
    Xe = asm.rows(ev)
    ye = y[ev].astype(np.float32)
    dtr = lgb.Dataset(asm.rows(samp), y[samp].astype(np.float32), weight=wt, feature_name=asm.names,
                      free_raw_data=True, params={"verbose": -1, "max_bin": 255, "feature_pre_filter": False}).construct()
    dva = lgb.Dataset(Xe, ye, reference=dtr, free_raw_data=False).construct()
    es1, eqs, eqr = s1[ev], qs[ev], qr[ev]
    del asm, cand, s1, qs, qr, y, samp, wt  # only the binned datasets, Xe and eval ids are needed from here
    log(f"datasets built, peak={peak_gb():.2f}G")

    def objective(trial):
        p = dict(PARAMS)
        p.update(learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 31, 511, log=True),
                 min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 400, log=True),
                 feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
                 bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                 lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 30, log=True))

        def prune_cb(env):
            if env.iteration % 25 == 24:
                trial.report(env.evaluation_result_list[0][2], env.iteration)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        m = lgb.train(p, dtr, a.max_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), prune_cb])
        pr = m.predict(Xe, num_iteration=m.best_iteration, num_threads=12)
        bq = best_per_query(es1, eqs, eqr, pr)
        res = max(((macro_f05(ev_s1, assign(bq, t, 0.2), gd), t) for t in T_GRID), key=lambda r: r[0]["macro_f05"])
        r, t = res
        for k, v in (("t", t), ("best_iter", m.best_iteration), ("f1", r["macro_f1"]), ("precision", r["pair_precision"]),
                     ("recall", r["pair_recall"]), ("peak_gb", peak_gb())):
            trial.set_user_attr(k, float(v))
        log(f"trial {trial.number}: F05={r['macro_f05']:.5f} P={r['pair_precision']:.5f} R={r['pair_recall']:.5f} "
            f"t={t} iters={m.best_iteration} peak={peak_gb():.2f}G {json.dumps(trial.params)}")
        return r["macro_f05"]

    st = optuna.create_study(study_name=a.study, direction="maximize", load_if_exists=True,
                             storage=f"sqlite:///{wpath('exp', 'optuna.db')}".replace("\\", "/"),
                             sampler=optuna.samplers.TPESampler(seed=7),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=300))
    if not st.trials:
        st.enqueue_trial(E006)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st.optimize(objective, timeout=a.timeout, gc_after_trial=True)
    done = [t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE]
    log(f"trials complete={len(done)} pruned={sum(t.state == optuna.trial.TrialState.PRUNED for t in st.trials)}")
    for t in sorted(done, key=lambda t: -t.value)[:8]:
        log(f"  #{t.number} F05={t.value:.5f} {json.dumps(t.params)} {json.dumps(t.user_attrs)}")
    log(f"trial0 (E006) F05={st.trials[0].value}  TUNE_DONE")


if __name__ == "__main__":
    main()
