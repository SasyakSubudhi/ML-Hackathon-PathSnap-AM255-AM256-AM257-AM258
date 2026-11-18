"""Bayesian hyperparameter optimisation (Optuna TPE) for LightGBM candidate scorer.

* objective   : mean 3-fold GroupKFold (by trip) CV macro-F1 on the 78 clean training trips
* partitions  : training trips only for the search; student_test only for the final tuned-vs-untuned comparison
* outputs     : results/tuning/*  and out-of-fold probabilities used later to tune the HMM decoder
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
import pathsnap_data as P
import train_models as T

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "tuning"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "data_cache"
SEED, N_TRIALS = 42, 25
F = P.FEATURES

UNTUNED = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8)          # identical to model M6 in the comparison


def make(params):
    return LGBMClassifier(**params, n_jobs=2, random_state=SEED, verbose=-1)


def load_clean():
    tr_c, tr_o = T.load_part("train")
    te_c, te_o = T.load_part("student_test")
    for c, o in ((tr_c, tr_o), (te_c, te_o)):
        c["ref_in"] = c["key"].map(o.set_index("key")["ref_in_candidates"]).to_numpy()
    med = tr_o.groupby("trip")["d_ref"].median()
    bad = sorted(med[med > T.ANOMALY_MEDIAN_DREF_M].index)
    keep = sorted(set(tr_o["trip"]) - set(bad))
    tr_c = tr_c[tr_c["trip"].isin(keep)].reset_index(drop=True)
    tr_o = tr_o[tr_o["trip"].isin(keep)].reset_index(drop=True)
    return tr_c, tr_o, te_c, te_o, keep, bad


def quick_metrics(y, key, p, n_obs):
    mf1 = f1_score(y, (p >= 0.5).astype(int), average="macro")
    order = np.lexsort((-p, key))
    ks = key[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    return mf1, y[order][first].sum() / n_obs


def main():
    t0 = time.time()
    tr_c, tr_o, te_c, te_o, keep, bad = load_clean()
    folds = list(GroupKFold(n_splits=T.N_FOLDS).split(tr_c, groups=tr_c["trip"].to_numpy()))
    fold_data = []
    for k, (tri, vai) in enumerate(folds):
        fit_df = T.sample_fit_rows(tr_c.iloc[tri], np.random.default_rng(SEED + k))
        val_df = tr_c.iloc[vai]
        n_obs = int(tr_o["trip"].isin(set(val_df["trip"])).sum())
        fold_data.append(dict(Xf=fit_df[F].to_numpy(np.float32), yf=fit_df["y"].to_numpy(),
                              Xv=val_df[F].to_numpy(np.float32), yv=val_df["y"].to_numpy(),
                              kv=val_df["key"].to_numpy(), n_obs=n_obs, vai=vai))
    print(f"data ready {time.time()-t0:.0f}s, clean train trips {len(keep)}, excluded {bad}", flush=True)

    def cv_eval(params, trial=None):
        f1s, t1s, tsec = [], [], 0.0
        for k, d in enumerate(fold_data):
            s = time.time()
            m = make(params).fit(d["Xf"], d["yf"])
            p = m.predict_proba(d["Xv"])[:, 1]
            tsec += time.time() - s
            mf1, t1 = quick_metrics(d["yv"], d["kv"], p, d["n_obs"])
            f1s.append(mf1); t1s.append(t1)
            if trial is not None:
                trial.report(float(np.mean(f1s)), k)
                if trial.should_prune():
                    trial.set_user_attr("folds_done", k + 1)
                    raise optuna.TrialPruned()
        return np.array(f1s), np.array(t1s), tsec

    # ---- untuned reference under the identical protocol (same folds, same fit samples)
    f1u, t1u, tu = cv_eval(UNTUNED)
    print(f"UNTUNED CV macro-F1 {f1u.mean():.4f}+-{f1u.std(ddof=1):.4f}  top1 {t1u.mean():.4f}  ({tu:.0f}s)", flush=True)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 14),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )
        f1s, t1s, tsec = cv_eval(params, trial)
        trial.set_user_attr("cv_top1", float(t1s.mean()))
        trial.set_user_attr("cv_f1_std", float(f1s.std(ddof=1)))
        trial.set_user_attr("seconds", float(tsec))
        print(f"trial {trial.number}: F1 {f1s.mean():.4f} top1 {t1s.mean():.4f} ({tsec:.0f}s) {params}", flush=True)
        return float(f1s.mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0))
    study.enqueue_trial(dict(n_estimators=300, learning_rate=0.05, num_leaves=63, max_depth=14, min_child_samples=20,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1e-3, reg_alpha=1e-3))  # warm start near default
    t_s = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    search_seconds = time.time() - t_s
    trials = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs", "duration"))
    trials.to_csv(OUT / "optuna_trials.csv", index=False)
    best = dict(study.best_params); best["subsample_freq"] = 1
    json.dump(dict(best_params=best, best_cv_macro_f1=study.best_value, n_trials=len(study.trials),
                   n_complete=sum(t.state.name == "COMPLETE" for t in study.trials),
                   n_pruned=sum(t.state.name == "PRUNED" for t in study.trials), search_seconds=search_seconds,
                   sampler="TPE (n_startup_trials=8, seed=42)", pruner="MedianPruner", objective="mean 3-fold GroupKFold CV macro-F1",
                   untuned_params=UNTUNED), open(OUT / "best_params.json", "w"), indent=2)
    try:
        imp = optuna.importance.get_param_importances(study)
        pd.Series(imp).to_csv(OUT / "param_importance.csv", header=["importance"])
    except Exception as e:
        print("importance skipped", e)
    comp = trials[trials.state == "COMPLETE"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(trials.number, trials.value, "o", c="tab:blue", label="trial value (pruned = NaN)")
    ax[0].plot(comp.number, comp.value.cummax(), "-", c="tab:red", label="best so far")
    ax[0].axhline(f1u.mean(), ls="--", c="gray", label="untuned config (same protocol)")
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("CV macro-F1"); ax[0].legend(fontsize=8); ax[0].set_title("Optuna optimisation history")
    if (OUT / "param_importance.csv").exists():
        pi = pd.read_csv(OUT / "param_importance.csv", index_col=0)["importance"].sort_values()
        pi.plot.barh(ax=ax[1], color="tab:green"); ax[1].set_title("Hyperparameter importance (fANOVA)")
    plt.tight_layout(); plt.savefig(OUT / "plot_optuna_history.png", dpi=140); plt.close()

    # ---- refit tuned and untuned the same way, out-of-fold probabilities, final evaluation on student_test
    f1b, t1b, _ = cv_eval(best)
    oof = np.zeros(len(tr_c))
    for k, d in enumerate(fold_data):
        m = make(best).fit(d["Xf"], d["yf"])
        oof[d["vai"]] = m.predict_proba(d["Xv"])[:, 1]
    tr_c[["key", "trip", "obs", "arc"]].assign(p=oof).to_pickle(CACHE / "train_oof_tuned.pkl")

    fit_df = T.sample_fit_rows(tr_c, np.random.default_rng(SEED))
    Xf, yf = fit_df[F].to_numpy(np.float32), fit_df["y"].to_numpy()
    Xte = te_c[F].to_numpy(np.float32)
    adj_te = T.load_adjacency("student_test", sorted(te_o["trip"].unique()))
    rows, probs = [], {}
    for name, params, cvf1, cvt1 in (("untuned", UNTUNED, f1u, t1u), ("tuned", best, f1b, t1b)):
        s = time.time(); m = make(params).fit(Xf, yf); fit_s = time.time() - s
        ts = []
        for _ in range(3):
            s = time.time(); p = m.predict_proba(Xte)[:, 1]; ts.append(time.time() - s)
        r = T.evaluate(te_c, p, te_o, adj_te)
        r.update(variant=name, cv_macro_f1=cvf1.mean(), cv_macro_f1_std=cvf1.std(ddof=1), cv_top1=cvt1.mean(), fit_seconds=fit_s,
                 infer_ms_per_1000_obs=float(np.median(ts)) / len(te_o) * 1e6,
                 train_sample_macro_f1=f1_score(yf, (m.predict_proba(Xf)[:, 1] >= 0.5).astype(int), average="macro"))
        rows.append(r)
        probs[name] = p
        joblib.dump(m, ROOT / "results" / "models" / f"LightGBM_{name}.joblib")
        print(name, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
    pd.DataFrame(rows).set_index("variant").to_csv(OUT / "tuned_vs_untuned.csv")
    te_c[["key", "trip", "obs", "arc"]].assign(p_untuned=probs["untuned"], p_tuned=probs["tuned"]).to_pickle(CACHE / "student_test_probs.pkl")
    print("TUNING DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
