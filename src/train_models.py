"""Review-1 model comparison for PathSnap candidate scoring.

Task: for every (GPS observation, candidate arc) row predict P(candidate == reference arc).
Evaluation:
  * 3-fold GroupKFold CV on the training trips (a trip never spans folds)
  * final fit on all (clean) training trips, evaluated on the 10 student_test trips
Nothing from the evaluator-testing partition is used anywhere.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
import pathsnap_data as P

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
CACHE, OUT = ROOT / "data_cache", ROOT / "results" / "models"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
MAX_TRAIN_ROWS = 200_000          # rows used to FIT any model (same cap for every model)
N_FOLDS = 3
ANOMALY_MEDIAN_DREF_M = 50.0      # trips whose reference route is >50 m from GPS (median) are excluded from fitting


# ------------------------------------------------------------------ helpers
def load_part(part):
    c = pd.read_pickle(CACHE / f"{part}_cand.pkl")
    o = pd.read_pickle(CACHE / f"{part}_obs.pkl")
    c["key"] = c["trip"].astype(np.int64) * 1_000_000 + c["obs"]
    o["key"] = o["trip"].astype(np.int64) * 1_000_000 + o["obs"]
    return c, o


def load_adjacency(part, trips):
    adj = {}
    for t in trips:
        arcs = pd.read_csv(P.DATA_ROOT_DEFAULT / part / t / f"{t}.arcs", sep=r"\s+", header=None).to_numpy()
        n = int(arcs.max()) + 2
        order = np.argsort(arcs[:, 0], kind="stable")
        indptr = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(arcs[:, 0], minlength=n), out=indptr[1:])
        adj[t] = (arcs[:, 0], arcs[:, 1], indptr, order)
    return adj


def continuity(pred_by_trip, obs, adj, max_dt=5.0):
    """Share of consecutive observations (dt<=max_dt s) whose predicted arcs are the same arc or
    reachable within 2 forward hops. Higher = more physically coherent route."""
    ok_total, n_total = 0, 0
    for trip, g in obs.groupby("trip"):
        pred = pred_by_trip.get(trip)
        if pred is None:
            continue
        frm, to, indptr, order = adj[trip]
        t = g["t"].to_numpy()
        a, b = pred[:-1], pred[1:]
        elig = (a >= 0) & (b >= 0) & (np.diff(t) <= max_dt)
        ia = np.where(elig)[0]
        aa, bb = a[ia], b[ia]
        ok = (aa == bb) | (to[aa] == frm[bb])
        for k in np.where(~ok)[0]:                      # 2-hop check only for the remainder
            node = to[aa[k]]
            for m in order[indptr[node]:indptr[node + 1]]:
                if to[m] == frm[bb[k]]:
                    ok[k] = True
                    break
        ok_total += int(ok.sum())
        n_total += len(ia)
    return ok_total / max(n_total, 1)


def top1_predictions(cand, p):
    tmp = pd.DataFrame({"key": cand["key"].to_numpy(), "trip": cand["trip"].to_numpy(),
                        "obs": cand["obs"].to_numpy(), "arc": cand["arc"].to_numpy(),
                        "y": cand["y"].to_numpy(), "p": p})
    return tmp.sort_values(["key", "p"], ascending=[True, False]).drop_duplicates("key")


def evaluate(cand, p, obs, adj, thr=0.5, prob_ok=True):
    top = top1_predictions(cand, p)
    pred_by_trip = {}
    for trip, g in obs.groupby("trip"):
        arr = np.full(len(g), -1, np.int64)
        tt = top[top["trip"] == trip]
        arr[tt["obs"].to_numpy()] = tt["arc"].to_numpy()
        pred_by_trip[trip] = arr
    y = cand["y"].to_numpy()
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return dict(
        macro_f1=f1_score(y, yhat, average="macro"),
        f1_pos=f1_score(y, yhat, pos_label=1),
        precision_pos=precision_score(y, yhat, zero_division=0),
        recall_pos=recall_score(y, yhat),
        log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]) if prob_ok else np.nan,
        top1_acc=top["y"].sum() / len(obs),
        route_continuity=continuity(pred_by_trip, obs, adj),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def sample_fit_rows(cand, rng, cap=MAX_TRAIN_ROWS):
    """Fit rows: only observations whose reference arc is among the candidates; sampled by observation."""
    c = cand[cand["ref_in"]]
    if len(c) <= cap:
        return c
    uniq, inv = np.unique(c["key"].to_numpy(), return_inverse=True)
    keep = rng.random(len(uniq)) < cap / len(c)
    return c[keep[inv]]


# ------------------------------------------------------------------ model zoo
def model_zoo():
    return [
        ("M1_LogReg", "Linear / probabilistic", "StandardScaler + LogisticRegression(C=1.0, lbfgs, max_iter=300)",
         lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300, random_state=SEED))),
        ("M2_GaussNB", "Probabilistic (generative)", "GaussianNB(var_smoothing=1e-9)",
         lambda: GaussianNB()),
        ("M3_DecisionTree", "Tree-based", "DecisionTree(max_depth=12, min_samples_leaf=20)",
         lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED)),
        ("M4_RandomForest", "Bagging ensemble", "RandomForest(n_estimators=50, max_depth=14, min_samples_leaf=10)",
         lambda: RandomForestClassifier(n_estimators=50, max_depth=14, min_samples_leaf=10, n_jobs=2, random_state=SEED)),
        ("M5_HistGB", "Boosting (histogram GB, sklearn)", "HistGradientBoosting(max_iter=200, lr=0.1, max_leaf_nodes=31)",
         lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=SEED)),
        ("M6_LightGBM", "Boosting (LightGBM)", "LGBM(n_estimators=300, lr=0.05, num_leaves=63, subsample=0.8, colsample=0.8)",
         lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, n_jobs=2, random_state=SEED, verbose=-1)),
        ("M7_MLP", "Neural network (shallow MLP)", "StandardScaler + MLP(64,32 ReLU, adam, alpha=1e-4, early_stopping)",
         lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512,
                                                              early_stopping=True, n_iter_no_change=5, max_iter=60,
                                                              random_state=SEED))),
    ]


def main():
    t_start = time.time()
    tr_c, tr_o = load_part("train")
    te_c, te_o = load_part("student_test")
    for c, o in ((tr_c, tr_o), (te_c, te_o)):
        ref_in = o.set_index("key")["ref_in_candidates"]
        c["ref_in"] = c["key"].map(ref_in).to_numpy()

    # data-quality filter (documented): reference route far from GPS => unreliable labels
    med = tr_o.groupby("trip")["d_ref"].median()
    bad = sorted(med[med > ANOMALY_MEDIAN_DREF_M].index)
    fit_trips = sorted(set(tr_o["trip"]) - set(bad))
    print("excluded anomalous trips:", bad, "| fit trips:", len(fit_trips), flush=True)
    tr_c = tr_c[tr_c["trip"].isin(fit_trips)].reset_index(drop=True)
    tr_o = tr_o[tr_o["trip"].isin(fit_trips)].reset_index(drop=True)
    F = P.FEATURES
    assert np.isfinite(tr_c[F].to_numpy()).all() and np.isfinite(te_c[F].to_numpy()).all()

    adj_tr = load_adjacency("train", fit_trips)
    adj_te = load_adjacency("student_test", sorted(te_o["trip"].unique()))
    print("data ready %.0fs; train rows %d (obs %d) | test rows %d (obs %d)" %
          (time.time() - t_start, len(tr_c), len(tr_o), len(te_c), len(te_o)), flush=True)

    # ---- sanity baseline: nearest arc (not ML, not counted in the 5 required models)
    base = evaluate(te_c, 1.0 / (1.0 + te_c["d"].to_numpy()), te_o, adj_te, prob_ok=False)
    base_pred_pos = (te_c["rank_d"] == 1).astype(int).to_numpy()
    base["macro_f1"] = f1_score(te_c["y"], base_pred_pos, average="macro")
    base["f1_pos"] = f1_score(te_c["y"], base_pred_pos)
    print("BASELINE nearest-arc student_test:", {k: round(v, 4) for k, v in base.items() if isinstance(v, float)}, flush=True)

    trips_arr = tr_c["trip"].to_numpy()
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(tr_c, groups=trips_arr))
    fold_rows, final_rows, curves = [], [], {}
    rng = np.random.default_rng(SEED)

    for name, family, cfg, make in model_zoo():
        t_model = time.time()
        # ---------------- cross-validation
        for k, (tri, vai) in enumerate(folds):
            fit_df = sample_fit_rows(tr_c.iloc[tri], rng)
            val_df = tr_c.iloc[vai]
            val_obs = tr_o[tr_o["trip"].isin(set(val_df["trip"]))]
            m = make()
            m.fit(fit_df[F].to_numpy(np.float32), fit_df["y"].to_numpy())
            p = m.predict_proba(val_df[F].to_numpy(np.float32))[:, 1]
            r = evaluate(val_df, p, val_obs, adj_tr)
            r.update(model=name, fold=k, n_fit_rows=len(fit_df))
            fold_rows.append(r)
            print(f"  {name} fold{k}: macroF1={r['macro_f1']:.4f} top1={r['top1_acc']:.4f}", flush=True)
        # ---------------- final fit on all clean train trips -> student_test
        fit_df = sample_fit_rows(tr_c, rng)
        Xf, yf = fit_df[F].to_numpy(np.float32), fit_df["y"].to_numpy()
        m = make()
        t0 = time.time()
        if name == "M6_LightGBM":
            m.fit(Xf, yf, eval_set=[(Xf, yf), (te_c[F].to_numpy(np.float32), te_c["y"].to_numpy())],
                  eval_names=["train_sample", "student_test"], eval_metric="binary_logloss")
            curves[name] = {k2: v["binary_logloss"] for k2, v in m.evals_result_.items()}
        else:
            m.fit(Xf, yf)
        fit_s = time.time() - t0
        if name == "M7_MLP":
            mlp = m[-1]
            curves[name] = {"train_loss": list(map(float, mlp.loss_curve_)),
                            "val_accuracy(internal 10% split)": list(map(float, mlp.validation_scores_))}
        Xte = te_c[F].to_numpy(np.float32)
        t0 = time.time()
        p = m.predict_proba(Xte)[:, 1]
        inf_s = time.time() - t0
        r = evaluate(te_c, p, te_o, adj_te)
        ptr = m.predict_proba(Xf)[:, 1]
        r["train_sample_macro_f1"] = f1_score(yf, (ptr >= 0.5).astype(int), average="macro")
        r.update(model=name, family=family, config=cfg, fit_seconds=fit_s, n_fit_rows=len(fit_df),
                 infer_ms_per_1000_obs=inf_s / len(te_o) * 1e6, infer_seconds_total=inf_s)
        final_rows.append(r)
        if name != "M4_RandomForest":
            joblib.dump(m, OUT / f"{name}.joblib")
        print(f"{name} DONE {time.time()-t_model:.0f}s | test macroF1={r['macro_f1']:.4f} top1={r['top1_acc']:.4f} "
              f"logloss={r['log_loss']:.4f} cont={r['route_continuity']:.4f} fit={fit_s:.1f}s infer/1000obs={r['infer_ms_per_1000_obs']:.1f}ms",
              flush=True)
        # progressive save
        pd.DataFrame(fold_rows).to_csv(OUT / "cv_fold_results.csv", index=False)
        pd.DataFrame(final_rows).to_csv(OUT / "student_test_results_raw.csv", index=False)
        (OUT / "learning_curves.json").write_text(json.dumps(curves))

    # consolidated comparison table (same layout as the Deliverables document)
    fr = pd.DataFrame(fold_rows)
    cvm = fr.groupby("model").agg(cv_macro_f1=("macro_f1", "mean"), cv_macro_f1_std=("macro_f1", "std"),
                                  cv_top1=("top1_acc", "mean"), cv_continuity=("route_continuity", "mean"))
    tab = pd.DataFrame(final_rows).set_index("model").join(cvm)
    tab = tab[["family", "config", "cv_macro_f1", "cv_macro_f1_std", "cv_top1", "top1_acc", "route_continuity",
               "macro_f1", "f1_pos", "precision_pos", "recall_pos", "log_loss", "infer_ms_per_1000_obs",
               "fit_seconds", "train_sample_macro_f1", "tn", "fp", "fn", "tp"]]
    tab = tab.rename(columns={"top1_acc": "test_top1", "route_continuity": "test_continuity",
                              "macro_f1": "test_macro_f1", "f1_pos": "test_f1_pos",
                              "precision_pos": "test_precision", "recall_pos": "test_recall",
                              "log_loss": "test_log_loss"})
    tab.to_csv(OUT / "model_comparison.csv")
    json.dump({"baseline_nearest_arc_student_test": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in base.items()},
               "seed": SEED, "max_train_rows_per_fit": MAX_TRAIN_ROWS, "cv_folds": N_FOLDS,
               "excluded_anomalous_train_trips": bad, "features": F,
               "candidate_radius_k": (CACHE / "build_config.txt").read_text()},
              open(OUT / "experiment_config.json", "w"), indent=2)
    print("ALL DONE %.0fs" % (time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
