"""Tune the HMM decoder parameters and compare classifier-only vs classifier + HMM/Viterbi.

Decoder parameters are tuned ONLY on out-of-fold classifier probabilities of training trips
(a random subset of the clean training trips). student_test is used only for reporting the comparison.
Outputs: results/hmm/*, data_cache/student_test_hmm_pred.pkl
"""
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import hmm_decoder as H
import pathsnap_data as P
import train_models as T

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT = Path(__file__).resolve().parent.parent
CACHE, OUT = ROOT / "data_cache", ROOT / "results" / "hmm"
OUT.mkdir(parents=True, exist_ok=True)
SEED, N_TRIALS, N_TUNE_TRIPS = 42, 60, 26
DEFAULT = dict(alpha=1.0, beta=10.0, uturn_pen=3.0, switch_pen=0.0, floor=12.0, gap_s=30.0, normalize=True)


def prepare_part(part, cand, probs, obs, label):
    """-> dict trip -> dict(prep, p, graph, route, ref, n_obs)"""
    out, t0 = {}, time.time()
    trips = sorted(cand["trip"].unique())
    cg = cand.groupby("trip", sort=True).indices
    og = obs.groupby("trip", sort=True).indices
    for i, trip in enumerate(trips):
        tr = P.project_trip(P.load_trip(part, trip))
        g = H.TripGraph(tr["arcs"], tr["node_xy"])
        ci, oi = cg[trip], og[trip]
        c = cand.iloc[ci]
        prep = H.Prepared(g, c["obs"].to_numpy(), c["arc"].to_numpy(), c["frac"].to_numpy(),
                          tr["obs_xy"][:, 0], tr["obs_xy"][:, 1], tr["t"])
        g._cache.clear()
        out[trip] = dict(prep=prep, p=probs[ci], graph=g, route=tr["route"], ref=obs.iloc[oi]["ref_arc"].to_numpy(),
                         t=obs.iloc[oi]["t"].to_numpy(), n_obs=len(oi))
        if (i + 1) % 10 == 0:
            print(f"  {label}: prepared {i+1}/{len(trips)} trips ({time.time()-t0:.0f}s)", flush=True)
    return out


def evaluate(data, params=None, mode="hmm", adj=None, extra=False):
    """mode: 'clf' (unfilled top-1), 'clf_fill', 'hmm'. Returns metrics dict (+ per-trip frame and predictions)."""
    rows, preds = [], {}
    for trip, d in data.items():
        if mode == "clf":
            pred = d["prep"].classifier_only(d["p"])
        elif mode == "clf_fill":
            pred = d["prep"].classifier_only(d["p"], fill=True)
        else:
            pred = d["prep"].decode(d["p"], **params)
        preds[trip] = pred
        r = H.route_metrics(pred, d["route"], d["graph"])
        r.update(trip=trip, n_obs=d["n_obs"], correct=int((pred == d["ref"]).sum()))
        rows.append(r)
    df = pd.DataFrame(rows)
    n = df.n_obs.sum()
    res = dict(top1=df.correct.sum() / n, top1_trip_mean=(df.correct / df.n_obs).mean(),
               route_precision=df.route_precision.mean(), route_recall=df.route_recall.mean(), route_f1=df.route_f1.mean(),
               route_connected=df.route_connected.mean())
    if adj is not None:
        obs = pd.concat([pd.DataFrame({"trip": t, "t": d["t"]}) for t, d in data.items()], ignore_index=True)
        res["continuity"] = T.continuity(preds, obs, adj)
    return (res, df, preds) if extra else res


def main():
    t0 = time.time()
    # ------------------------------------------------------------ training trips (out-of-fold probabilities)
    oof = pd.read_pickle(CACHE / "train_oof_tuned.pkl")
    cand, obs = T.load_part("train")
    cand = cand[["key", "trip", "obs", "arc", "frac"]]
    keep = sorted(oof["trip"].unique())
    cand = cand[cand["trip"].isin(keep)].reset_index(drop=True)
    obs = obs[obs["trip"].isin(keep)].reset_index(drop=True)
    assert (cand["key"].to_numpy() == oof["key"].to_numpy()).all() and (cand["arc"].to_numpy() == oof["arc"].to_numpy()).all()
    train = prepare_part("train", cand, oof["p"].to_numpy(), obs, "train")
    print(f"train prepared {time.time()-t0:.0f}s", flush=True)

    rng = np.random.default_rng(SEED)
    tune_trips = sorted(rng.choice(keep, N_TUNE_TRIPS, replace=False).tolist())
    tune = {t: train[t] for t in tune_trips}
    print("tuning subset:", N_TUNE_TRIPS, "trips,", sum(d["n_obs"] for d in tune.values()), "observations", flush=True)

    def objective(trial):
        prm = dict(alpha=trial.suggest_float("alpha", 0.3, 3.0, log=True), beta=trial.suggest_float("beta", 1.0, 60.0, log=True),
                   uturn_pen=trial.suggest_float("uturn_pen", 0.0, 12.0), switch_pen=trial.suggest_float("switch_pen", 0.0, 3.0),
                   floor=trial.suggest_float("floor", 3.0, 30.0), gap_s=trial.suggest_categorical("gap_s", [15.0, 30.0, 60.0, 120.0]),
                   normalize=trial.suggest_categorical("normalize", [True, False]))
        return evaluate(tune, prm)["top1"]

    base_tune = evaluate(tune, mode="clf")["top1"]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=15))
    study.enqueue_trial(dict(alpha=1.0, beta=10.0, uturn_pen=3.0, switch_pen=0.0, floor=12.0, gap_s=30.0, normalize=True))
    study.optimize(objective, n_trials=N_TRIALS)
    best = dict(study.best_params)
    print(f"decoder tuning done {time.time()-t0:.0f}s | tune-subset classifier-only {base_tune:.4f}, "
          f"default HMM {study.trials[0].value:.4f}, best HMM {study.best_value:.4f}\n{best}", flush=True)
    study.trials_dataframe(attrs=("number", "value", "params", "state")).to_csv(OUT / "decoder_tuning_trials.csv", index=False)
    json.dump(dict(best_params=best, default_params=DEFAULT, objective="pointwise Top-1 on out-of-fold probabilities (26 train trips)",
                   n_trials=N_TRIALS, tune_trips=tune_trips, max_dist_m=H.MAX_DIST), open(OUT / "decoder_params.json", "w"), indent=2)

    # ------------------------------------------------------------ comparison tables
    adj_tr = T.load_adjacency("train", keep)
    rows = []
    for name, mode, prm in (("Classifier only (no-candidate points unassigned)", "clf", None),
                            ("Classifier only (no-candidate points filled)", "clf_fill", None),
                            ("Classifier + HMM (default decoder parameters)", "hmm", DEFAULT),
                            ("Classifier + HMM (tuned decoder parameters)", "hmm", best)):
        r = evaluate(train, prm, mode, adj_tr)
        r["method"] = name; r["partition"] = "train, out-of-fold (78 trips)"
        rows.append(r)
    del train

    te_c, te_o = T.load_part("student_test")
    te_c = te_c[["key", "trip", "obs", "arc", "frac"]]
    pr = pd.read_pickle(CACHE / "student_test_probs.pkl")
    assert (te_c["key"].to_numpy() == pr["key"].to_numpy()).all()
    adj_te = T.load_adjacency("student_test", sorted(te_o["trip"].unique()))
    detail = {}
    for tag, col in (("tuned LightGBM", "p_tuned"), ("untuned LightGBM", "p_untuned")):
        test = prepare_part("student_test", te_c, pr[col].to_numpy(), te_o, f"test {tag}")
        variants = [("Classifier only (no-candidate points unassigned)", "clf", None),
                    ("Classifier only (no-candidate points filled)", "clf_fill", None),
                    ("Classifier + HMM (default decoder parameters)", "hmm", DEFAULT),
                    ("Classifier + HMM (tuned decoder parameters)", "hmm", best)]
        if tag == "tuned LightGBM":
            variants += [("Ablation: tuned HMM without U-turn penalty", "hmm", {**best, "uturn_pen": 0.0}),
                         ("Ablation: tuned HMM without gap restart", "hmm", {**best, "gap_s": 1e9}),
                         ("Ablation: tuned HMM with raw (unnormalised) emissions", "hmm", {**best, "normalize": False}),
                         ("Ablation: tuned HMM with no switch/U-turn/floor shaping", "hmm", {**best, "uturn_pen": 0.0, "switch_pen": 0.0})]
        for name, mode, prm in variants:
            r, per_trip, preds = evaluate(test, prm, mode, adj_te, extra=True)
            r["method"] = name; r["partition"] = f"student_test (10 trips), {tag}"
            rows.append(r)
            detail[(tag, name)] = (per_trip, preds, test)
    comp = pd.DataFrame(rows)[["partition", "method", "top1", "top1_trip_mean", "continuity", "route_precision", "route_recall", "route_f1", "route_connected"]]
    comp.to_csv(OUT / "hmm_comparison.csv", index=False)
    print(comp.round(4).to_string(), flush=True)

    # ------------------------------------------------------------ per-trip table + error analysis (tuned model)
    key_clf, key_hmm = ("tuned LightGBM", "Classifier only (no-candidate points filled)"), ("tuned LightGBM", "Classifier + HMM (tuned decoder parameters)")
    pt_c, preds_c, test = detail[key_clf]
    pt_h, preds_h, _ = detail[key_hmm]
    per = pt_c[["trip", "n_obs", "correct", "route_f1"]].merge(pt_h[["trip", "correct", "route_f1", "route_connected"]], on="trip", suffixes=("_clf", "_hmm"))
    per["top1_clf"], per["top1_hmm"] = per.correct_clf / per.n_obs, per.correct_hmm / per.n_obs
    per["delta_top1"] = per.top1_hmm - per.top1_clf
    per.to_csv(OUT / "per_trip_student_test.csv", index=False)
    err = {"classifier only": dict(wrong_direction=0, adjacent_arc=0, other=0), "classifier + HMM": dict(wrong_direction=0, adjacent_arc=0, other=0)}
    for label, preds in (("classifier only", preds_c), ("classifier + HMM", preds_h)):
        for trip, d in test.items():
            g, ref, pred = d["graph"], d["ref"], preds[trip]
            w = pred != ref
            a, r = pred[w], ref[w]
            rev = (g.frm[a] == g.to[r]) & (g.to[a] == g.frm[r])
            adjc = ((g.to[a] == g.frm[r]) | (g.to[r] == g.frm[a])) & ~rev
            err[label]["wrong_direction"] += int(rev.sum()); err[label]["adjacent_arc"] += int(adjc.sum())
            err[label]["other"] += int((~rev & ~adjc).sum())
    pd.DataFrame(err).T.to_csv(OUT / "error_breakdown_student_test.csv")
    print(per.round(4).to_string(), "\n", pd.DataFrame(err).T, flush=True)
    pd.to_pickle({t: dict(clf=preds_c[t], hmm=preds_h[t]) for t in preds_c}, CACHE / "student_test_hmm_pred.pkl")

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(per)); w = 0.38
    ax.bar(x - w / 2, per.top1_clf, w, label="classifier only", color="tab:orange")
    ax.bar(x + w / 2, per.top1_hmm, w, label="classifier + HMM", color="tab:green")
    ax.set_xticks(x); ax.set_xticklabels(per.trip, rotation=45, fontsize=8); ax.set_ylim(0.6, 1.0); ax.set_ylabel("Top-1 accuracy")
    ax.set_title("student_test: per-trip accuracy, classifier only vs classifier + HMM"); ax.legend()
    plt.tight_layout(); plt.savefig(OUT / "plot_hmm_per_trip.png", dpi=140); plt.close()
    tr = pd.read_csv(OUT / "decoder_tuning_trials.csv")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(tr.number, tr.value, "o", ms=3); ax.plot(tr.number, tr.value.cummax(), "-r"); ax.axhline(base_tune, ls="--", c="gray", label="classifier only (same trips)")
    ax.set_xlabel("trial"); ax.set_ylabel("Top-1 (26 train trips, out-of-fold)"); ax.legend(); ax.set_title("HMM decoder parameter search")
    plt.tight_layout(); plt.savefig(OUT / "plot_decoder_tuning.png", dpi=140); plt.close()
    print("HMM DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
