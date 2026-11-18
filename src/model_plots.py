"""Plots from the REAL experiment outputs in results/models (nothing is synthetic)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "results" / "models"
tab = pd.read_csv(M / "model_comparison.csv", index_col=0)
folds = pd.read_csv(M / "cv_fold_results.csv")
cfg = json.load(open(M / "experiment_config.json"))
base = cfg["baseline_nearest_arc_student_test"]
short = [i.split("_", 1)[1] for i in tab.index]
x = np.arange(len(tab))

# 1) CV macro-F1 (mean +- std across the 3 trip-grouped folds)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(x, tab.cv_macro_f1, yerr=tab.cv_macro_f1_std, capsize=4, color="tab:blue")
for i, v in enumerate(tab.cv_macro_f1):
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(short, rotation=25); ax.set_ylim(0.8, 0.97)
ax.set_ylabel("CV macro-F1 (candidate level)"); ax.set_title("3-fold GroupKFold (by trip) on clean train trips")
plt.tight_layout(); plt.savefig(M / "plot_cv_macro_f1.png", dpi=140); plt.close()

# 2) student-test Top-1 and macro-F1
fig, ax = plt.subplots(figsize=(9, 4.4))
w = 0.38
ax.bar(x - w / 2, tab.test_top1, w, label="Top-1 accuracy (observation level)", color="tab:green")
ax.bar(x + w / 2, tab.test_macro_f1, w, label="Macro-F1 (candidate level)", color="tab:orange")
ax.axhline(base["top1_acc"], ls="--", c="gray", lw=1); ax.text(len(x) - 0.5, base["top1_acc"] + 0.01, "nearest-arc baseline Top-1", ha="right", fontsize=8, color="gray")
for i in range(len(tab)):
    ax.text(i - w / 2, tab.test_top1.iloc[i] + 0.006, f"{tab.test_top1.iloc[i]:.3f}", ha="center", fontsize=7)
    ax.text(i + w / 2, tab.test_macro_f1.iloc[i] + 0.006, f"{tab.test_macro_f1.iloc[i]:.3f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(short, rotation=25); ax.set_ylim(0.5, 1.0); ax.legend(loc="lower left", fontsize=8)
ax.set_title("student_test (10 trips): Top-1 accuracy and macro-F1")
plt.tight_layout(); plt.savefig(M / "plot_student_test_top1_f1.png", dpi=140); plt.close()

# 3) log loss + 4) inference time + 5) route continuity
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].bar(x, tab.test_log_loss, color="tab:red"); ax[0].set_title("student_test log loss (lower = better calibrated)")
ax[1].bar(x, tab.infer_ms_per_1000_obs, color="tab:purple"); ax[1].set_yscale("log"); ax[1].set_title("Inference time (ms per 1000 observations, model only)")
ax[2].bar(x, tab.test_continuity, color="tab:cyan"); ax[2].set_ylim(0.9, 1.0)
ax[2].axhline(base["route_continuity"], ls="--", c="gray"); ax[2].set_title("Route continuity (same arc or <=2 hops), dashed = nearest-arc baseline")
for a in ax:
    a.set_xticks(x); a.set_xticklabels(short, rotation=25)
plt.tight_layout(); plt.savefig(M / "plot_logloss_time_continuity.png", dpi=140); plt.close()

# 6) confusion matrices (candidate level, student_test, threshold 0.5)
fig, axs = plt.subplots(1, len(tab), figsize=(3.0 * len(tab), 3.2))
for a, (name, r) in zip(axs, tab.iterrows()):
    cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
    a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=8, color="black")
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["pred 0", "pred 1"]); a.set_yticklabels(["true 0", "true 1"])
    a.set_title(name.split("_", 1)[1], fontsize=9)
plt.suptitle("Confusion matrices on student_test (colour = row-normalised)")
plt.tight_layout(); plt.savefig(M / "plot_confusion_matrices.png", dpi=140); plt.close()

# 7) overfitting evidence: train-sample vs CV vs student-test macro-F1
fig, ax = plt.subplots(figsize=(8.5, 4))
ax.bar(x - w, tab.train_sample_macro_f1, w, label="train sample", color="tab:gray")
ax.bar(x, tab.cv_macro_f1, w, label="CV (held-out trips)", color="tab:blue")
ax.bar(x + w, tab.test_macro_f1, w, label="student_test", color="tab:orange")
ax.set_xticks(x); ax.set_xticklabels(short, rotation=25); ax.set_ylim(0.8, 1.0); ax.legend(fontsize=8)
ax.set_title("Macro-F1: train vs CV vs student_test (gap = overfitting evidence)")
plt.tight_layout(); plt.savefig(M / "plot_overfit_gap.png", dpi=140); plt.close()

# 8) epoch-wise histories (only models that really produce them)
curves = json.load(open(M / "learning_curves.json"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
if "M6_LightGBM" in curves:
    for k, v in curves["M6_LightGBM"].items():
        ax[0].plot(v, label=k)
    ax[0].set_title("LightGBM binary logloss per boosting round"); ax[0].set_xlabel("round"); ax[0].legend()
if "M7_MLP" in curves:
    ax[1].plot(curves["M7_MLP"]["train_loss"], label="train loss")
    ax2 = ax[1].twinx(); ax2.plot(curves["M7_MLP"]["val_accuracy(internal 10% split)"], c="tab:orange", label="val accuracy (internal split)")
    ax[1].set_title("MLP: training loss and internal validation accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend(loc="center right")
plt.tight_layout(); plt.savefig(M / "plot_training_histories.png", dpi=140); plt.close()

# 9) feature importance of the leading tree model (LightGBM, gain)
import joblib
from sys import path
path.insert(0, str(Path(__file__).parent))
import pathsnap_data as P
lg = joblib.load(M / "M6_LightGBM.joblib")
imp = pd.Series(lg.booster_.feature_importance("gain"), index=P.FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 7))
(imp / imp.sum()).plot.barh(ax=ax, color="tab:green"); ax.set_title("LightGBM feature importance (share of total gain)")
plt.tight_layout(); plt.savefig(M / "plot_lightgbm_feature_importance.png", dpi=140); plt.close()
(imp / imp.sum()).sort_values(ascending=False).to_csv(M / "lightgbm_feature_importance.csv", header=["gain_share"])
print("plots written")
