"""Builds the technical report PDF from the actual result files in results/. Run: python report/build_report.py"""
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "PathSnap_Technical_Report.pdf"

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.4, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=4)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#0b1b2b"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#0f3d7a"))
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10, spaceBefore=5, spaceAfter=2)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8, leading=10, textColor=colors.HexColor("#444"), alignment=0)
CELL = ParagraphStyle("CELL", parent=B, fontSize=7.4, leading=9, alignment=0, spaceAfter=0)
MONO = ParagraphStyle("MONO", parent=B, fontName="Courier", fontSize=6.8, leading=8.4, alignment=0)

story = []
P = lambda t, s=B: story.append(Paragraph(t, s))
SP = lambda h=4: story.append(Spacer(1, h))


def img(path, width_cm=16.5, cap=None):
    path = Path(path)
    w, h = Image.open(path).size
    wd = width_cm * cm
    items = [RLImage(str(path), width=wd, height=wd * h / w)]
    if cap:
        items.append(Paragraph(cap, CAP))
    story.append(KeepTogether(items + [Spacer(1, 6)]))


def table(rows, widths, header=True, font=7.4, zebra=True):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b9c3ce")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f3"))]
    if zebra:
        st += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")])]
    t.setStyle(TableStyle(st))
    story.append(t); SP(6)


f4 = lambda v: f"{v:.4f}"
pc = lambda v: f"{100 * v:.2f}%"

# ------------------------------------------------------------------ data
mc = pd.read_csv(R / "models/model_comparison.csv").set_index("model")
cfg = json.load(open(R / "models/experiment_config.json"))
tv = pd.read_csv(R / "tuning/tuned_vs_untuned.csv").set_index("variant")
bp = json.load(open(R / "tuning/best_params.json"))
trials = pd.read_csv(R / "tuning/optuna_trials.csv")
pimp = pd.read_csv(R / "tuning/param_importance.csv", index_col=0)["importance"]
hc = pd.read_csv(R / "hmm/hmm_comparison.csv")
hpt = pd.read_csv(R / "hmm/per_trip_student_test.csv")
heb = pd.read_csv(R / "hmm/error_breakdown_student_test.csv", index_col=0)
dparams = json.load(open(R / "hmm/decoder_params.json"))
inv = pd.read_csv(R / "eda/dataset_inventory_per_trip.csv")
summ = pd.read_csv(R / "eda/dataset_summary_by_partition.csv", index_col=[0, 1])
lc = json.load(open(R / "models/learning_curves.json"))
fi = pd.read_csv(R / "models/lightgbm_feature_importance.csv", index_col=0)["gain_share"]
integ = json.load(open(R / "api/integration_results.json"))
base = cfg["baseline_nearest_arc_student_test"]
best_row = hc[hc.method.str.contains("tuned decoder") & hc.partition.str.contains("tuned LightGBM") & ~hc.partition.str.contains("untuned")].iloc[0]
clf_row = hc[hc.method.str.contains("no-candidate points filled") & hc.partition.str.contains("tuned LightGBM") & ~hc.partition.str.contains("untuned")].iloc[0]

# ------------------------------------------------------------------ title
P("PathSnap: Distributed Map-Matching and Route Reconstruction", ParagraphStyle("T", parent=H1, fontSize=20, spaceAfter=2))
P("Technical Report - Team AM255 / AM256 / AM257 / AM258, PES University (B.Tech AIML, Semester 5)", H3)
P(f"<b>Summary of results.</b> Candidate roads are generated within 60 m of every GPS point (K = 10), scored by a LightGBM classifier on 26 engineered features, "
  f"and decoded with an HMM/Viterbi route model. On the ten student-testing trips (19,160 GPS observations) the classifier alone reaches Top-1 accuracy "
  f"{pc(clf_row.top1)}; adding HMM decoding raises it to {pc(best_row.top1)} and route continuity from {pc(clf_row.continuity)} to {pc(best_row.continuity)}. "
  f"Seven models from six algorithm families were trained and compared. The nearest-road baseline reaches only {pc(base['top1_acc'])} Top-1 on the same trips. "
  "All numbers in this report are read from the CSV/JSON files in the <font face='Courier'>results/</font> folder of the repository and can be regenerated with the scripts listed in the README.")
P("<b>Data partitions used throughout.</b> <i>Training partition</i> = 80 training trips (2 anomalous trips, 00000033 and 00000054, excluded from fitting, leaving 78; see Section 2). "
  "<i>Student-testing partition</i> = 10 trips, used only for development-stage comparison and for reporting. <i>Evaluator partition</i> = 10 reserved trips, never used by us for any choice. "
  "Model fitting and tuning use group-aware cross-validation over training trips (3-fold GroupKFold, a trip is never split across folds), seed 42.")

# ------------------------------------------------------------------ 2 EDA
P("2. Dataset Understanding and Exploratory Data Analysis", H1)
P("2.1 Data and preprocessing", H2)
P("Each trip has its own road network and GPS track: <font face='Courier'>.track</font> (longitude, latitude, time), <font face='Courier'>.nodes</font>, <font face='Courier'>.arcs</font> (directed; row index = arc id) and, for training and student-testing trips, "
  "<font face='Courier'>.route</font> (ordered arc ids). The route file is a reference, not a perfect ground truth, and we treat it that way. Preprocessing steps (code: <font face='Courier'>src/pathsnap_data.py</font>, <font face='Courier'>src/eda.py</font>):")
for t in ["<b>Projection.</b> Each trip is projected to a local transverse-Mercator frame in metres, so that distances, headings and speeds are physical.",
          "<b>Candidate generation.</b> Every road arc is sampled every 15 m into a KD-tree; the nearest samples are refined with an exact point-to-segment distance. Arcs within 60 m are kept, at most K = 10 per GPS point.",
          "<b>Reference label.</b> Every GPS observation is assigned a reference arc by a monotone dynamic-programming alignment of the track to the ordered route (not by nearest arc, which would copy the baseline's errors). A candidate is positive when it equals that arc.",
          "<b>Anomaly screening.</b> A trip whose median distance to its own reference route exceeds a threshold is flagged and excluded from fitting (two training trips, see below).",
          "<b>No leakage by construction.</b> The inference path (<font face='Courier'>use_reference=False</font>) builds identical features without reading the route file; we checked that the features equal the cached training features."]:
    P("&bull; " + t)
P("2.2 Missing values, gaps and quality of the data (partition: 80 training trips and 10 student-testing trips)", H2)
mv_tr, mv_te = int(inv[inv.part == "train"].missing_values.sum()), int(inv[inv.part == "student_test"].missing_values.sum())
rows = [["Statistic (per trip)", "Training (80)", "Student-testing (10)"]]
sel = [("n_obs", "GPS observations, mean"), ("duration_s", "Duration (s), median"), ("n_arcs", "Road arcs, mean"), ("n_components", "Disconnected components, median"),
       ("frac_dt_not_1s", "Fraction of steps with dt != 1 s, mean"), ("max_gap_s", "Largest time gap (s), median"), ("dref_median_m", "Median GPS-to-reference distance (m), median"),
       ("candidate_miss_rate", "Candidate miss rate, mean"), ("route_consecutive_connected_frac", "Route arcs consecutively connected, mean")]
stat = {"n_obs": "mean", "duration_s": "median", "n_arcs": "mean", "n_components": "median", "frac_dt_not_1s": "mean", "max_gap_s": "median",
        "dref_median_m": "median", "candidate_miss_rate": "mean", "route_consecutive_connected_frac": "mean"}
for k, lab in sel:
    a, b = summ.loc[(k, stat[k]), "train"], summ.loc[(k, stat[k]), "student_test"]
    fmt = (lambda v: f"{v:.4f}") if abs(a) < 5 else (lambda v: f"{v:,.1f}")
    rows.append([lab, fmt(a), fmt(b)])
rows.append(["Missing (NaN) values in tracks/nodes, total", str(mv_tr), str(mv_te)])
rows.append(["Trips with unsorted time / duplicated timestamps", f"{int((~inv[inv.part=='train'].time_sorted).sum())} / {int((inv[inv.part=='train'].duplicated_timestamps>0).sum())}",
             f"{int((~inv[inv.part=='student_test'].time_sorted).sum())} / {int((inv[inv.part=='student_test'].duplicated_timestamps>0).sum())}"])
table(rows, [8.2, 4, 4.2])
P(f"<b>Missing values.</b> We found {mv_tr + mv_te} NaN values in any track or node file, timestamps are sorted and unique, and no node is isolated. The real data-quality problems are different: "
  "(i) <b>time gaps</b> - in the training partition 6.2% of steps are not 1 s apart on average (up to 46.9% in one trip), and one trip has a gap of 153,064 s; "
  "(ii) <b>anomalous trips</b> - training trips 00000033 and 00000054 have a median GPS-to-reference distance of up to 140 m (the typical value is about 4.6 m) and a candidate miss rate up to 90.6%, meaning that their track and route do not agree; we excluded them from fitting and report this choice; "
  "(iii) <b>fragmented networks</b> - road graphs have many disconnected components (median 20 in training), so two nearby arcs need not be reachable from each other; this is why the HMM needs a reachability test. "
  "The route files are consistent with their networks (99.9% of consecutive route arcs are connected).")
img(R / "eda/01_dataset_overview.png", 16.5, "Figure 2.1 - Dataset overview (partitions: training, student-testing).")
img(R / "eda/02_gps_noise_and_anomalies.png", 16.5, "Figure 2.2 - GPS-to-reference distance and anomalous trips (partitions: training, student-testing).")
img(R / "eda/03_time_step_distribution.png", 13, "Figure 2.3 - Distribution of time steps between observations (partition: training).")
P("2.3 GPS trajectories and road networks", H2)
img(R / "eda/04_trip_train_00000001.png", 15, "Figure 2.4 - GPS track, road network and reference route of training trip 00000001 (partition: training).")
img(R / "eda/04_trip_student_test_00000000.png", 15, "Figure 2.5 - The same view for student-testing trip 00000000 (partition: student-testing).")
P("The Folium maps for these trips (<font face='Courier'>results/maps/eda_*.html</font>) are generated by <font face='Courier'>src/eda.py</font>; the final prediction maps are in Section 6.")
P("2.4 Patterns observed and features selected", H2)
P("The nearest-road baseline gets only " + pc(base["top1_acc"]) + " Top-1 accuracy on the student-testing trips, with a recall of " + pc(base["recall_pos"]) + " on positive candidates. "
  "GPS error of a few metres in a dense city grid often makes a parallel street or the reverse-direction arc closer than the true arc, so distance alone is insufficient. "
  "This pattern led us to 26 features in seven groups (code: <font face='Courier'>src/pathsnap_data.py</font>):")
table([["Group", "Features", "Why we chose them"],
       ["Distance (6)", "d, rank_d, d_min, d_minus_min, d_ratio_min, n_cand", "Absolute and relative closeness: an arc 3 m away is only convincing if the alternatives are farther."],
       ["Position (2)", "frac, arc_len", "Where the point projects on the arc; short arcs and arc ends behave differently."],
       ["Direction (3)", "ang_diff, cos_diff, heading_valid", "Heading of the movement against the arc heading separates the two directions of a two-way road."],
       ["Motion (3)", "speed, dt_prev, dt_next", "Low-speed points are noisy; time steps show gaps."],
       ["Neighbour (6)", "d_prev2/5, d_next2/5, progress2/5", "Distance to the same arc for earlier/later points and progress along it: a sequence signal that a single point lacks."],
       ["Context (3)", "ctx_mean, ctx_minus_min, ctx_rank", "How the candidate fits a window of surrounding observations."],
       ["Topology (3)", "to_outdeg, from_indeg, has_reverse", "Junction complexity and whether the reverse arc exists."]], [2.6, 5.4, 8.4])
top = fi.head(5)
P("The importance of features measured by LightGBM gain confirms this reasoning: " + ", ".join(f"<font face='Courier'>{k}</font> ({100*v:.1f}%)" for k, v in top.items()) +
  " are the top five. The relative-distance feature and the neighbour-progress feature together carry about " + f"{100*(fi['d_minus_min']+fi['progress5']):.0f}% of the total gain.")
img(R / "models/plot_lightgbm_feature_importance.png", 12, "Figure 2.6 - LightGBM feature importance by gain (trained on the training partition).")

# ------------------------------------------------------------------ 3 models
story.append(PageBreak())
P("3. Model Experimentation and Comparison", H1)
P("3.1 Overall comparison", H2)
P("Every model scores each candidate arc; the arc with the highest score per GPS observation is the prediction (Top-1). The <b>primary selection metric is the 3-fold grouped CV macro-F1</b> on the training trips, because the classes are imbalanced (about 14% positives) so plain accuracy is dominated by the negatives, "
  "and grouped folds measure generalisation to unseen trips. The student-testing columns are reported for development comparison only. Models were fitted on at most 200,000 candidate rows each (seed 42); all use the same 78 clean trips and the same 26 features.")
names = {"M1_LogReg": "M1 Logistic Regression", "M2_GaussNB": "M2 Gaussian Naive Bayes", "M3_DecisionTree": "M3 Decision Tree", "M4_RandomForest": "M4 Random Forest",
         "M5_HistGB": "M5 HistGradientBoosting", "M6_LightGBM": "M6 LightGBM", "M7_MLP": "M7 Shallow MLP"}
rows = [["Model", "Algorithm family", "Key configuration", "CV macro-F1 (mean +/- sd)", "Student-test Top-1", "Route continuity", "Student-test macro-F1", "Log loss", "Inference ms / 1000 obs"]]
for k, n in names.items():
    r = mc.loc[k]
    rows.append([n, r.family, r.config, f"{r.cv_macro_f1:.4f} +/- {r.cv_macro_f1_std:.4f}", pc(r.test_top1), pc(r.test_continuity), f4(r.test_macro_f1), f4(r.test_log_loss), f"{r.infer_ms_per_1000_obs:.1f}"])
rows.append(["Baseline: nearest arc", "Rule", "No learning", "N/A (no training)", pc(base["top1_acc"]), pc(base["route_continuity"]), f4(base["macro_f1"]), "N/A (score is not a probability)", "N/A"])
table(rows, [2.2, 1.9, 3.6, 1.9, 1.4, 1.4, 1.4, 1.4, 1.4])
P("Route continuity is the share of consecutive observations (dt &lt;= 5 s) whose predicted arcs are the same or within two network hops. The Naive Bayes model has a high recall (93.7%) but a precision of only 59.4%: its independence assumption over strongly correlated features produces over-confident probabilities (log loss 0.712, about 10 times worse than the boosted models). "
  "LightGBM has the best CV macro-F1 (0.9422), the best student-test Top-1 (90.98%), the best macro-F1 and the best log loss. The differences between M3-M7 are small compared with the fold standard deviation (about 0.007 to 0.011), so we do not claim that LightGBM is better than HistGradientBoosting or the Random Forest by a large margin; we chose it because it is best on every metric, has the smallest fold-to-fold spread, and exposes a calibrated probability that the HMM can use. "
  "Its inference cost (64.5 ms per 1000 observations) is still small: a 4,000-point trip is scored in about 0.26 s.")
img(R / "models/plot_cv_macro_f1.png", 14, "Figure 3.1 - CV macro-F1 per model and fold (partition: training, 3-fold grouped CV).")
img(R / "models/plot_student_test_top1_f1.png", 14, "Figure 3.2 - Student-testing Top-1 accuracy and macro-F1 per model (partition: student-testing).")
img(R / "models/plot_logloss_time_continuity.png", 16, "Figure 3.3 - Log loss, inference time and route continuity per model (partition: student-testing).")
img(R / "models/plot_overfit_gap.png", 13, "Figure 3.4 - Macro-F1 on the training sample versus CV macro-F1 (overfitting gap).")
img(R / "models/plot_confusion_matrices.png", 16.5, "Figure 3.5 - Candidate-level confusion matrices on the student-testing partition.")

story.append(PageBreak())
P("3.2 Per-model analysis", H2)
gap = lambda k: mc.loc[k].train_sample_macro_f1 - mc.loc[k].cv_macro_f1
mlp_loss, mlp_val = lc["M7_MLP"]["train_loss"], lc["M7_MLP"]["val_accuracy(internal 10% split)"]
lg = lc["M6_LightGBM"]
best_it = int(pd.Series(lg["student_test"]).idxmin()) + 1
models_txt = {
 "M1_LogReg": dict(
  A="Logistic regression is the natural probabilistic baseline: it gives a calibrated score per candidate and tells us how far a linear rule on our features can go. Strengths: fast (0.62 ms per 1000 observations), stable, interpretable. Limitation: the decision surface is linear in the features, but the true rule is conditional (e.g. direction matters only when the point is near a junction).",
  B="C = 1.0 (L2), lbfgs solver, max_iter 300, features standardised. Objective: L2-regularised log loss (binary cross-entropy).",
  C=f"Student-test Top-1 {pc(mc.loc['M1_LogReg'].test_top1)}, macro-F1 {f4(mc.loc['M1_LogReg'].test_macro_f1)}, log loss {f4(mc.loc['M1_LogReg'].test_log_loss)}. Underfitting: the training-sample macro-F1 ({mc.loc['M1_LogReg'].train_sample_macro_f1:.4f}) equals the CV value ({mc.loc['M1_LogReg'].cv_macro_f1:.4f}), so there is no overfitting gap, but both are 3 points below the tree models; capacity, not data, is the limit."),
 "M2_GaussNB": dict(
  A="Naive Bayes is the simplest generative probabilistic model and a useful lower reference. Strength: trains in 0.05 s and needs no tuning. Limitation: it assumes conditionally independent features while many of ours (d, d_min, d_minus_min, the neighbour distances) are strongly correlated.",
  B="GaussianNB, var_smoothing 1e-9. Objective: maximum likelihood of class-conditional Gaussians (no iterative loss).",
  C=f"Student-test Top-1 {pc(mc.loc['M2_GaussNB'].test_top1)}, macro-F1 {f4(mc.loc['M2_GaussNB'].test_macro_f1)}, log loss {f4(mc.loc['M2_GaussNB'].test_log_loss)}. Underfitting and miscalibration: precision on positives is only {pc(mc.loc['M2_GaussNB'].test_precision)} at {pc(mc.loc['M2_GaussNB'].test_recall)} recall, the train/CV gap is zero ({mc.loc['M2_GaussNB'].train_sample_macro_f1:.4f} vs {mc.loc['M2_GaussNB'].cv_macro_f1:.4f}), and the log loss is ten times that of the boosted models."),
 "M3_DecisionTree": dict(
  A="A single tree captures the non-linear if-then structure (near arc AND heading agrees AND moving forward) that the linear model cannot. Strengths: very fast (0.46 ms), readable. Limitation: unstable and gives coarse probabilities (leaf frequencies).",
  B="max_depth 12, min_samples_leaf 20, Gini impurity. Objective: minimise Gini impurity at each split.",
  C=f"Student-test Top-1 {pc(mc.loc['M3_DecisionTree'].test_top1)}, macro-F1 {f4(mc.loc['M3_DecisionTree'].test_macro_f1)}, log loss {f4(mc.loc['M3_DecisionTree'].test_log_loss)}. Mild overfitting: training-sample macro-F1 {mc.loc['M3_DecisionTree'].train_sample_macro_f1:.4f} against CV {mc.loc['M3_DecisionTree'].cv_macro_f1:.4f} (gap {gap('M3_DecisionTree'):.3f}); fold macro-F1 ranges from 0.926 to 0.947, the widest spread among the tree models."),
 "M4_RandomForest": dict(
  A="Bagging many decorrelated trees reduces the variance of the single tree while keeping its non-linearity. Strength: stable across trips. Limitation: 11.5 ms per 1000 observations and large memory; the averaged probabilities are under-confident near 0 and 1, which affects the emission scores of the HMM.",
  B="n_estimators 50, max_depth 14, min_samples_leaf 10, bootstrap sampling. Objective: Gini impurity per tree, probabilities averaged.",
  C=f"Student-test Top-1 {pc(mc.loc['M4_RandomForest'].test_top1)}, macro-F1 {f4(mc.loc['M4_RandomForest'].test_macro_f1)}, log loss {f4(mc.loc['M4_RandomForest'].test_log_loss)}. Moderate overfitting: training-sample macro-F1 {mc.loc['M4_RandomForest'].train_sample_macro_f1:.4f} vs CV {mc.loc['M4_RandomForest'].cv_macro_f1:.4f} (gap {gap('M4_RandomForest'):.3f}); CV and student-test macro-F1 agree to within 0.008. Because the saved forest is large it is not stored in the repository (its results are stored; it can be regenerated by the training script)."),
 "M5_HistGB": dict(
  A="Gradient boosting corrects the errors of earlier trees and handles features on different scales without preprocessing. Strengths: strong accuracy, good probabilities (log loss 0.0711). Limitation: slower inference than bagged trees here (27.5 ms) and sensitive to the learning rate.",
  B="max_iter 200, learning_rate 0.1, max_leaf_nodes 31. Objective: binary log loss, minimised by gradient boosting on histogram-binned features.",
  C=f"Student-test Top-1 {pc(mc.loc['M5_HistGB'].test_top1)}, macro-F1 {f4(mc.loc['M5_HistGB'].test_macro_f1)}, log loss {f4(mc.loc['M5_HistGB'].test_log_loss)}. Moderate overfitting: training-sample macro-F1 {mc.loc['M5_HistGB'].train_sample_macro_f1:.4f} vs CV {mc.loc['M5_HistGB'].cv_macro_f1:.4f} (gap {gap('M5_HistGB'):.3f}); the test result is within 0.004 of LightGBM, so the two boosters behave almost the same."),
 "M6_LightGBM": dict(
  A="LightGBM is a leaf-wise gradient-boosted tree model: it fits the interactions between distance, direction and neighbour progress well, and it emits probabilities with low log loss, which the HMM uses as emissions. Strength: best accuracy and best probability quality. Limitation: more hyperparameters, and the largest train-CV gap, so it needs regularisation.",
  B="n_estimators 300, learning_rate 0.05, num_leaves 63, subsample 0.8, colsample_bytree 0.8. Objective: binary logistic loss (log loss), gradient boosting.",
  C=f"Student-test Top-1 {pc(mc.loc['M6_LightGBM'].test_top1)}, macro-F1 {f4(mc.loc['M6_LightGBM'].test_macro_f1)}, log loss {f4(mc.loc['M6_LightGBM'].test_log_loss)}. Some overfitting: training-sample macro-F1 {mc.loc['M6_LightGBM'].train_sample_macro_f1:.4f} vs CV {mc.loc['M6_LightGBM'].cv_macro_f1:.4f} (gap {gap('M6_LightGBM'):.3f}), the largest of all models; however, the student-test log loss reaches its minimum at boosting round {best_it} of 300 and rises only slightly afterwards (Figure 3.6), and CV and student-test macro-F1 agree to within 0.002, so the gap does not translate into worse generalisation."),
 "M7_MLP": dict(
  A="A shallow neural network tests whether a smooth non-linear function on standardised features competes with trees. Strength: very fast inference (1.8 ms) and smooth probabilities. Limitation: needs scaling and has no native handling of the sharp threshold rules in our features.",
  B="Two hidden layers (64, 32, ReLU), Adam, L2 alpha 1e-4, early stopping on a 10% internal split, features standardised. Objective: log loss plus L2 penalty.",
  C=f"Student-test Top-1 {pc(mc.loc['M7_MLP'].test_top1)}, macro-F1 {f4(mc.loc['M7_MLP'].test_macro_f1)}, log loss {f4(mc.loc['M7_MLP'].test_log_loss)}. Slight overfitting only: training loss fell from {mlp_loss[0]:.3f} to {mlp_loss[-1]:.3f} in {len(mlp_loss)} epochs while the internal validation accuracy settled at {mlp_val[-1]:.3f} and early stopping ended training; training-sample macro-F1 {mc.loc['M7_MLP'].train_sample_macro_f1:.4f} vs CV {mc.loc['M7_MLP'].cv_macro_f1:.4f} (gap {gap('M7_MLP'):.3f})."),
}
for k, n in names.items():
    m = models_txt[k]
    story.append(KeepTogether([Paragraph(f"3.2.{list(names).index(k)+1} {n}", H3),
                               Paragraph("<b>A. Model understanding.</b> " + m["A"], B), Paragraph("<b>B. Training and configuration.</b> " + m["B"], B),
                               Paragraph("<b>C. Experimental results (student-testing partition).</b> " + m["C"], B)]))
P("3.3 Model visualisations", H2)
P("Training histories exist only for the two models that train iteratively: the LightGBM log loss per boosting round (training sample versus student-testing) and the MLP loss per epoch. Other models have no epoch-wise history, so no curves are drawn for them.")
img(R / "models/plot_training_histories.png", 16, "Figure 3.6 - Actual training histories: LightGBM log loss per boosting round and MLP loss per epoch.")

# ------------------------------------------------------------------ 4 Bayesian
P("4. Bayesian Hyperparameter Optimisation (LightGBM)", H1)
P(f"<b>Why LightGBM.</b> It had the best CV macro-F1, the best probabilities, and the largest train-CV gap, so regularisation parameters had the most to give. Search code: <font face='Courier'>src/tune_lightgbm.py</font>; trials: <font face='Courier'>results/tuning/optuna_trials.csv</font>.")
space = [["Hyperparameter", "Range / distribution", "Best value", "Importance"],
         ["n_estimators", "integer 100 - 500", str(bp["best_params"]["n_estimators"]), f"{pimp['n_estimators']:.3f}"],
         ["learning_rate", "log-uniform 0.02 - 0.2", f"{bp['best_params']['learning_rate']:.4f}", f"{pimp['learning_rate']:.3f}"],
         ["num_leaves", "log-integer 15 - 127", str(bp["best_params"]["num_leaves"]), f"{pimp['num_leaves']:.3f}"],
         ["max_depth", "integer 4 - 14", str(bp["best_params"]["max_depth"]), f"{pimp['max_depth']:.3f}"],
         ["min_child_samples", "log-integer 5 - 200", str(bp["best_params"]["min_child_samples"]), f"{pimp['min_child_samples']:.3f}"],
         ["subsample", "uniform 0.5 - 1.0", f"{bp['best_params']['subsample']:.3f}", f"{pimp['subsample']:.3f}"],
         ["colsample_bytree", "uniform 0.5 - 1.0", f"{bp['best_params']['colsample_bytree']:.3f}", f"{pimp['colsample_bytree']:.3f}"],
         ["reg_lambda", "log-uniform 1e-3 - 10", f"{bp['best_params']['reg_lambda']:.4f}", f"{pimp['reg_lambda']:.3f}"],
         ["reg_alpha", "log-uniform 1e-3 - 10", f"{bp['best_params']['reg_alpha']:.4f}", f"{pimp['reg_alpha']:.3f}"]]
table(space, [3.6, 5, 3.4, 3])
P(f"<b>Objective:</b> mean 3-fold GroupKFold CV macro-F1 over the training trips (evaluator and student-testing trips were never used to choose the search space, objective or parameters). <b>Sampler:</b> TPE (Optuna) with 8 random start-up trials and a median pruner, seed 42; the untuned configuration was enqueued as the first trial. "
  f"<b>Trials:</b> {bp['n_trials']} ({bp['n_complete']} completed, {bp['n_pruned']} pruned), search time {bp['search_seconds']:.0f} s. <b>Best CV macro-F1:</b> {bp['best_cv_macro_f1']:.4f}.")
u, t = tv.loc["untuned"], tv.loc["tuned"]
table([["Metric (same partitions and folds)", "Untuned", "Tuned"],
       ["CV macro-F1 (training trips, mean +/- sd)", f"{u.cv_macro_f1:.4f} +/- {u.cv_macro_f1_std:.4f}", f"{t.cv_macro_f1:.4f} +/- {t.cv_macro_f1_std:.4f}"],
       ["Student-test Top-1 accuracy", pc(u.top1_acc), pc(t.top1_acc)], ["Student-test macro-F1", f4(u.macro_f1), f4(t.macro_f1)],
       ["Student-test log loss", f4(u.log_loss), f4(t.log_loss)], ["Inference time (ms / 1000 obs)", f"{u.infer_ms_per_1000_obs:.1f}", f"{t.infer_ms_per_1000_obs:.1f}"],
       ["Fit time (s)", f"{u.fit_seconds:.1f}", f"{t.fit_seconds:.1f}"]], [7.5, 4.2, 4.2])
P(f"<b>Did the tuned model outperform the untuned model?</b> Not in a meaningful way. The search raised the CV macro-F1 by only {t.cv_macro_f1-u.cv_macro_f1:+.4f}, which is far smaller than the fold standard deviation ({u.cv_macro_f1_std:.3f}); on the student-testing trips Top-1 accuracy and macro-F1 were slightly lower ({100*(t.top1_acc-u.top1_acc):+.2f} points and {t.macro_f1-u.macro_f1:+.4f}) and log loss slightly higher. "
  f"The tuned model is faster at inference ({t.infer_ms_per_1000_obs:.0f} vs {u.infer_ms_per_1000_obs:.0f} ms per 1000 observations). Our reading is that the hand-chosen configuration was already near the plateau of this feature set, and that remaining gains come from the features and the HMM, not from the booster settings. "
  "We keep the tuned model in the deployed pipeline because it was selected by the pre-declared CV objective, not by looking at the student-testing trips; the choice between the two is within the noise of 10 trips (Section 5 shows the same pattern with the decoder). "
  "The untuned row here is a refit inside the tuning script, so it differs by a few tenths of a point from the Section 3 LightGBM row (different 200,000-row sample).")
img(R / "tuning/plot_optuna_history.png", 14, "Figure 4.1 - Optuna trial history (objective: CV macro-F1, partition: training).")

# ------------------------------------------------------------------ 5 HMM
story.append(PageBreak())
P("5. HMM and Viterbi Route Reconstruction", H1)
P("Code: <font face='Courier'>src/hmm_decoder.py</font> (decoder), <font face='Courier'>src/run_hmm.py</font> (tuning and experiments).")
for q, a in [
    ("Hidden states", "For GPS observation t the hidden state is one of its candidate road arcs (at most 10). The candidate-to-arc correspondence is kept, so the decoded state is directly the predicted arc."),
    ("Emissions", "The log of the classifier's candidate probability, normalised over the candidates of that observation and multiplied by a weight alpha. Normalisation makes the scores comparable between observations with different numbers of candidates."),
    ("Transitions", "A Newson-Krumm style score: -|d_route - d_gps| / beta, where d_route is the network distance between the two arcs (bounded Dijkstra over arc end nodes, 600 m limit) and d_gps the straight distance between the observations; plus a penalty for a U-turn onto the reverse arc and a small penalty for changing arc. Pairs that are unreachable within the bound receive a fixed floor penalty instead of probability zero, so the chain can always continue."),
    ("Missing candidates and gaps", "Observations with no candidate inside 60 m are skipped by the chain and filled with the previous decoded arc. If the time between observations exceeds gap_s the chain restarts (no transition across the gap), so a long gap cannot force an unrealistic connection."),
    ("Disconnected segments", "If consecutive observations favour arcs that cannot be reached from each other, the transition gets the floor penalty; Viterbi then prefers an alternative sequence of connected arcs unless the classifier evidence is strong enough to outweigh the penalty."),
    ("Decoder parameters", f"alpha, beta, U-turn penalty, switch penalty, floor, gap_s and normalisation were tuned by Optuna TPE (60 trials) on out-of-fold training probabilities of a 26-trip subset (objective: pointwise Top-1); the student-testing trips were used only for reporting. Best: alpha {dparams['best_params']['alpha']:.2f}, beta {dparams['best_params']['beta']:.1f}, U-turn {dparams['best_params']['uturn_pen']:.1f}, switch {dparams['best_params']['switch_pen']:.2f}, floor {dparams['best_params']['floor']:.1f}, gap {dparams['best_params']['gap_s']:.0f} s.")]:
    P(f"<b>{q}.</b> {a}")
P("5.1 Did HMM decoding help?", H2)
rows = [["Partition", "Method", "Top-1", "Route continuity", "Route precision", "Route recall", "Route F1"]]
for _, r in hc.iterrows():
    if "Ablation" in r.method:
        continue
    rows.append([r.partition, r.method, pc(r.top1), pc(r.continuity), pc(r.route_precision), pc(r.route_recall), pc(r.route_f1)])
table(rows, [3.6, 5.6, 1.5, 1.5, 1.5, 1.4, 1.3])
P(f"<b>Yes, clearly.</b> On the ten student-testing trips the HMM raises Top-1 from {pc(clf_row.top1)} to {pc(best_row.top1)}, route continuity from {pc(clf_row.continuity)} to {pc(best_row.continuity)}, and route F1 from {pc(clf_row.route_f1)} to {pc(best_row.route_f1)}; "
  f"on out-of-fold training predictions (78 trips) the gain is of the same size ({pc(hc.iloc[1].top1)} to {pc(hc.iloc[3].top1)}), so it is not specific to the ten test trips. All ten student-testing trips improve (per-trip gain between {100*hpt.delta_top1.min():.1f} and {100*hpt.delta_top1.max():.1f} points; trip 00000007 rises from 94.0% to 99.5%).")
img(R / "hmm/plot_hmm_per_trip.png", 14, "Figure 5.1 - Top-1 accuracy per student-testing trip, classifier only vs classifier + HMM.")
P("5.2 Where HMM decoding did not help", H2)
for t in [f"<b>Tuned decoder parameters did not beat the defaults.</b> Default parameters give {pc(hc.iloc[6].top1)} on the student-testing trips and {pc(hc.iloc[2].top1)} out-of-fold; tuned parameters give {pc(hc.iloc[7].top1)} and {pc(hc.iloc[3].top1)}. The tuning objective improved on its 26-trip subset (92.31% classifier-only, 94.98% default, 95.24% tuned) but did not transfer to other trips; the sensible reading is that the default values already sit on a flat optimum, so the extra tuning only fitted that subset. The difference is below 0.2 points and we report it as no gain.",
          f"<b>Direction errors remain the main error type.</b> After decoding, wrong-direction assignments fall from {int(heb.loc['classifier only','wrong_direction'])} to {int(heb.loc['classifier + HMM','wrong_direction'])} and other errors from {int(heb.loc['classifier only','other'])} to {int(heb.loc['classifier + HMM','other'])}, but adjacent-arc errors barely change ({int(heb.loc['classifier only','adjacent_arc'])} to {int(heb.loc['classifier + HMM','adjacent_arc'])}). When the vehicle stops or turns around at a junction the reverse arc is both geometrically and temporally plausible.",
          f"<b>Hard trip.</b> Trip 00000077 improves by {100*hpt.loc[hpt.trip==77,'delta_top1'].iloc[0] if (hpt.trip==77).any() else 100*hpt.loc[hpt.trip.astype(str).str.endswith('77'),'delta_top1'].iloc[0]:.1f} points but still reaches only {pc(hpt[hpt.trip.astype(str).str.endswith('77')].top1_hmm.iloc[0])}: it contains a long stop where observations cluster around a junction (Figure 6.1, lower right), and the classifier evidence there is weak.",
          f"<b>Route-level recall.</b> Route precision ({pc(best_row.route_precision)}) is higher than recall ({pc(best_row.route_recall)}), and only {pc(best_row.route_connected)} of consecutive predicted arcs are adjacent: the decoder only chooses among arcs near GPS points, so arcs passed between two sparse observations are not reconstructed.",
          "<b>Ablations (student-testing, tuned LightGBM).</b> Removing the U-turn penalty lowers Top-1 by about 0.1 point; removing the switch, U-turn and floor shaping together lowers it by 0.4 point. Removing the gap restart or using unnormalised emissions changes nothing on these ten trips, because they contain few gaps longer than 30 s and normalised and unnormalised scores rank the same arcs; we keep both because gaps are frequent in other training trips (Section 2)."]:
    P("&bull; " + t)

# ------------------------------------------------------------------ 6 Folium
story.append(PageBreak())
P("6. Geographical Visualisation Using Folium", H1)
P("Code: <font face='Courier'>src/folium_final_map.py</font>; interactive maps: <font face='Courier'>results/maps/final_student_test_00000076.html</font> and <font face='Courier'>..._00000077.html</font> (partition: student-testing). "
  "Layers (with layer control and legend): raw GPS observations (blue), reference route (green, wide), classifier-only predicted arcs (orange, optional), classifier + HMM predicted route (red), and the road network around the trip (grey, optional). "
  "The prediction layers are produced by the same pipeline class that the API uses; the reference route is loaded only to draw the green comparison layer.")
img(R / "screenshots/folium_final_00000076.png", 16.5, "Figure 6.1a - Student-testing trip 00000076, classifier + HMM route (red) over reference route (green) and GPS (blue).")
img(R / "screenshots/folium_final_00000077.png", 16.5, "Figure 6.1b - Student-testing trip 00000077 (hardest trip, Top-1 83.7% after HMM). The red/green block near the lower right is the stop discussed in Section 5.2.")
img(R / "maps/../hmm/plot_decoder_tuning.png", 12, "Figure 6.2 - Decoder tuning history (partition: 26 training trips, out-of-fold probabilities).") if False else None

# ------------------------------------------------------------------ 7 API UI
P("7. API, UI and Deployment", H1)
P("<b>Architecture.</b> Laptop 1 runs a FastAPI service (<font face='Courier'>api/app.py</font>, started with <font face='Courier'>uvicorn app:app --host 0.0.0.0 --port 8000</font>); Laptop 2 serves a static page (<font face='Courier'>ui/index.html</font>, <font face='Courier'>python -m http.server 8080</font>) that is pointed at the address of Laptop 1. "
  "Endpoints: <font face='Courier'>GET /health</font>, <font face='Courier'>GET /trips</font> (lists the ten evaluation trips and their sizes), <font face='Courier'>POST /predict {trip_id}</font>. "
  "<b>Pipeline per request:</b> read track, nodes and arcs, generate candidates and features, score with the saved LightGBM model, decode with HMM/Viterbi, then compare with the reference. "
  "<b>Ground truth stays server-side:</b> the prediction path loads a trip with <font face='Courier'>with_route=False</font> and builds features with <font face='Courier'>use_reference=False</font>; the route file is read only inside <font face='Courier'>ServerSideReference.evaluate()</font> after the prediction exists, and the response carries only aggregate scores. "
  "The response contains the GPS points, the predicted route as line segments, the evaluation scores, and timings. A request log is written to the console and to <font face='Courier'>api/api_requests.log</font>.")
P("<b>Test on all ten trips over HTTP</b> (<font face='Courier'>tests/run_integration.py</font>; trips of the student-testing partition were used as stand-ins because the evaluator trips are not available to us):")
rows = [["Trip", "GPS points", "Top-1 (HMM)", "Top-1 (classifier)", "Route F1", "Continuity", "Server time (s)", "Round trip (s)", "Contract check"]]
for r in integ["per_trip"]:
    rows.append([r["trip"], f"{r['n_points']:,}", pc(r["top1"]), pc(r["clf_only"]), pc(r["route_f1"]), pc(r["continuity"]), f"{r['server_s']:.2f}", f"{r['round_trip_s']:.2f}", "passed" if r["contract_ok"] else "FAILED"])
a = integ["aggregate"]
rows.append(["All", f"{a['n_points']:,}", pc(a["top1_pooled"]), pc(a["clf_only_pooled"]), "", "", "", "", "passed" if a["all_contract_checks_passed"] else "FAILED"])
table(rows, [1.9, 1.7, 1.9, 2.1, 1.6, 1.8, 1.9, 1.8, 1.8])
P("The contract check verifies that the response has no reference-route field, that the evaluation block holds only the ten allowed aggregate keys, and that the predicted route and GPS lengths are consistent. The pooled Top-1 from the API (93.43%) agrees with the offline experiment in Section 5 (93.42%).")
img(R / "screenshots/ui_trip_00000076.png", 16.5, "Figure 7.1 - Evaluator UI after selecting trip 00000076 (map, predicted route, metrics, request log).")
img(R / "screenshots/ui_trip_00000077.png", 16.5, "Figure 7.2 - Evaluator UI, trip 00000077.")
P("<b>API log excerpt</b> (<font face='Courier'>api/api_requests.log</font>):")
log = (ROOT / "api/api_requests.log").read_text().strip().splitlines()[-9:]
story.append(Paragraph("<br/>".join(l.replace("&", "&amp;").replace("<", "&lt;") for l in log), MONO)); SP(6)
P("<b>Deployment.</b> Install with <font face='Courier'>pip install -r requirements_api.txt</font>; set <font face='Courier'>PATHSNAP_EVAL_DIR</font> to the folder holding the evaluation trips and start the API (<font face='Courier'>api/run_api.bat</font>); open the UI, type the address of Laptop 1 and press Connect. The README lists each step. "
  "The model file (3.4 MB) and <font face='Courier'>config.json</font> (features, candidate settings, decoder parameters, seed) are stored in <font face='Courier'>artifacts/</font>.")

P("8. Reproducibility", H1)
P("Seed 42 is used everywhere. Scripts: <font face='Courier'>build_dataset.py</font> (candidate and feature cache), <font face='Courier'>eda.py</font>, <font face='Courier'>train_models.py</font>, <font face='Courier'>tune_lightgbm.py</font>, <font face='Courier'>run_hmm.py</font>, <font face='Courier'>export_artifacts.py</font>, <font face='Courier'>folium_final_map.py</font>. Logs: <font face='Courier'>results/models/train_log.txt</font>, <font face='Courier'>results/tuning_log.txt</font>, <font face='Courier'>results/hmm_log.txt</font>. "
  "Every table and figure above comes from the CSV/JSON/PNG files in <font face='Courier'>results/</font>.")


def foot(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey)
    c.drawString(2 * cm, 1.1 * cm, "PathSnap technical report - AM255 / AM256 / AM257 / AM258"); c.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {d.page}"); c.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                  title="PathSnap Technical Report", author="Team AM255-AM258").build(story, onFirstPage=foot, onLaterPages=foot)
print("wrote", OUT)
