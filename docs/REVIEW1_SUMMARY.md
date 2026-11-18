# Review 1 summary (all numbers come from results/ files)

## Problem in one line
Each trip has its own road network (nodes, directed arcs). Given noisy 1 Hz GPS points, pick the arc the vehicle is on.
`.route` is only an ordered arc list, so the per-observation reference arc is derived by a monotone minimum-distance
alignment of the track onto the route (limitation: label noise at arc boundaries, reported openly).

## Data facts
- 80 train trips, 10 student_test trips; 208,417 / 19,160 GPS points. No missing values, no isolated nodes.
- Networks are not fully connected (many small components; largest component 69-100% of nodes). Reference routes are
  consecutive-connected for >= 98.1% of arc pairs.
- 96.5% of train steps are exactly 1 s; some trips have huge gaps (max 153,064 s in train, 210 s in student_test).
- Distance GPS -> reference arc (clean trips): median 5.0 m, p95 18.8 m. Student_test: median 5.3 m, p95 16.9 m.
- Data-quality decision: train trips 00000033 and 00000054 have the GPS ~120-140 m (median) from their reference route
  (54: only 16% of points have any arc within 60 m). They are excluded from fitting/CV, kept in EDA.
- Geometry: per-trip local transverse-Mercator projection (metres), not degrees.

## Candidate generation
KD-tree over points sampled every 15 m along each arc, exact point-to-segment distance, radius 60 m, at most 10 arcs.
Candidate miss rate: 0.26% (student_test), 1.5% (clean train); mean 7.35 candidates per observation (test).

## Features (26)
Distance (d, rank, gap to nearest, ratio), position on arc, arc length, heading difference/cosine to track direction,
speed, time gaps, neighbouring-observation distances (+-2, +-5 samples) and along-arc progress, context-mean distance
and rank, topology (out/in degree, reverse arc exists). Top LightGBM gains: d_minus_min 31%, progress5 27%, rank_d 12%.

## Experiments (3-fold GroupKFold by trip; fit capped at 200k rows for every model; seed 42)
| Model | CV macro-F1 | Test Top-1 | Test macro-F1 | Log loss | Continuity | Infer ms/1000 obs |
|---|---|---|---|---|---|---|
| Nearest-arc baseline (not ML) | N/A | 0.546 | 0.738 | N/A | 0.906 | N/A |
| M1 LogReg | 0.9105 | 0.8628 | 0.9083 | 0.1074 | 0.9607 | 0.6 |
| M2 GaussianNB | 0.8345 | 0.8328 | 0.8346 | 0.7116 | 0.9595 | 1.0 |
| M3 DecisionTree | 0.9369 | 0.8837 | 0.9279 | 0.1054 | 0.9604 | 0.5 |
| M4 RandomForest | 0.9399 | 0.8926 | 0.9321 | 0.0744 | 0.9652 | 11.5 |
| M5 HistGB | 0.9404 | 0.9016 | 0.9384 | 0.0711 | 0.9665 | 27.5 |
| M6 LightGBM | 0.9422 | 0.9098 | 0.9407 | 0.0677 | 0.9685 | 64.5 |
| M7 MLP | 0.9363 | 0.8902 | 0.9302 | 0.0794 | 0.9660 | 1.8 |

Leader: LightGBM (best on every accuracy/probability metric; mild overfit: train-sample F1 0.976 vs CV 0.942 vs test 0.941).
Why the baseline is weak: every road has two directed arcs at the same distance, so distance alone is a coin flip on
direction; the progress/heading features fix this. Remaining Top-1 errors are mostly neighbouring short arcs -> HMM target.

## Review 2 (not done yet)
Optuna Bayesian tuning of LightGBM, HMM/Viterbi (classifier-only vs +HMM), FastAPI, UI, report, README for deployment.
