# PathSnap – Two-Half Plan

Source: the 5 hackathon PDFs (Main Guidelines, Deliverables, Dataset Exploration, Models and Training Guidelines, Assessment Policy).
Rule: nothing is pushed to GitHub until the team decides to (everything stays local).

## What the data actually is (verified from the files)
- 80 train trips, 10 student_test trips. Each trip folder has `.track` (lon, lat, t), `.nodes` (lon, lat), `.arcs` (from_node, to_node; row index = arc_id), `.route` (arc_ids in order).
- Every trip has its OWN road network (different city per trip), so there is no global map.
- No per-observation label is given; `.route` is only the ordered list of arcs. We derive an observation-level reference arc by monotone alignment (documented limitation).
- 1 Hz nominal, but some trips have gaps up to 153,064 s (anomalies to document). No NaNs.

## Problem framing we use
For each GPS observation: generate candidate arcs near it -> score each candidate with a classifier (is this the reference arc?) -> pick Top-1 (classifier only) -> later smooth with HMM/Viterbi over the network.

## FIRST HALF – Review 1 (12:15)  — DONE LOCALLY
1. Understand problem + read all PDFs
2. Dataset inventory + EDA (counts, gaps, extents, network size, connectivity, route stats, missing values)
3. Geography: per-trip local metric projection (not degrees)
4. Candidate generation (radius + top-K) and Candidate Miss Rate
5. Observation-level reference label (monotone alignment of track to route)
6. Feature engineering (distance, heading, speed, topology, neighbour-obs context)
7. Train & compare 5+ models with group-aware CV (GroupKFold by trip) + student_test evaluation
8. Folium map (GPS + network + reference route) + plots + saved CSV/logs

## SECOND HALF – Review 2
1. Pick final model from table; Optuna Bayesian optimisation (group CV on 80 trips), tuned vs untuned
2. HMM + Viterbi (states = candidate arcs, emission from classifier prob, transition from network connectivity/shortest path), compare classifier-only vs +HMM
3. Export artifacts (model, config)
4. FastAPI backend (Laptop 1) – ground truth kept server-side
5. UI (Laptop 2) – 10 evaluation trips, calls API, shows route + accuracy
6. Integration test on all student_test trips
7. Technical report PDF + README + deployment instructions, push to public GitHub

## Rules to respect
- Evaluator-test data never used for training/selection. Student_test is for development comparison only.
- Fit scalers/imputers on train only. Group CV by trip (no trip split across folds).
- Report real numbers only; every figure produced by the project code.
