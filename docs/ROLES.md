# Team roles (4 people) – each person explains one part

Everyone should know the one-line story: noisy GPS -> candidate arcs near each point -> classifier scores each
candidate -> (Review 2) HMM/Viterbi picks a connected route -> API on Laptop 1, UI on Laptop 2.

## Person 1 – Data and geography
Owns: `src/eda.py`, `src/pathsnap_data.py` (load_trip, project_trip), `results/eda/`, `results/maps/eda_*.html`
Explain: the four file types (.track lon/lat/t, .nodes, .arcs with row = arc_id, .route arc ids); each trip has its own
network; why metres (local transverse Mercator) and not degrees; missing values (none), gaps (up to 153,064 s), network
components, the two anomalous train trips (33, 54) and why they are excluded; the Folium map layers.
Numbers: 80/10 trips; 208,417 / 19,160 points; 96.5% steps at 1 s; median GPS error ~5 m.
Likely questions: Why not Euclidean in degrees? How do you know trips 33/54 are bad? Is the route true ground truth?
(No – reference only; derived labels have boundary noise.) Is there leakage? (Split is by trip; evaluator data untouched.)
Review 2: report sections 2.2 and 6 (EDA, Folium).

## Person 2 – Candidates, labels and features
Owns: `gen_candidates`, `align_route`, feature code in `src/pathsnap_data.py`, `src/build_dataset.py`
Explain: KD-tree on points sampled every 15 m + exact point-to-segment distance; radius 60 m, K=10 and the candidate miss
rate (0.26% test); the monotone alignment that gives each observation a reference arc (DP, route index never decreases);
each feature group and why (progress and heading resolve direction; context distances use neighbours).
Numbers: 7.35 candidates/obs; top gains d_minus_min 31%, progress5 27%, rank_d 12%.
Likely questions: Why does nearest-arc only get 0.546 Top-1? (two directed arcs per road). Why neighbour features, are
they leakage? (No – they use GPS only, never the route.) What if the true arc is not a candidate? (counts as an error.)
Review 2: feature export for the API; HMM emission uses candidate sets.

## Person 3 – Models and evaluation
Owns: `src/train_models.py`, `src/model_plots.py`, `results/models/`
Explain: the 7 models and families; GroupKFold by trip (no trip split across folds); fit capped at 200k rows for fairness;
metrics (candidate macro-F1, observation Top-1, log loss, continuity, timing); why LightGBM leads; overfitting evidence
(train 0.976 vs CV 0.942 vs test 0.941); GaussianNB's poor log loss (0.71, independence assumption); confusion matrices.
Numbers: see table in REVIEW1_SUMMARY.md (LightGBM test Top-1 0.910, F1 0.941, log loss 0.068).
Likely questions: Why macro-F1 here? Why is Top-1 lower than F1? (imbalance, one positive among ~7 candidates, direction
ambiguity). Why not choose by train score? Where are the epoch curves? (only LightGBM rounds and MLP epochs exist).
Review 2: Optuna Bayesian search, tuned vs untuned table, model section of the report.

## Person 4 – System, decoder and delivery
Owns: `docs/PLAN.md`, README, route-continuity metric in `train_models.py`, `src/folium_classifier_map.py`; Review 2:
HMM/Viterbi, FastAPI, UI, deployment
Explain: the two-laptop architecture (API + model + ground truth on Laptop 1, UI on Laptop 2, truth never sent to UI);
why classifier-only routes need an HMM (continuity 0.9685 for the best model, errors on neighbouring short arcs); the
planned design: states = candidate arcs, emission = classifier probability, transition = network connectivity/hops,
gap handling; the classifier-only map (trip 00000076, Top-1 0.937).
Likely questions: What is route continuity exactly? (consecutive points with dt<=5 s: same arc or reachable within 2 hops.)
What happens at big GPS gaps or disconnected candidates? How do you prevent hardcoding? (API computes from the trip ID.)
Review 2: HMM, API, UI, integration test on all 10 trips, report sections 5 and 7, public repo with README.
