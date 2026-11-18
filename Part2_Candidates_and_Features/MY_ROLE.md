# Team roles (4 people) – each person explains one part

Everyone should know the one-line story: noisy GPS -> candidate arcs near each point -> classifier scores each
candidate -> (Review 2) HMM/Viterbi picks a connected route -> API on Laptop 1, UI on Laptop 2.

## Person 2 – Candidates, labels and features
Owns: `gen_candidates`, `align_route`, feature code in `src/pathsnap_data.py`, `src/build_dataset.py`
Explain: KD-tree on points sampled every 15 m + exact point-to-segment distance; radius 60 m, K=10 and the candidate miss
rate (0.26% test); the monotone alignment that gives each observation a reference arc (DP, route index never decreases);
each feature group and why (progress and heading resolve direction; context distances use neighbours).
Numbers: 7.35 candidates/obs; top gains d_minus_min 31%, progress5 27%, rank_d 12%.
Likely questions: Why does nearest-arc only get 0.546 Top-1? (two directed arcs per road). Why neighbour features, are
they leakage? (No – they use GPS only, never the route.) What if the true arc is not a candidate? (counts as an error.)
Review 2: feature export for the API; HMM emission uses candidate sets.
