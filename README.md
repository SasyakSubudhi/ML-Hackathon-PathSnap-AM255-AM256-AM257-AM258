# PathSnap – map-matching of noisy GPS (work in progress: Review 1 state)

Match noisy GPS observations to road-network arcs. Per observation we generate candidate arcs, score each
candidate with a classifier, and (Review 2) decode a coherent route with HMM/Viterbi behind a FastAPI + UI.

## Status
| Stage | State |
|---|---|
| Dataset exploration, projection, data-quality log | done |
| Candidate generation, reference alignment, features | done |
| 7 models, group CV, student_test comparison | done |
| Bayesian optimisation (Optuna), HMM/Viterbi, API, UI, report | Review 2 |

## Layout
- `src/pathsnap_data.py` – loading, local metric projection, candidate generation, reference alignment, features
- `src/build_dataset.py` – builds cached tables for all trips (`python build_dataset.py 60 10`)
- `src/eda.py` – inventory CSVs, EDA plots, Folium maps
- `src/train_models.py` – 7-model comparison (3-fold GroupKFold by trip + student_test)
- `src/model_plots.py`, `src/folium_classifier_map.py` – plots and the classifier-only map
- `results/eda`, `results/models`, `results/maps` – every number/figure comes from these scripts
- `docs/PLAN.md`, `docs/ROLES.md`, `docs/REVIEW1_SUMMARY.md`

## Run (Review 1 part)
```
pip install -r requirements_review1.txt   # versions used
export PATHSNAP_DATA="<path to 'Student folder'>"   # contains train/ and student_test/
cd src
python build_dataset.py 60 10
python eda.py
python train_models.py
python model_plots.py
python folium_classifier_map.py
```
Seeds are fixed (42); rerunning reproduces the reported metrics.
