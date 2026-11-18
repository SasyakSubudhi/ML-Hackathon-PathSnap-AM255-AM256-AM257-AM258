# PathSnap – distributed map-matching and route reconstruction

Noisy GPS observations are matched to directed road arcs. Per observation we generate candidate arcs (60 m, K=10),
score each candidate with a LightGBM classifier on 26 features, and decode a coherent route with HMM/Viterbi.
A FastAPI service (Laptop 1) runs the full pipeline; a browser UI (Laptop 2) selects one of ten trips and shows the
predicted route and scores. Ground truth is read only inside the API after prediction and is never returned.

Team: AM255, AM256, AM257, AM258 (PES University). Report: `PathSnap_Technical_Report.pdf`.

## Results (student_test, 10 trips, 19,160 observations)
| Method | Top-1 | Route continuity | Route F1 |
|---|---|---|---|
| Nearest arc baseline | 54.59% | 90.56% | – |
| LightGBM only | 90.03% | 96.78% | 90.24% |
| LightGBM + HMM/Viterbi (deployed) | 93.42% | 99.56% | 93.27% |

## Layout
- `src/pathsnap_data.py` loading, projection, candidates, reference alignment, features
- `src/build_dataset.py` cached candidate/feature tables; `src/eda.py` EDA and Folium
- `src/train_models.py` 7 models, 3-fold GroupKFold; `src/model_plots.py` plots
- `src/tune_lightgbm.py` Optuna (TPE) tuning; `src/hmm_decoder.py`, `src/run_hmm.py` HMM/Viterbi and experiments
- `src/pipeline.py` inference pipeline; `src/export_artifacts.py` writes `artifacts/`
- `src/folium_final_map.py`, `src/folium_classifier_map.py` interactive maps
- `api/app.py` FastAPI backend; `ui/index.html` evaluator UI; `tests/run_integration.py` HTTP test on all trips
- `artifacts/` saved model + `config.json`; `results/` all CSV/JSON/plots/maps/logs; `report/build_report.py` builds the PDF; `docs/` plan and roles

## Install
```
pip install -r requirements.txt
```
Python 3.10+. Dataset folder: set `PATHSNAP_DATA` to the `Student folder` (containing `train/` and `student_test/`), or place it at `data/Student folder`.

## Deploy and run (two laptops)
Laptop 1 (API):
```
set PATHSNAP_EVAL_DIR=<folder with the evaluation trips, one sub-folder per trip id>
cd api
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
(or edit and run `api\run_api.bat`). Note Laptop 1's IP address (`ipconfig`) and allow port 8000 in the firewall.
The console and `api/api_requests.log` show every request.

Laptop 2 (UI):
```
cd ui
python -m http.server 8080
```
Open `http://localhost:8080`, type `http://<Laptop-1-IP>:8000` in the address box, press Connect, pick a trip.

Check from any machine: `python tests/run_integration.py http://<Laptop-1-IP>:8000`.

## Reproduce experiments (run from `src/`)
```
python build_dataset.py 60 10
python eda.py
python train_models.py
python tune_lightgbm.py
python run_hmm.py
python export_artifacts.py
python folium_final_map.py 00000077
```
Seed 42 everywhere. Anomalous training trips 00000033 and 00000054 are excluded from fitting (see report, Section 2).
