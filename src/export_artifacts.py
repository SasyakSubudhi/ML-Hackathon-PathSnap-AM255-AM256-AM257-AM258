"""Collect the deployable model + configuration into artifacts/ (model.joblib, config.json)."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "artifacts"
A.mkdir(exist_ok=True)
bp = json.load(open(ROOT / "results" / "tuning" / "best_params.json"))
dp = json.load(open(ROOT / "results" / "hmm" / "decoder_params.json"))
bc = dict(l.strip().split("=") for l in (ROOT / "data_cache" / "build_config.txt").read_text().splitlines())
import sys
sys.path.insert(0, str(ROOT / "src"))
import pathsnap_data as P
shutil.copy(ROOT / "results" / "models" / "LightGBM_tuned.joblib", A / "model.joblib")
json.dump(dict(model="LightGBM (Optuna-tuned) + HMM/Viterbi", lightgbm_params=bp["best_params"], features=P.FEATURES,
               candidate_radius_m=float(bc["radius_m"]), candidate_k_max=int(bc["k_max"]), decoder_params=dp["best_params"],
               decoder_max_dist_m=dp["max_dist_m"], seed=42,
               training_data="78 clean training trips (trips 00000033 and 00000054 excluded); student_test and evaluator trips not used for fitting"),
          open(A / "config.json", "w"), indent=2)
print("exported", [p.name for p in A.iterdir()])
