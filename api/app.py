"""PathSnap FastAPI backend (runs on Laptop 1).

Run:  uvicorn app:app --host 0.0.0.0 --port 8000        (from the api/ folder)
Env : PATHSNAP_EVAL_DIR  folder that contains one sub-folder per trip (<id>/<id>.track|.nodes|.arcs|.route)

Pipeline per request: candidate generation -> LightGBM scoring -> HMM/Viterbi decoding.
Ground truth (.route) is read only inside ServerSideReference.evaluate() AFTER the prediction exists and
is never returned to the client (only aggregate scores are).
"""
import logging
import os
import re
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pipeline  # noqa: E402

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "Student folder" / "student_test"))
LOG_FILE = Path(__file__).resolve().parent / "api_requests.log"

log = logging.getLogger("pathsnap")
log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
for h in (logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")):
    h.setFormatter(fmt)
    log.addHandler(h)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = pipeline.PathSnapPredictor()
reference = pipeline.ServerSideReference(EVAL_DIR)
log.info(f"PathSnap API ready | model={predictor.cfg['model']} | decoder={predictor.decoder} | trips dir={EVAL_DIR}")


class PredictRequest(BaseModel):
    trip_id: str


def _trip_ids():
    return sorted(p.name for p in EVAL_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


@app.get("/")
def root():
    return {"service": "PathSnap API", "endpoints": ["/health", "/trips", "POST /predict {trip_id}"]}


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.cfg["model"], "n_trips": len(_trip_ids())}


@app.get("/trips")
def trips():
    out = []
    for t in _trip_ids():
        with open(EVAL_DIR / t / f"{t}.track") as f:
            n = sum(1 for _ in f)
        out.append({"trip_id": t, "n_gps_points": n})
    log.info(f"GET /trips -> {len(out)} trips")
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    t_req = time.time()
    trip_id = req.trip_id.strip()
    log.info(f"POST /predict received trip_id={trip_id}")
    if not re.fullmatch(r"\d{1,16}", trip_id) or trip_id not in _trip_ids():
        log.info(f"  -> 404 unknown trip_id={trip_id}")
        raise HTTPException(status_code=404, detail=f"unknown trip_id {trip_id}")
    pred = predictor.predict(EVAL_DIR / trip_id, trip_id)
    tm = pred["timing"]
    log.info(f"  pipeline: {len(pred['hmm_pred'])} GPS points, {pred['n_candidates']} candidate arcs | "
             f"features {tm['features_s']:.2f}s model {tm['model_s']:.2f}s HMM {tm['hmm_s']:.2f}s")
    ev = reference.evaluate(trip_id, pred, pred["tr"], pred["graph"])      # server-side ground truth
    tr, g, hp = pred["tr"], pred["graph"], pred["hmm_pred"]
    seq = hp[np.r_[True, hp[1:] != hp[:-1]]]
    nodes = tr["nodes"]
    arcs_ll = [[float(nodes[g.frm[a], 1]), float(nodes[g.frm[a], 0]), float(nodes[g.to[a], 1]), float(nodes[g.to[a], 0])] for a in seq]
    step = max(1, len(tr["track"]) // 3000)
    gps = [[float(r[1]), float(r[0])] for r in tr["track"][::step]]
    total = time.time() - t_req
    log.info(f"  evaluation (server-side): HMM top1={ev['top1_accuracy']:.4f} classifier-only top1={ev['classifier_only_top1_accuracy']:.4f} "
             f"route F1={ev['route_f1']:.4f} | total {total:.2f}s -> 200")
    return {"trip_id": trip_id, "pipeline": predictor.cfg["model"], "n_gps_points": len(tr["track"]),
            "gps_points": gps, "predicted_arc_ids": [int(a) for a in seq], "predicted_route_arcs": arcs_ll,
            "evaluation": ev, "timing": {**{k: round(v, 3) for k, v in tm.items()}, "request_total_s": round(total, 3)}}
