"""Prediction pipeline used by the API (raw trip files -> predicted arc per GPS point).

The prediction path reads ONLY .track, .nodes and .arcs. The reference route is loaded by a separate
class (ServerSideReference) that the API calls after the prediction has been produced.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np

import hmm_decoder as H
import pathsnap_data as P

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


class PathSnapPredictor:
    def __init__(self, artifacts=ARTIFACTS):
        artifacts = Path(artifacts)
        self.model = joblib.load(artifacts / "model.joblib")
        cfg = json.load(open(artifacts / "config.json"))
        self.cfg = cfg
        self.radius, self.k_max = cfg["candidate_radius_m"], cfg["candidate_k_max"]
        self.decoder = cfg["decoder_params"]
        self.features = cfg["features"]
        assert self.features == P.FEATURES, "feature list in config does not match pathsnap_data.FEATURES"

    def predict(self, trip_dir, trip_id):
        t0 = time.time()
        tr = P.load_trip("api", trip_id, trip_dir=trip_dir, with_route=False)
        cand, obs = P.build_trip_table(tr, radius=self.radius, k_max=self.k_max, use_reference=False)
        t1 = time.time()
        p = self.model.predict_proba(cand[self.features].to_numpy(np.float32))[:, 1]
        t2 = time.time()
        g = H.TripGraph(tr["arcs"], tr["node_xy"])
        prep = H.Prepared(g, cand["obs"].to_numpy(), cand["arc"].to_numpy(), cand["frac"].to_numpy(),
                          tr["obs_xy"][:, 0], tr["obs_xy"][:, 1], tr["t"])
        clf = prep.classifier_only(p, fill=True)
        hmm = prep.decode(p, **self.decoder)
        t3 = time.time()
        return dict(tr=tr, graph=g, clf_pred=clf, hmm_pred=hmm, n_candidates=len(cand),
                    timing=dict(features_s=t1 - t0, model_s=t2 - t1, hmm_s=t3 - t2, total_s=t3 - t0))


class ServerSideReference:
    """Ground truth. Only the API's evaluation step touches this; it is never passed to the predictor or the UI."""

    def __init__(self, eval_dir):
        self.eval_dir = Path(eval_dir)

    def evaluate(self, trip_id, pred, tr, graph):
        route = np.loadtxt(self.eval_dir / trip_id / f"{trip_id}.route", dtype=np.int64, ndmin=1)
        p0, p1 = tr["node_xy"][tr["arcs"][:, 0]], tr["node_xy"][tr["arcs"][:, 1]]
        ridx, d_ref = P.align_route(tr["obs_xy"], p0[route], p1[route])
        ref = route[ridx]
        res = dict(top1_accuracy=float((pred["hmm_pred"] == ref).mean()),
                   classifier_only_top1_accuracy=float((pred["clf_pred"] == ref).mean()))
        res.update({k: (float(v) if not isinstance(v, int) else v) for k, v in H.route_metrics(pred["hmm_pred"], route, graph).items()})
        n = len(ref)
        res["n_correct_points"] = int((pred["hmm_pred"] == ref).sum())
        res["n_points"] = n
        res["route_continuity"] = float(H.route_continuity(pred["hmm_pred"], tr["t"], graph))
        return res
