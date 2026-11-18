"""Folium map: raw GPS + reference route + CLASSIFIER-ONLY predicted arcs (LightGBM) for one student_test trip.
Usage: python folium_classifier_map.py [trip_id]   (default: the student_test trip with median length)"""
import sys
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import pathsnap_data as P

ROOT = Path(__file__).resolve().parent.parent
CACHE, MAPS, M = ROOT / "data_cache", ROOT / "results" / "maps", ROOT / "results" / "models"

cand = pd.read_pickle(CACHE / "student_test_cand.pkl")
obs = pd.read_pickle(CACHE / "student_test_obs.pkl")
sizes = obs.groupby("trip").size().sort_values()
trip = sys.argv[1] if len(sys.argv) > 1 else sizes.index[len(sizes) // 2]
c, o = cand[cand.trip == trip].copy(), obs[obs.trip == trip].reset_index(drop=True)
model = joblib.load(M / "M6_LightGBM.joblib")
c["p"] = model.predict_proba(c[P.FEATURES].to_numpy(np.float32))[:, 1]
top = c.sort_values(["obs", "p"], ascending=[True, False]).drop_duplicates("obs").set_index("obs")
o["pred_arc"] = top["arc"].reindex(o["obs"]).fillna(-1).astype(int).to_numpy()
o["pred_p"] = top["p"].reindex(o["obs"]).to_numpy()
o["correct"] = o["pred_arc"] == o["ref_arc"]
acc = o["correct"].mean()

tr = P.load_trip("student_test", trip)
nodes, arcs = tr["nodes"], tr["arcs"]


def seg(a):
    return [(nodes[arcs[a, 0], 1], nodes[arcs[a, 0], 0]), (nodes[arcs[a, 1], 1], nodes[arcs[a, 1], 0])]


m = folium.Map(location=[o.lat.mean(), o.lon.mean()], zoom_start=14)
fg_ref = folium.FeatureGroup("Reference route (green)", show=True)
for a in tr["route"]:
    folium.PolyLine(seg(a), color="green", weight=8, opacity=0.55).add_to(fg_ref)
fg_pred = folium.FeatureGroup("Classifier-only predicted arcs (red)", show=True)
for a in sorted(set(o.pred_arc[o.pred_arc >= 0])):
    folium.PolyLine(seg(a), color="red", weight=3, opacity=0.9).add_to(fg_pred)
fg_gps = folium.FeatureGroup("Raw GPS (blue)", show=True)
fg_err = folium.FeatureGroup("GPS points where Top-1 arc != reference (orange)", show=True)
step = max(1, len(o) // 1500)
for i in range(0, len(o), step):
    r = o.iloc[i]
    folium.CircleMarker((r.lat, r.lon), radius=2, color="blue", fill=True, fill_opacity=0.8,
                        tooltip=f"obs {int(r.obs)} t={r.t:.0f}s pred={r.pred_arc} ref={r.ref_arc} p={r.pred_p:.2f}").add_to(fg_gps)
    if not r.correct:
        folium.CircleMarker((r.lat, r.lon), radius=4, color="orange", fill=True, fill_opacity=0.9).add_to(fg_err)
for g in (fg_ref, fg_pred, fg_gps, fg_err):
    g.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.get_root().html.add_child(folium.Element(
    '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px;border:1px solid #888;font-size:13px">'
    f'<b>student_test/{trip}</b> (classifier only, LightGBM)<br>Top-1 accuracy on this trip: {acc:.3f}<br>'
    '<span style="color:blue">&#9679;</span> raw GPS &nbsp;<span style="color:green">&#9632;</span> reference route<br>'
    '<span style="color:red">&#9632;</span> predicted arcs &nbsp;<span style="color:orange">&#9679;</span> wrong Top-1</div>'))
m.fit_bounds([[o.lat.min(), o.lon.min()], [o.lat.max(), o.lon.max()]])
out = MAPS / f"classifier_only_student_test_{trip}.html"
m.save(out)
print(trip, "obs", len(o), "top1", round(acc, 4), "->", out)
