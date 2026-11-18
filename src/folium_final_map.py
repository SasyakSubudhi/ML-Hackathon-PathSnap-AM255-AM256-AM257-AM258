"""Final Folium map for one student_test trip: raw GPS, road network around the trip, reference route,
classifier-only prediction and classifier + HMM/Viterbi prediction, with layer control and legend.
Usage: python folium_final_map.py [trip_id]   (default 00000077, the hardest dev trip)"""
import sys
from pathlib import Path

import folium
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pathsnap_data as P
from pipeline import PathSnapPredictor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "maps"
trip = sys.argv[1] if len(sys.argv) > 1 else "00000077"
eval_dir = Path(P.DATA_ROOT_DEFAULT) / "student_test"

pred = PathSnapPredictor().predict(eval_dir / trip, trip)       # prediction path never reads the reference route
ref = P.load_trip("student_test", trip)                          # reference route is loaded only to draw the comparison layer
nodes, arcs = ref["nodes"], ref["arcs"]
ref_arcs = np.asarray(ref["route"])
clf, hmm = pred["clf_pred"], pred["hmm_pred"]
tr = pred["tr"]


def seg(a):
    return [(nodes[arcs[a, 0], 1], nodes[arcs[a, 0], 0]), (nodes[arcs[a, 1], 1], nodes[arcs[a, 1], 0])]


def ordered_unique(x):
    seen, out = set(), []
    for a in x:
        if a >= 0 and a not in seen:
            seen.add(a); out.append(int(a))
    return out


lon, lat = ref["track"][:, 0], ref["track"][:, 1]
m = folium.Map(location=[float(np.mean(lat)), float(np.mean(lon))], zoom_start=13, tiles="OpenStreetMap")

# road network near the trip (arcs with an end node within 150 m of the track bounding box)
pad = 0.0015
la0, la1, lo0, lo1 = lat.min() - pad, lat.max() + pad, lon.min() - pad, lon.max() + pad
mid_lat = (nodes[arcs[:, 0], 1] + nodes[arcs[:, 1], 1]) / 2
mid_lon = (nodes[arcs[:, 0], 0] + nodes[arcs[:, 1], 0]) / 2
inb = np.where((mid_lat > la0) & (mid_lat < la1) & (mid_lon > lo0) & (mid_lon < lo1))[0]
if len(inb) > 6000:
    inb = inb[:: int(np.ceil(len(inb) / 6000))]
fg_net = folium.FeatureGroup("Road network (grey)", show=False)
for a in inb:
    folium.PolyLine(seg(a), color="#777", weight=1, opacity=0.6).add_to(fg_net)
fg_ref = folium.FeatureGroup("Reference route (green)", show=True)
for a in ordered_unique(ref_arcs):
    folium.PolyLine(seg(a), color="green", weight=9, opacity=0.5).add_to(fg_ref)
fg_clf = folium.FeatureGroup("Classifier-only predicted arcs (orange)", show=False)
for a in ordered_unique(clf):
    folium.PolyLine(seg(a), color="orange", weight=4, opacity=0.95).add_to(fg_clf)
fg_hmm = folium.FeatureGroup("Classifier + HMM/Viterbi predicted route (red)", show=True)
for a in ordered_unique(hmm):
    folium.PolyLine(seg(a), color="red", weight=4, opacity=0.95).add_to(fg_hmm)
fg_gps = folium.FeatureGroup("Raw GPS observations (blue)", show=True)
step = max(1, len(lat) // 1500)
for i in range(0, len(lat), step):
    folium.CircleMarker((float(lat[i]), float(lon[i])), radius=1.5, color="blue", weight=1, fill=True, fill_opacity=0.5,
                        tooltip=f"obs {i}  clf arc {int(clf[i])}  HMM arc {int(hmm[i])}").add_to(fg_gps)
folium.Marker((float(lat[0]), float(lon[0])), tooltip="Start", icon=folium.Icon(color="green")).add_to(fg_gps)
folium.Marker((float(lat[-1]), float(lon[-1])), tooltip="End", icon=folium.Icon(color="black")).add_to(fg_gps)
for g in (fg_net, fg_gps, fg_ref, fg_clf, fg_hmm):
    g.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
m.get_root().html.add_child(folium.Element(
    '<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;border:1px solid #888;font-size:13px">'
    f'<b>student_test / trip {trip}</b><br>'
    '<span style="color:blue">&#9679;</span> raw GPS &nbsp; <span style="color:green">&#9632;</span> reference route<br>'
    '<span style="color:orange">&#9632;</span> classifier only &nbsp; <span style="color:red">&#9632;</span> classifier + HMM</div>'))
OUT.mkdir(parents=True, exist_ok=True)
m.save(OUT / f"final_student_test_{trip}.html")
print("saved", OUT / f"final_student_test_{trip}.html", "| network arcs drawn:", len(inb))
