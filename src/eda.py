"""Exploratory data analysis for PathSnap (train + student_test partitions only).
Outputs CSVs + PNGs in results/eda and an interactive Folium map in results/maps."""
import sys
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).parent))
import pathsnap_data as P

ROOT = Path(__file__).resolve().parent.parent
EDA, MAPS, CACHE = ROOT / "results" / "eda", ROOT / "results" / "maps", ROOT / "data_cache"
EDA.mkdir(parents=True, exist_ok=True)
MAPS.mkdir(parents=True, exist_ok=True)


def inventory():
    rows = []
    for part in ("train", "student_test"):
        for trip in sorted(p.name for p in (P.DATA_ROOT_DEFAULT / part).iterdir()):
            tr = P.project_trip(P.load_trip(part, trip))
            xy, t, arcs, nodes, route = tr["obs_xy"], tr["t"], tr["arcs"], tr["node_xy"], tr["route"]
            n_nodes = len(nodes)
            dt = np.diff(t)
            step = np.hypot(*np.diff(xy, axis=0).T)
            # network connectivity (weak components) + isolated nodes
            A = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n_nodes, n_nodes))
            ncomp, lab = connected_components(A, directed=False)
            deg = np.bincount(arcs.ravel(), minlength=n_nodes)
            sizes = np.bincount(lab)
            p0, p1 = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
            alen = np.hypot(*(p1 - p0).T)
            # reference route continuity (consecutive route arcs share a node)
            rt_cont = np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0]) if len(route) > 1 else 1.0
            size_m = np.hypot(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))
            rows.append(dict(
                part=part, trip=trip, n_obs=len(xy), n_nodes=n_nodes, n_arcs=len(arcs), n_route_arcs=len(route),
                missing_values=int(np.isnan(tr["track"]).sum() + np.isnan(nodes).sum()),
                time_sorted=bool((dt > 0).all()), duplicated_timestamps=int((dt == 0).sum()),
                duration_s=float(t[-1] - t[0]), frac_dt_not_1s=float((dt != 1).mean()), max_gap_s=float(dt.max()),
                n_gaps_gt10s=int((dt > 10).sum()), n_gaps_gt60s=int((dt > 60).sum()),
                median_step_m=float(np.median(step)), max_step_m=float(step.max()),
                median_speed_ms=float(np.median(step / np.maximum(dt, 1))),
                extent_diag_km=float(size_m / 1000), lon_min=float(tr["track"][:, 0].min()), lat_min=float(tr["track"][:, 1].min()),
                n_components=int(ncomp), largest_component_frac=float(sizes.max() / n_nodes),
                isolated_nodes=int((deg == 0).sum()), mean_arc_len_m=float(alen.mean()), max_arc_len_m=float(alen.max()),
                has_reverse_frac=float(np.isin(arcs[:, 1].astype(np.int64) * n_nodes + arcs[:, 0],
                                               arcs[:, 0].astype(np.int64) * n_nodes + arcs[:, 1]).mean()),
                route_consecutive_connected_frac=float(rt_cont), route_repeats_arcs=int(len(route) - len(set(route))),
            ))
    inv = pd.DataFrame(rows)
    # add reference-alignment quality from cached observation tables
    quality = []
    for part in ("train", "student_test"):
        o = pd.read_pickle(CACHE / f"{part}_obs.pkl")
        q = o.groupby("trip").agg(dref_median_m=("d_ref", "median"), dref_p95_m=("d_ref", lambda s: s.quantile(0.95)),
                                  dref_max_m=("d_ref", "max"), candidate_miss_rate=("ref_in_candidates", lambda s: 1 - s.mean()),
                                  mean_candidates=("n_cand", "mean")).reset_index()
        q["part"] = part
        quality.append(q)
    inv = inv.merge(pd.concat(quality), on=["part", "trip"], how="left")
    inv["anomalous_reference"] = inv["dref_median_m"] > 50
    inv.to_csv(EDA / "dataset_inventory_per_trip.csv", index=False)
    summ = inv.groupby("part")[["n_obs", "n_nodes", "n_arcs", "n_route_arcs", "duration_s", "max_gap_s", "frac_dt_not_1s",
                                "extent_diag_km", "n_components", "isolated_nodes", "dref_median_m", "candidate_miss_rate",
                                "route_consecutive_connected_frac"]].agg(["count", "mean", "median", "min", "max"]).T
    summ.to_csv(EDA / "dataset_summary_by_partition.csv")
    return inv


def plots(inv):
    tr = inv[inv.part == "train"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, col, ttl, log in [(ax[0, 0], "n_obs", "GPS observations per trip", True), (ax[0, 1], "n_arcs", "Road arcs per trip network", True),
                             (ax[0, 2], "max_gap_s", "Largest time gap per trip (s)", True),
                             (ax[1, 0], "frac_dt_not_1s", "Fraction of steps with dt != 1 s", False),
                             (ax[1, 1], "dref_median_m", "Median GPS-to-reference-route distance (m)", False),
                             (ax[1, 2], "candidate_miss_rate", "Candidate miss rate per trip (R=60 m, K=10)", False)]:
        for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
            v = inv[inv.part == part][col]
            a.hist(np.log10(v + 1) if log else v, bins=25, alpha=0.65, label=f"{part} ({len(v)})", color=c)
        a.set_title(ttl + (" [log10(x+1)]" if log else ""), fontsize=10)
        a.legend()
    plt.suptitle("Dataset inventory: train (80 trips) vs student_test (10 trips)")
    plt.tight_layout(); plt.savefig(EDA / "01_dataset_overview.png", dpi=130); plt.close()

    # GPS noise (distance from GPS point to its aligned reference arc) per partition
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        o = pd.read_pickle(CACHE / f"{part}_obs.pkl")
        ax[0].hist(o.d_ref.clip(upper=100), bins=100, alpha=0.6, label=part, color=c, density=True)
    ax[0].set_xlabel("distance GPS point -> reference arc (m), clipped at 100"); ax[0].set_title("GPS noise relative to reference route"); ax[0].legend()
    ot = pd.read_pickle(CACHE / "train_obs.pkl")
    ax[1].scatter(ot.groupby("trip").d_ref.median(), ot.groupby("trip").ref_in_candidates.apply(lambda s: 1 - s.mean()), c="tab:red")
    ax[1].set_xlabel("per-trip median distance to reference (m)"); ax[1].set_ylabel("candidate miss rate")
    ax[1].set_title("Train trips: anomalous trips stand out (33, 54)")
    plt.tight_layout(); plt.savefig(EDA / "02_gps_noise_and_anomalies.png", dpi=130); plt.close()

    # time-gap distribution (train, all steps)
    dts = []
    for trip in sorted(tr.trip):
        tk = pd.read_csv(P.DATA_ROOT_DEFAULT / "train" / trip / f"{trip}.track", sep=r"\s+", header=None)
        dts.append(np.diff(tk[2].to_numpy()))
    dts = np.concatenate(dts)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    vals, cnt = np.unique(np.clip(dts, 0, 12), return_counts=True)
    ax.bar(vals, cnt); ax.set_yscale("log"); ax.set_xlabel("time step dt (s), values >= 12 grouped"); ax.set_ylabel("count (log)")
    ax.set_title(f"Train sampling interval: {100*(dts==1).mean():.1f}% of steps are exactly 1 s")
    plt.tight_layout(); plt.savefig(EDA / "03_time_step_distribution.png", dpi=130); plt.close()
    pd.Series({"steps": len(dts), "frac_dt_eq_1": (dts == 1).mean(), "frac_dt_gt_10": (dts > 10).mean(),
               "frac_dt_gt_60": (dts > 60).mean(), "max_dt": dts.max()}).to_csv(EDA / "train_time_step_stats.csv")


def trip_plot_and_map(part="student_test", trip="00000000"):
    tr = P.project_trip(P.load_trip(part, trip))
    nodes_ll, arcs, route, track = tr["nodes"], tr["arcs"], tr["route"], tr["track"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    segs_x = np.c_[nodes_ll[arcs[:, 0], 0], nodes_ll[arcs[:, 1], 0]].T
    segs_y = np.c_[nodes_ll[arcs[:, 0], 1], nodes_ll[arcs[:, 1], 1]].T
    for a in ax:
        a.plot(segs_x, segs_y, color="lightgray", lw=0.4, zorder=1)
        a.set_aspect(1 / np.cos(np.radians(track[:, 1].mean())))
        a.set_xlabel("longitude"); a.set_ylabel("latitude")
    rx = np.c_[nodes_ll[arcs[route, 0], 0], nodes_ll[arcs[route, 1], 0]].T
    ry = np.c_[nodes_ll[arcs[route, 0], 1], nodes_ll[arcs[route, 1], 1]].T
    ax[0].plot(rx, ry, color="limegreen", lw=5, alpha=0.9, zorder=2)
    ax[0].scatter(track[:, 0], track[:, 1], s=1, c="blue", zorder=3)
    ax[0].set_title(f"{part}/{trip}: network (grey), reference route (green), GPS (blue)")
    pad = 0.002
    ax[1].plot(rx, ry, color="limegreen", lw=6, alpha=0.9, zorder=2)
    ax[1].scatter(track[:, 0], track[:, 1], s=3, c="blue", zorder=3)
    ax[1].set_xlim(track[:400, 0].min() - pad, track[:400, 0].max() + pad); ax[1].set_ylim(track[:400, 1].min() - pad, track[:400, 1].max() + pad)
    ax[1].set_title("zoom: first 400 GPS points")
    plt.tight_layout(); plt.savefig(EDA / f"04_trip_{part}_{trip}.png", dpi=130); plt.close()

    # Folium map: GPS + network + reference route (+ legend/layer control)
    c_lat, c_lon = track[:, 1].mean(), track[:, 0].mean()
    m = folium.Map(location=[c_lat, c_lon], zoom_start=14, tiles="OpenStreetMap")
    # network layer limited to a bbox around the track to keep the HTML light
    lo, hi = track[:, :2].min(0) - 0.003, track[:, :2].max(0) + 0.003
    inb = ((nodes_ll[arcs[:, 0], 0] > lo[0]) & (nodes_ll[arcs[:, 0], 0] < hi[0]) &
           (nodes_ll[arcs[:, 0], 1] > lo[1]) & (nodes_ll[arcs[:, 0], 1] < hi[1]))
    fg_net = folium.FeatureGroup("Road network (arcs)", show=True)
    for a in np.where(inb)[0][:6000]:
        folium.PolyLine([(nodes_ll[arcs[a, 0], 1], nodes_ll[arcs[a, 0], 0]), (nodes_ll[arcs[a, 1], 1], nodes_ll[arcs[a, 1], 0])],
                        color="#888888", weight=1, opacity=0.6).add_to(fg_net)
    fg_ref = folium.FeatureGroup("Reference route (.route)", show=True)
    for a in route:
        folium.PolyLine([(nodes_ll[arcs[a, 0], 1], nodes_ll[arcs[a, 0], 0]), (nodes_ll[arcs[a, 1], 1], nodes_ll[arcs[a, 1], 0])],
                        color="green", weight=5, opacity=0.9).add_to(fg_ref)
    fg_gps = folium.FeatureGroup("Raw GPS points (.track)", show=True)
    step = max(1, len(track) // 1500)
    for i in range(0, len(track), step):
        folium.CircleMarker((track[i, 1], track[i, 0]), radius=2, color="blue", fill=True, fill_opacity=0.8,
                            tooltip=f"obs {i}, t={track[i,2]:.0f}s").add_to(fg_gps)
    folium.Marker((track[0, 1], track[0, 0]), tooltip="start", icon=folium.Icon(color="green")).add_to(fg_gps)
    folium.Marker((track[-1, 1], track[-1, 0]), tooltip="end", icon=folium.Icon(color="red")).add_to(fg_gps)
    for g in (fg_net, fg_ref, fg_gps):
        g.add_to(m)
    folium.LayerControl().add_to(m)
    legend = ('<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px;border:1px solid #888;font-size:13px">'
              f'<b>PathSnap {part}/{trip}</b><br><span style="color:blue">&#9679;</span> raw GPS<br>'
              '<span style="color:green">&#9632;</span> reference route<br><span style="color:#888">&#9632;</span> road network</div>')
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([[track[:, 1].min(), track[:, 0].min()], [track[:, 1].max(), track[:, 0].max()]])
    m.save(MAPS / f"eda_{part}_{trip}.html")


if __name__ == "__main__":
    inv = inventory()
    plots(inv)
    trip_plot_and_map("student_test", "00000000")
    trip_plot_and_map("train", "00000001")
    print(inv.groupby("part").size())
    print(inv[inv.anomalous_reference][["part", "trip", "n_obs", "dref_median_m", "candidate_miss_rate"]])
    print("trips with isolated nodes:", int((inv.isolated_nodes > 0).sum()), "| trips with >1 component:", int((inv.n_components > 1).sum()),
          "| trips with missing values:", int((inv.missing_values > 0).sum()))
    print("route consecutive-arc connectivity (min over trips):", inv.route_consecutive_connected_frac.min().round(4))
