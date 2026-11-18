"""PathSnap data layer: raw loading, metric projection, candidate generation,
reference alignment and per-candidate feature engineering.

Coordinate convention: raw files are (longitude, latitude). All distance maths is done
in a per-trip local Transverse-Mercator projection (metres), never in degrees.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

DATA_ROOT_DEFAULT = Path(os.environ.get("PATHSNAP_DATA", Path(__file__).resolve().parent.parent / "data" / "Student folder"))


# ----------------------------------------------------------------------------- loading
def _read(path, dtype):
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype).to_numpy()


def load_trip(part, trip, root=DATA_ROOT_DEFAULT):
    d = Path(root) / part / trip
    track = _read(d / f"{trip}.track", np.float64)      # lon, lat, t(s)
    nodes = _read(d / f"{trip}.nodes", np.float64)      # lon, lat
    arcs = _read(d / f"{trip}.arcs", np.int64)          # from_node, to_node
    route = _read(d / f"{trip}.route", np.int64).ravel()  # arc ids
    return dict(part=part, trip=trip, track=track, nodes=nodes, arcs=arcs, route=route)


def make_projector(lon0, lat0):
    """Local transverse Mercator centred on the trip (metres). Avoids UTM zone borders."""
    crs = CRS.from_proj4(
        f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def project_trip(tr):
    lon0, lat0 = np.median(tr["track"][:, 0]), np.median(tr["track"][:, 1])
    proj = make_projector(lon0, lat0)
    ox, oy = proj.transform(tr["track"][:, 0], tr["track"][:, 1])
    nx, ny = proj.transform(tr["nodes"][:, 0], tr["nodes"][:, 1])
    tr["obs_xy"] = np.column_stack([ox, oy])
    tr["node_xy"] = np.column_stack([nx, ny])
    tr["t"] = tr["track"][:, 2]
    tr["center"] = (lon0, lat0)
    return tr


# ----------------------------------------------------------------------------- geometry
def point_seg(px, py, x0, y0, x1, y1):
    """Distance from points to segments. Returns (dist, clipped_fraction, unclipped_fraction)."""
    dx, dy = x1 - x0, y1 - y0
    l2 = np.maximum(dx * dx + dy * dy, 1e-9)
    s = ((px - x0) * dx + (py - y0) * dy) / l2
    t = np.clip(s, 0.0, 1.0)
    return np.hypot(px - (x0 + t * dx), py - (y0 + t * dy)), t, s


def densify(p0, p1, step):
    vec = p1 - p0
    length = np.hypot(vec[:, 0], vec[:, 1])
    c = np.maximum(np.ceil(length / step).astype(np.int64), 1)
    n_pts = c + 1
    arc_idx = np.repeat(np.arange(len(length)), n_pts)
    start = np.cumsum(n_pts) - n_pts
    k = np.arange(n_pts.sum()) - np.repeat(start, n_pts)
    t = k / np.repeat(c, n_pts)
    return p0[arc_idx] + vec[arc_idx] * t[:, None], arc_idx


# ----------------------------------------------------------------------------- candidates
def gen_candidates(obs_xy, p0, p1, radius=100.0, k_max=12, step=15.0, ksamp=48):
    """For each observation return up to k_max nearest arcs within `radius` metres.
    Spatial index: KD-tree over points sampled every `step` m along every arc, then exact
    point-to-segment distances. Returns arrays obs_idx, arc_id, dist, frac, s_unclipped."""
    pts, aidx = densify(p0, p1, step)
    tree = cKDTree(pts)
    k = min(ksamp, len(pts))
    dist, idx = tree.query(obs_xy, k=k, distance_upper_bound=radius + step)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    valid = idx < len(pts)
    ids = np.where(valid, aidx[np.minimum(idx, len(pts) - 1)], -1)
    order = np.argsort(ids, axis=1, kind="stable")
    ids = np.take_along_axis(ids, order, 1)
    bad = ids < 0
    bad[:, 1:] |= ids[:, 1:] == ids[:, :-1]          # duplicate arcs (several samples of one arc)
    safe = np.maximum(ids, 0)
    d, t, s = point_seg(obs_xy[:, 0:1], obs_xy[:, 1:2], p0[safe, 0], p0[safe, 1], p1[safe, 0], p1[safe, 1])
    d = np.where(bad | (d > radius), np.inf, d)
    sel = np.argsort(d, axis=1)[:, :k_max]
    d = np.take_along_axis(d, sel, 1)
    ids = np.take_along_axis(ids, sel, 1)
    t = np.take_along_axis(t, sel, 1)
    s = np.take_along_axis(s, sel, 1)
    keep = np.isfinite(d)
    oi = np.repeat(np.arange(len(obs_xy)), keep.sum(1))
    return oi, ids[keep], d[keep], t[keep], s[keep]


# ----------------------------------------------------------------------------- reference alignment
def align_route(obs_xy, p0r, p1r):
    """Reference arc per observation: monotone (non-decreasing route index) minimum-total-distance
    alignment of the track to the supplied route. Returns (route_index_per_obs, dist_to_ref)."""
    n, m = len(obs_xy), len(p0r)
    D = np.empty((n, m), np.float32)
    for a in range(0, n, 1500):
        b = min(n, a + 1500)
        D[a:b] = point_seg(obs_xy[a:b, 0:1], obs_xy[a:b, 1:2], p0r[None, :, 0], p0r[None, :, 1],
                           p1r[None, :, 0], p1r[None, :, 1])[0]
    bdt = np.int16 if m < 32000 else np.int32
    back = np.zeros((n, m), bdt)
    ar = np.arange(m)
    dp = D[0].astype(np.float64)
    for i in range(1, n):
        pm = np.minimum.accumulate(dp)
        back[i] = np.maximum.accumulate(np.where(dp <= pm, ar, 0))
        dp = D[i] + pm
    j = int(np.argmin(dp))
    ridx = np.empty(n, np.int64)
    for i in range(n - 1, -1, -1):
        ridx[i] = j
        j = int(back[i][j])
    return ridx, D[np.arange(n), ridx].astype(np.float64)


# ----------------------------------------------------------------------------- features
FEATURES = [
    "d", "rank_d", "d_min", "d_minus_min", "d_ratio_min", "n_cand", "frac", "arc_len",
    "ang_diff", "cos_diff", "heading_valid", "speed", "dt_prev", "dt_next",
    "d_prev2", "d_next2", "d_prev5", "d_next5", "progress2", "progress5",
    "ctx_mean", "ctx_minus_min", "ctx_rank", "to_outdeg", "from_indeg", "has_reverse",
]


def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def build_trip_table(tr, radius=100.0, k_max=12):
    """Full pipeline for one trip. Returns (candidate_table, observation_table)."""
    tr = project_trip(tr)
    nxy, arcs, oxy, t = tr["node_xy"], tr["arcs"], tr["obs_xy"], tr["t"]
    n_obs, n_arc, n_node = len(oxy), len(arcs), len(nxy)
    p0, p1 = nxy[arcs[:, 0]], nxy[arcs[:, 1]]
    seg = p1 - p0
    arc_len = np.hypot(seg[:, 0], seg[:, 1])
    arc_bearing = np.degrees(np.arctan2(seg[:, 0], seg[:, 1])) % 360.0

    # --- reference label per observation (derived from supplied route)
    route = tr["route"]
    ridx, d_ref = align_route(oxy, p0[route], p1[route])
    ref_arc = route[ridx]

    # --- observation-level kinematics (central difference, +-2 samples)
    i = np.arange(n_obs)
    j1, j0 = np.minimum(i + 2, n_obs - 1), np.maximum(i - 2, 0)
    dtw = np.maximum(t[j1] - t[j0], 1.0)
    vel = (oxy[j1] - oxy[j0]) / dtw[:, None]
    speed = np.hypot(vel[:, 0], vel[:, 1])
    trk_bearing = np.degrees(np.arctan2(vel[:, 0], vel[:, 1])) % 360.0
    dt_prev = np.r_[0.0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0.0]

    # --- candidates
    oi, aid, d, frac, s = gen_candidates(oxy, p0, p1, radius=radius, k_max=k_max)
    df = pd.DataFrame({"obs": oi, "arc": aid, "d": d, "frac": frac})
    df["arc_len"] = arc_len[aid]
    g = df.groupby("obs")["d"]
    df["rank_d"] = g.rank(method="first").astype(np.int16)
    df["d_min"] = g.transform("min")
    df["d_minus_min"] = df["d"] - df["d_min"]
    df["d_ratio_min"] = df["d"] / (df["d_min"] + 1.0)
    df["n_cand"] = g.transform("size")
    df["speed"] = speed[oi]
    df["heading_valid"] = (speed[oi] > 1.0).astype(np.int8)
    ad = _wrap180(arc_bearing[aid] - trk_bearing[oi])
    df["ang_diff"] = np.where(df["heading_valid"] == 1, np.abs(ad), 90.0)
    df["cos_diff"] = np.where(df["heading_valid"] == 1, np.cos(np.radians(ad)), 0.0)
    df["dt_prev"] = dt_prev[oi]
    df["dt_next"] = dt_next[oi]

    # neighbour-observation context relative to the same arc (+-2, +-5 samples)
    s_at = {}
    for off in (2, 5):
        for sign, nm in ((-1, "prev"), (1, "next")):
            jj = np.clip(oi + sign * off, 0, n_obs - 1)
            dd, _, ss = point_seg(oxy[jj, 0], oxy[jj, 1], p0[aid, 0], p0[aid, 1], p1[aid, 0], p1[aid, 1])
            df[f"d_{nm}{off}"] = dd
            s_at[(nm, off)] = ss
    for off in (2, 5):
        df[f"progress{off}"] = np.clip((s_at[("next", off)] - s_at[("prev", off)]) * arc_len[aid], -300, 300)
    ctx = df[["d", "d_prev2", "d_next2", "d_prev5", "d_next5"]].mean(axis=1)
    df["ctx_mean"] = ctx
    gc = ctx.groupby(df["obs"])
    df["ctx_minus_min"] = ctx - gc.transform("min")
    df["ctx_rank"] = gc.rank(method="first").astype(np.int16)

    # topology (road-class proxies)
    outdeg = np.bincount(arcs[:, 0], minlength=n_node)
    indeg = np.bincount(arcs[:, 1], minlength=n_node)
    df["to_outdeg"] = outdeg[arcs[aid, 1]]
    df["from_indeg"] = indeg[arcs[aid, 0]]
    fwd = arcs[:, 0].astype(np.int64) * n_node + arcs[:, 1]
    rev = arcs[:, 1].astype(np.int64) * n_node + arcs[:, 0]
    df["has_reverse"] = np.isin(rev, fwd)[aid].astype(np.int8)

    # label + bookkeeping
    df["y"] = (aid == ref_arc[oi]).astype(np.int8)
    df["trip"] = tr["trip"]
    df["part"] = tr["part"]

    obs = pd.DataFrame({
        "trip": tr["trip"], "part": tr["part"], "obs": i, "t": t,
        "lon": tr["track"][:, 0], "lat": tr["track"][:, 1], "x": oxy[:, 0], "y": oxy[:, 1],
        "ref_arc": ref_arc, "d_ref": d_ref, "route_idx": ridx,
        "speed": speed, "dt_prev": dt_prev,
    })
    has_ref = df.groupby("obs")["y"].max().reindex(i).fillna(0).astype(bool).to_numpy()
    obs["ref_in_candidates"] = has_ref
    obs["n_cand"] = np.bincount(oi, minlength=n_obs)
    return df, obs
