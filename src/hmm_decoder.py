"""HMM / Viterbi route reconstruction for PathSnap.

Hidden states   : the candidate road arcs of each GPS observation (up to K=10 per observation).
Emission        : classifier probability of the candidate (log of the normalised score), weight `alpha`.
Transition      : Newson-Krumm style. For consecutive observations the network distance between the
                  two candidate positions is compared with the straight GPS displacement:
                      log T = -|d_route - d_gps| / beta  - uturn_pen * [arc reversed] - switch_pen * [arc changed]
                  Unreachable pairs (not connected within `max_dist` metres) get log T = -floor.
Gaps            : if two usable observations are more than `gap_s` seconds apart the chain is restarted
                  (transition term constant), so a long gap never forces a fake connection.
No candidates   : observations without any candidate arc are skipped by the chain and take the arc decoded
                  for the nearest previous usable observation (or the next one at the start of a trip).
"""
import heapq

import numpy as np

MAX_DIST = 600.0


class TripGraph:
    """Directed arc graph of one trip with lazily cached bounded Dijkstra from arc end-nodes."""

    def __init__(self, arcs, node_xy, max_dist=MAX_DIST):
        self.frm = arcs[:, 0].astype(np.int64)
        self.to = arcs[:, 1].astype(np.int64)
        seg = node_xy[self.to] - node_xy[self.frm]
        self.len = np.hypot(seg[:, 0], seg[:, 1])
        n_nodes = int(max(self.frm.max(), self.to.max())) + 1
        order = np.argsort(self.frm, kind="stable")
        self.out_arc = order
        self.out_ptr = np.zeros(n_nodes + 1, np.int64)
        np.cumsum(np.bincount(self.frm, minlength=n_nodes), out=self.out_ptr[1:])
        self.max_dist = max_dist
        self._cache = {}

    def reach(self, node):
        r = self._cache.get(node)
        if r is not None:
            return r
        dist = {node: 0.0}
        heap = [(0.0, node)]
        out_arc, out_ptr, to, ln, lim = self.out_arc, self.out_ptr, self.to, self.len, self.max_dist
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, 1e18):
                continue
            for a in out_arc[out_ptr[u]:out_ptr[u + 1]]:
                nd = d + ln[a]
                if nd <= lim:
                    v = to[a]
                    if nd < dist.get(v, 1e18):
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
        self._cache[node] = dist
        return dist


class Prepared:
    """Everything that does not depend on the decoder parameters (computed once per trip)."""

    def __init__(self, graph, cand_obs, cand_arc, cand_frac, obs_x, obs_y, obs_t):
        g = graph
        self.n_obs = len(obs_x)
        starts = np.flatnonzero(np.r_[True, cand_obs[1:] != cand_obs[:-1]]) if len(cand_obs) else np.array([], np.int64)
        ends = np.r_[starts[1:], len(cand_obs)] if len(starts) else starts
        self.starts, self.ends = starts, ends
        self.step_obs = cand_obs[starts] if len(starts) else starts
        self.cand_arc = cand_arc
        n = len(starts)
        self.dt = np.zeros(n)
        self.AD, self.REACH, self.U, self.S = [None] * n, [None] * n, [None] * n, [None] * n
        for j in range(1, n):
            s0, e0, s1, e1 = starts[j - 1], ends[j - 1], starts[j], ends[j]
            oa, ob = self.step_obs[j - 1], self.step_obs[j]
            self.dt[j] = obs_t[ob] - obs_t[oa]
            dgps = float(np.hypot(obs_x[ob] - obs_x[oa], obs_y[ob] - obs_y[oa]))
            A, FA = cand_arc[s0:e0], cand_frac[s0:e0]
            B, FB = cand_arc[s1:e1], cand_frac[s1:e1]
            ad = np.full((len(A), len(B)), 1e6)
            reach = np.zeros((len(A), len(B)), bool)
            uturn = np.zeros((len(A), len(B)), bool)
            same = np.zeros((len(A), len(B)), bool)
            frm_b = g.frm[B]
            for i, a in enumerate(A):
                nd = g.reach(g.to[a])
                head = (1.0 - FA[i]) * g.len[a]
                for k, b in enumerate(B):
                    if b == a:
                        r = (FB[k] - FA[i]) * g.len[a]
                        same[i, k] = True
                    else:
                        gp = nd.get(frm_b[k])
                        if gp is None:
                            continue
                        r = head + gp + FB[k] * g.len[b]
                        if g.frm[b] == g.to[a] and g.to[b] == g.frm[a]:
                            uturn[i, k] = True
                    reach[i, k] = True
                    ad[i, k] = abs(r - dgps)
            self.AD[j], self.REACH[j], self.U[j], self.S[j] = ad, reach, uturn, same

    def emissions(self, p, normalize=True):
        q = np.asarray(p, float) + 1e-4
        if normalize and len(self.starts):
            sums = np.add.reduceat(q, self.starts)
            q = q / np.repeat(sums, self.ends - self.starts)
        return np.log(np.clip(q, 1e-6, 1.0))

    def classifier_only(self, p, fill=False):
        """Top-1 candidate per observation (-1 where no candidate exists unless fill=True)."""
        pred = np.full(self.n_obs, -1, np.int64)
        for j, (s, e) in enumerate(zip(self.starts, self.ends)):
            pred[self.step_obs[j]] = self.cand_arc[s + int(np.argmax(p[s:e]))]
        return fill_missing(pred) if fill else pred

    def decode(self, p, alpha=1.0, beta=10.0, uturn_pen=3.0, switch_pen=0.0, floor=12.0, gap_s=30.0, normalize=True):
        n = len(self.starts)
        pred = np.full(self.n_obs, -1, np.int64)
        if n == 0:
            return pred
        lp = alpha * self.emissions(p, normalize)
        starts, ends = self.starts, self.ends
        score = lp[starts[0]:ends[0]].copy()
        back = [None] * n
        for j in range(1, n):
            em = lp[starts[j]:ends[j]]
            if self.dt[j] > gap_s:
                bp = np.full(len(em), int(np.argmax(score)))
                new = score.max() + em
            else:
                trans = -(self.AD[j] / beta) - uturn_pen * self.U[j] - switch_pen * (~self.S[j])
                trans = np.where(self.REACH[j], trans, -floor)
                cand = score[:, None] + trans
                bp = cand.argmax(0)
                new = cand[bp, np.arange(cand.shape[1])] + em
            back[j] = bp
            score = new - new.max()
        k = int(np.argmax(score))
        for j in range(n - 1, -1, -1):
            pred[self.step_obs[j]] = self.cand_arc[starts[j] + k]
            if j > 0:
                k = int(back[j][k])
        return fill_missing(pred)


def fill_missing(pred):
    """Fill -1 entries with the previous decoded arc (leading ones with the first decoded arc)."""
    pred = pred.copy()
    valid = np.flatnonzero(pred >= 0)
    if len(valid) == 0:
        return pred
    idx = np.searchsorted(valid, np.arange(len(pred)), side="right") - 1
    idx = np.maximum(idx, 0)
    return pred[valid[idx]]


# ------------------------------------------------------------------------------------- metrics
def route_metrics(pred, route_arcs, graph):
    """Sequence-level quality: collapse repeated arcs, compare arc sets with the reference route,
    and report how often consecutive distinct arcs are physically connected."""
    seq = pred[pred >= 0]
    if len(seq) == 0:
        return dict(route_precision=0.0, route_recall=0.0, route_f1=0.0, route_connected=0.0, n_pred_arcs=0)
    seq = seq[np.r_[True, seq[1:] != seq[:-1]]]
    ps, rs = set(seq.tolist()), set(np.asarray(route_arcs).tolist())
    tp = len(ps & rs)
    prec, rec = tp / len(ps), tp / len(rs)
    f1 = 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
    conn = float(np.mean(graph.to[seq[:-1]] == graph.frm[seq[1:]])) if len(seq) > 1 else 1.0
    return dict(route_precision=prec, route_recall=rec, route_f1=f1, route_connected=conn, n_pred_arcs=len(seq))


def route_continuity(pred, t, graph, max_dt=5.0):
    """Share of consecutive observations (dt <= max_dt s) whose arcs are identical or reachable within 2 forward hops."""
    a, b = pred[:-1], pred[1:]
    el = (a >= 0) & (b >= 0) & (np.diff(t) <= max_dt)
    if not el.any():
        return 1.0
    aa, bb = a[el], b[el]
    ok = (aa == bb) | (graph.to[aa] == graph.frm[bb])
    for k in np.where(~ok)[0]:
        node = graph.to[aa[k]]
        for m in graph.out_arc[graph.out_ptr[node]:graph.out_ptr[node + 1]]:
            if graph.to[m] == graph.frm[bb[k]]:
                ok[k] = True
                break
    return float(ok.mean())
