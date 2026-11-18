"""Integration test: calls the running API over HTTP for every evaluation trip and checks the response contract.
Usage: python tests/run_integration.py [http://localhost:8000]"""
import json, sys, time, urllib.request
from pathlib import Path

API = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
OUT = Path(__file__).resolve().parent.parent / "results" / "api"
OUT.mkdir(parents=True, exist_ok=True)
FORBIDDEN = ("route", "ground", "truth", "label", "reference_route", "true_")
ALLOWED_EVAL = {"top1_accuracy", "classifier_only_top1_accuracy", "route_precision", "route_recall", "route_f1", "route_connected",
                "n_pred_arcs", "n_correct_points", "n_points", "route_continuity"}

def call(path, body=None):
    req = urllib.request.Request(API + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read()), time.time() - t0

def main():
    health, _ = call("/health"); trips, _ = call("/trips")
    assert health["status"] == "ok"
    rows, checks = [], []
    for t in trips["trips"]:
        d, rt = call("/predict", {"trip_id": t["trip_id"]})
        e = d["evaluation"]
        keys_ok = not any(any(f in k.lower() for f in ("ground", "truth", "true_", "reference")) for k in d)
        eval_ok = set(e) <= ALLOWED_EVAL
        len_ok = len(d["predicted_arc_ids"]) > 0 and len(d["predicted_route_arcs"]) == len(d["predicted_arc_ids"]) and d["n_gps_points"] == len(d["gps_points"])
        checks.append(keys_ok and eval_ok and len_ok)
        rows.append(dict(trip=d["trip_id"], n_points=e["n_points"], top1=round(e["top1_accuracy"], 4),
                         clf_only=round(e["classifier_only_top1_accuracy"], 4), route_f1=round(e["route_f1"], 4),
                         continuity=round(e["route_continuity"], 4), server_s=d["timing"]["total_s"], round_trip_s=round(rt, 3),
                         contract_ok=keys_ok and eval_ok and len_ok))
        print(rows[-1], flush=True)
    tot = sum(r["n_points"] for r in rows)
    agg = dict(n_trips=len(rows), n_points=tot,
               top1_pooled=round(sum(r["top1"] * r["n_points"] for r in rows) / tot, 4),
               clf_only_pooled=round(sum(r["clf_only"] * r["n_points"] for r in rows) / tot, 4),
               all_contract_checks_passed=all(checks))
    print(agg)
    json.dump(dict(per_trip=rows, aggregate=agg), open(OUT / "integration_results.json", "w"), indent=2)
    import csv
    with open(OUT / "integration_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sys.exit(0 if all(checks) else 1)

if __name__ == "__main__":
    main()
