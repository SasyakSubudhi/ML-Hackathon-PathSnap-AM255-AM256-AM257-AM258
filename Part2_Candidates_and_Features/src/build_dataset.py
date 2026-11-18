"""Build candidate+feature tables for every trip (train and student_test) and cache them.
Usage: python build_dataset.py [radius_m] [k_max]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import pathsnap_data as P

CACHE = Path(__file__).resolve().parent.parent / "data_cache"
RADIUS = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
K_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 10


def work(args):
    part, trip = args
    t0 = time.time()
    df, obs = P.build_trip_table(P.load_trip(part, trip), radius=RADIUS, k_max=K_MAX)
    return part, trip, df, obs, time.time() - t0


if __name__ == "__main__":
    jobs = []
    for part in ("train", "student_test"):
        for t in sorted(p.name for p in (P.DATA_ROOT_DEFAULT / part).iterdir()):
            jobs.append((part, t))
    # biggest trips first is not known cheaply; imap_unordered keeps both cores busy
    cands, obss = {"train": [], "student_test": []}, {"train": [], "student_test": []}
    t0 = time.time()
    with Pool(2) as pool:
        for k, (part, trip, df, obs, dt) in enumerate(pool.imap_unordered(work, jobs), 1):
            cands[part].append(df)
            obss[part].append(obs)
            if k % 10 == 0:
                print(f"{k}/{len(jobs)} done, {time.time()-t0:.0f}s (last {trip}: {dt:.1f}s)", flush=True)
    for part in cands:
        c = pd.concat(cands[part], ignore_index=True).sort_values(["trip", "obs", "rank_d"], ignore_index=True)
        o = pd.concat(obss[part], ignore_index=True).sort_values(["trip", "obs"], ignore_index=True)
        c.to_pickle(CACHE / f"{part}_cand.pkl")
        o.to_pickle(CACHE / f"{part}_obs.pkl")
        print(part, "candidate rows", len(c), "observations", len(o),
              "candidate miss rate", round(1 - o.ref_in_candidates.mean(), 5))
    (CACHE / "build_config.txt").write_text(f"radius_m={RADIUS}\nk_max={K_MAX}\n")
    print("total", round(time.time() - t0), "s")
