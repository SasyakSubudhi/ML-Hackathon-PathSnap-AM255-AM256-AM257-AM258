#!/bin/bash
# Laptop 1: starts the PathSnap API (set PATHSNAP_EVAL_DIR to the folder containing the evaluation trips).
cd "$(dirname "$0")"
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
