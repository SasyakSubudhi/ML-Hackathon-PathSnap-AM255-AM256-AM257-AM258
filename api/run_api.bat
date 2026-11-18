@echo off
REM Laptop 1: starts the PathSnap API. Edit the path below to the folder with the 10 evaluation trips.
set PATHSNAP_EVAL_DIR=C:\path\to\evaluation_trips
cd /d %~dp0
python -m uvicorn app:app --host 0.0.0.0 --port 8000
