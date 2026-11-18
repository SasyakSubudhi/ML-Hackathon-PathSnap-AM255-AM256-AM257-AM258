@echo off
REM Laptop 2: serves the UI at http://localhost:8080 ; then open it and type the API address of Laptop 1.
cd /d %~dp0
python -m http.server 8080
