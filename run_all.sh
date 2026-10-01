#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "== unit tests =="
python3 -m unittest -v test_ranking_metrics.py
echo "== differential test vs reference =="
python3 difftest.py
echo "== bootstrap confidence intervals =="
python3 demo_ci.py
