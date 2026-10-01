#!/bin/sh
# Run the full regression suite (stdlib unittest only).
# Reproduction tests -> tests/test_repro.py
# Edge/contract tests -> tests/test_edges.py
# Randomized differential data -> results/differential_report.json
# Restart-recovery data -> results/restart_report.json
cd "$(dirname "$0")"
exec python3 -m unittest discover -s tests -p 'test_*.py' -v
