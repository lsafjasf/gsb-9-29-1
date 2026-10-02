#!/bin/sh
# Run the full decaying-counter test suite (Python 3, stdlib only).
# FUZZ_TRIALS controls the number of differential trials (default 400).
set -e
cd "$(dirname "$0")"
python3 -m unittest discover -s tests -v
