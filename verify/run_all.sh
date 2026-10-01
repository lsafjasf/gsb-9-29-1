#!/usr/bin/env bash
# Runs all verification scripts and stores their output in results/.
# Set VERIFY_SCALE=0.1 (or 10) to shrink (or enlarge) sample sizes.
set -u
cd "$(dirname "$0")/.."
mkdir -p results
overall=0
for name in verify_uniform verify_weighted_one verify_ares verify_pps verify_merge verify_edge_cases; do
    python3 "verify/$name.py" > "results/$name.txt" 2>&1
    code=$?
    cat "results/$name.txt"
    echo
    if grep -q "OVERALL: FAIL" "results/$name.txt" || [ $code -ne 0 ]; then
        overall=1
    fi
done
if [ $overall -eq 0 ]; then
    echo "ALL VERIFICATIONS PASSED"
else
    echo "SOME VERIFICATIONS FAILED"
fi
exit $overall
