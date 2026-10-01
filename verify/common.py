"""Shared helpers for the verification scripts (standard library only)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

Z_LIMIT = 5.0
SCALE = float(os.environ.get("VERIFY_SCALE", "1"))


def scaled(reps):
    return max(1000, int(reps * SCALE))


def z_value(count, prob, reps):
    var = reps * prob * (1.0 - prob)
    if var <= 0.0:
        return 0.0
    return (count - reps * prob) / math.sqrt(var)


def report(title, rows, reps, z_limit=Z_LIMIT, show_rows=None):
    """rows: list of (label, observed_count, exact_prob_as_float).

    Prints observed vs expected frequencies and a z-score per row.
    Returns (ok, max_abs_z)."""
    print(title)
    print("-" * len(title))
    print("repetitions: %d" % reps)
    print("%-16s %12s %12s %12s %9s" % ("item", "observed", "expected", "obs-exp", "z"))
    worst = 0.0
    worst_label = None
    shown = 0
    for label, count, prob in rows:
        obs = count / reps
        z = z_value(count, prob, reps)
        if abs(z) > worst:
            worst, worst_label = abs(z), label
        if show_rows is None or shown < show_rows:
            print("%-16s %12.6f %12.6f %+12.6f %9.2f" % (label, obs, prob, obs - prob, z))
            shown += 1
    if show_rows is not None and len(rows) > show_rows:
        print("... (%d rows omitted)" % (len(rows) - show_rows))
    ok = worst < z_limit
    print("max |z| = %.2f (item %s, limit %.1f) -> %s"
          % (worst, worst_label, z_limit, "PASS" if ok else "FAIL"))
    print()
    return ok, worst
