"""Coverage simulation: normal-approximation vs percentile vs BC vs BCa.

For each scenario (distribution x sample size) we draw R independent
samples from a known distribution, build a nominal 95% CI for the mean
with each method, and record how often the interval contains the true
value.  The uncertainty of each empirical coverage is reported as a
Wilson score interval.  Everything is seeded and fully reproducible:

    python3 coverage_simulation.py            # writes coverage_results.md
"""

import math
import random
from statistics import NormalDist

from bootstrap_ci import bootstrap, stat_mean

NOMINAL = 0.95
R = 1000          # Monte Carlo replications per scenario
B = 999           # bootstrap resamples per replication
_Z = NormalDist().inv_cdf(0.975)


def wilson_interval(k: int, n: int) -> tuple:
    p = k / n
    denom = 1.0 + _Z ** 2 / n
    center = (p + _Z ** 2 / (2 * n)) / denom
    half = _Z * math.sqrt(p * (1 - p) / n + _Z ** 2 / (4 * n * n)) / denom
    return center - half, center + half


def make_sampler(kind: str):
    def normal(rng):
        return rng.gauss(0.0, 1.0)

    def exponential(rng):
        return rng.expovariate(1.0)

    def bimodal(rng):
        return rng.gauss(-2.0 if rng.random() < 0.5 else 2.0, 1.0)

    def outliers(rng):
        # 90% standard normal + 10% heavy contamination
        return rng.gauss(0.0, 1.0) if rng.random() < 0.9 else rng.gauss(0.0, 10.0)

    return {"normal": normal, "exponential": exponential,
            "bimodal": bimodal, "outliers": outliers}[kind]


TRUE_MEAN = {"normal": 0.0, "exponential": 1.0, "bimodal": 0.0, "outliers": 0.0}

SCENARIOS = [
    ("normal", 15), ("normal", 50),
    ("exponential", 15), ("exponential", 50),
    ("bimodal", 15), ("bimodal", 50),
    ("outliers", 15), ("outliers", 50),
]

METHODS = ["normal", "percentile", "bc", "bca"]


def run_scenario(kind: str, n: int, seed: int) -> dict:
    sampler = make_sampler(kind)
    true_value = TRUE_MEAN[kind]
    rng = random.Random(seed)
    hits = {m: 0 for m in METHODS}
    widths = {m: 0.0 for m in METHODS}
    for rep in range(R):
        sample = [sampler(rng) for _ in range(n)]
        res = bootstrap(sample, stat_mean, B, seed=rng.randrange(1 << 63))
        cis = {
            "normal": res.normal_ci(NOMINAL),
            "percentile": res.percentile_ci(NOMINAL),
            "bc": res.bc_ci(NOMINAL),
            "bca": res.bca_ci(NOMINAL),
        }
        for m, (lo, hi) in cis.items():
            if lo <= true_value <= hi:
                hits[m] += 1
            widths[m] += hi - lo
    return {
        "kind": kind, "n": n, "true": true_value,
        "coverage": {m: hits[m] / R for m in METHODS},
        "wilson": {m: wilson_interval(hits[m], R) for m in METHODS},
        "mean_width": {m: widths[m] / R for m in METHODS},
    }


def main() -> None:
    rows = []
    for i, (kind, n) in enumerate(SCENARIOS):
        result = run_scenario(kind, n, seed=20261003 + i)
        rows.append(result)
        print("done: %-12s n=%-3d" % (kind, n), flush=True)

    lines = []
    lines.append("# Coverage simulation results")
    lines.append("")
    lines.append("Nominal level: %.0f%%.  R = %d Monte Carlo replications, "
                 "B = %d bootstrap resamples, statistic = sample mean."
                 % (NOMINAL * 100, R, B))
    lines.append("Coverage CI = Wilson score interval for the empirical coverage.")
    lines.append("Reproduce with `python3 coverage_simulation.py` "
                 "(fixed seeds; results are deterministic).")
    lines.append("")
    header = "| distribution | n | method | coverage | Wilson 95% CI | mean width |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        for m in METHODS:
            cov = row["coverage"][m]
            lo, hi = row["wilson"][m]
            lines.append("| %s | %d | %s | %.3f | [%.3f, %.3f] | %.3f |"
                         % (row["kind"], row["n"], m, cov, lo, hi,
                            row["mean_width"][m]))
    lines.append("")
    text = "\n".join(lines)
    with open("coverage_results.md", "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
