"""Side-by-side comparison of percentile / BC / BCa on one skewed sample.

    python3 example.py
"""

from bootstrap_ci import (
    bootstrap,
    make_quantile_stat,
    make_ratio_stat,
    make_variance_stat,
    stat_mean,
    stat_median,
)

# Small, right-skewed sample (e.g. session durations in seconds).
SAMPLE = [12.1, 8.4, 45.2, 9.9, 15.3, 7.6, 11.0, 102.4, 13.8, 10.2,
          9.1, 22.7, 8.8, 14.4, 31.5]


def show(name, res):
    print("%-12s theta_hat=%8.3f" % (name, res.theta_hat))
    for label, ci in [("percentile", res.percentile_ci()),
                      ("BC", res.bc_ci()),
                      ("BCa", res.bca_ci()),
                      ("normal-approx", res.normal_ci())]:
        print("    %-13s 95%% CI = [%8.3f, %8.3f]  width=%.3f"
              % (label, ci[0], ci[1], ci[1] - ci[0]))


def main():
    print("sample (n=%d): %s\n" % (len(SAMPLE), SAMPLE))
    show("mean", bootstrap(SAMPLE, stat_mean, 20000, seed=2026))
    show("median", bootstrap(SAMPLE, stat_median, 20000, seed=2026))
    show("variance", bootstrap(SAMPLE, make_variance_stat(), 20000, seed=2026))
    show("q90", bootstrap(SAMPLE, make_quantile_stat(0.9), 20000, seed=2026))

    control = [9.8, 11.2, 10.5, 8.9, 12.3, 10.1, 9.4, 11.8, 10.9, 9.2,
               10.7, 11.5, 9.9, 10.3, 12.0]
    show("ratio mean/control",
         bootstrap((SAMPLE, control), make_ratio_stat(), 20000, seed=2026))


if __name__ == "__main__":
    main()
