"""Empirical error-rate simulation for sequential_test.BernoulliSPRT."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from typing import Any

from sequential_test import (
    BernoulliSPRT,
    Decision,
    MultiBernoulliSPRT,
)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    center = (phat + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(phat * (1.0 - phat) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def quantile(values: list[int], probability: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(probability * len(ordered))) - 1))
    return ordered[index]


def summarize(
    name: str,
    *,
    trials: int,
    positives: int,
    stop_n: list[int],
    event_rate_name: str,
    target_rate: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    low, high = wilson_interval(positives, trials)
    return {
        "scenario": name,
        "trials": trials,
        event_rate_name: positives / trials,
        "wilson_95_ci": [low, high],
        "target_rate": target_rate,
        "target_satisfied": positives / trials <= target_rate,
        "average_stop_n": sum(stop_n) / len(stop_n),
        "median_stop_n": quantile(stop_n, 0.50),
        "p95_stop_n": quantile(stop_n, 0.95),
        "max_stop_n": max(stop_n),
        "configured_max_n": max(stop_n),
        **metadata,
    }


def run_single_scenario(
    name: str,
    *,
    true_p: float,
    p0: float,
    p1: float,
    alpha: float,
    beta: float,
    trials: int,
    seed: int,
    event_rate_name: str,
) -> dict[str, Any]:
    rng = random.Random(seed)
    template = BernoulliSPRT(p0=p0, p1=p1, alpha=alpha, beta=beta)
    positives = 0
    no_positives = 0
    early_futility = 0
    max_n_hits = 0
    stop_n: list[int] = []

    for _ in range(trials):
        test = BernoulliSPRT(
            p0=p0,
            p1=p1,
            alpha=alpha,
            beta=beta,
            max_n=template.max_n,
        )
        while not test.stopped:
            test.update(1 if rng.random() < true_p else 0)
        if test.decision is Decision.POSITIVE:
            positives += 1
        else:
            no_positives += 1
            if test.n >= test.max_n:
                max_n_hits += 1
            else:
                early_futility += 1
        stop_n.append(test.n)

    if event_rate_name == "false_positive_rate":
        event_count = positives
    else:
        event_count = no_positives

    return summarize(
        name,
        trials=trials,
        positives=event_count,
        stop_n=stop_n,
        event_rate_name=event_rate_name,
        target_rate=alpha if event_rate_name == "false_positive_rate" else beta,
        metadata={
            "true_p": true_p,
            "p0": p0,
            "p1": p1,
            "alpha": alpha,
            "beta": beta,
            "configured_max_n": template.max_n,
            "actual_positive_rate": positives / trials,
            "actual_no_positive_rate": no_positives / trials,
            "early_futility_rate": early_futility / trials,
            "max_n_hit_rate": max_n_hits / trials,
        },
    )


def run_multi_null_scenario(
    *,
    trials: int,
    metrics: int,
    family_alpha: float,
    beta: float,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    configs = [(f"metric_{index + 1}", 0.2, 0.3) for index in range(metrics)]
    template = MultiBernoulliSPRT(configs, family_alpha=family_alpha, beta=beta)
    false_families = 0
    stop_n = []

    for _ in range(trials):
        suite = MultiBernoulliSPRT(configs, family_alpha=family_alpha, beta=beta)
        while not suite.stopped:
            row = [1 if rng.random() < 0.2 else 0 for _ in range(metrics)]
            suite.update(row)
        if suite.decision is Decision.POSITIVE:
            false_families += 1
        stop_n.append(suite.n)

    return summarize(
        f"null_multi_k{metrics}",
        trials=trials,
        positives=false_families,
        stop_n=stop_n,
        event_rate_name="family_wise_false_positive_rate",
        target_rate=family_alpha,
        metadata={
            "metrics": metrics,
            "family_alpha": family_alpha,
            "per_test_alpha": template.per_test_alpha,
            "beta": beta,
            "configured_max_n": template.max_n,
        },
    )


def run_multi_alt_scenario(
    *,
    trials: int,
    metrics: int,
    family_alpha: float,
    beta: float,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    configs = [(f"metric_{index + 1}", 0.2, 0.3) for index in range(metrics)]
    template = MultiBernoulliSPRT(configs, family_alpha=family_alpha, beta=beta)
    no_positive_families = 0
    stop_n = []

    for _ in range(trials):
        suite = MultiBernoulliSPRT(configs, family_alpha=family_alpha, beta=beta)
        while not suite.stopped:
            row = [1 if rng.random() < 0.3 else 0 for _ in range(metrics)]
            suite.update(row)
        if suite.decision is not Decision.POSITIVE:
            no_positive_families += 1
        stop_n.append(suite.n)

    return summarize(
        f"alt_multi_k{metrics}",
        trials=trials,
        positives=no_positive_families,
        stop_n=stop_n,
        event_rate_name="family_no_positive_rate",
        target_rate=beta,
        metadata={
            "metrics": metrics,
            "family_alpha": family_alpha,
            "per_test_alpha": template.per_test_alpha,
            "beta": beta,
            "configured_max_n": template.max_n,
            "actual_family_positive_rate": 1.0 - no_positive_families / trials,
        },
    )


def run_all(trials: int, base_seed: int) -> dict[str, Any]:
    scenarios = [
        run_single_scenario(
            "null_single",
            true_p=0.2,
            p0=0.2,
            p1=0.3,
            alpha=0.05,
            beta=0.20,
            trials=trials,
            seed=base_seed + 1,
            event_rate_name="false_positive_rate",
        ),
        run_multi_null_scenario(
            trials=trials,
            metrics=5,
            family_alpha=0.05,
            beta=0.20,
            seed=base_seed + 2,
        ),
        run_single_scenario(
            "alt_single",
            true_p=0.3,
            p0=0.2,
            p1=0.3,
            alpha=0.05,
            beta=0.20,
            trials=trials,
            seed=base_seed + 3,
            event_rate_name="false_negative_rate",
        ),
        run_multi_alt_scenario(
            trials=trials,
            metrics=5,
            family_alpha=0.05,
            beta=0.20,
            seed=base_seed + 4,
        ),
        run_single_scenario(
            "negative_effect_single",
            true_p=0.15,
            p0=0.2,
            p1=0.3,
            alpha=0.05,
            beta=0.20,
            trials=trials,
            seed=base_seed + 5,
            event_rate_name="false_positive_rate",
        ),
    ]
    return {
        "method": "Truncated one-sided Bernoulli SPRT/e-value test",
        "trials_per_scenario": trials,
        "random_source": "random.Random(base_seed + scenario_offset)",
        "base_seed": base_seed,
        "scenarios": scenarios,
    }


def write_csv(results: dict[str, Any], path: str) -> None:
    fields = [
        "scenario",
        "trials",
        "false_positive_rate",
        "family_wise_false_positive_rate",
        "false_negative_rate",
        "family_no_positive_rate",
        "target_rate",
        "target_satisfied",
        "wilson_95_ci_low",
        "wilson_95_ci_high",
        "average_stop_n",
        "median_stop_n",
        "p95_stop_n",
        "max_stop_n",
        "configured_max_n",
        "alpha",
        "family_alpha",
        "per_test_alpha",
        "beta",
        "metrics",
        "true_p",
        "p0",
        "p1",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scenario in results["scenarios"]:
            row = {field: scenario.get(field, "") for field in fields}
            row["wilson_95_ci_low"], row["wilson_95_ci_high"] = scenario[
                "wilson_95_ci"
            ]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--json", default="simulation_results.json")
    parser.add_argument("--csv", default="simulation_results.csv")
    args = parser.parse_args()

    results = run_all(args.trials, args.seed)
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    write_csv(results, args.csv)

    print(f"wrote {args.json} and {args.csv}")
    for scenario in results["scenarios"]:
        event_name = next(
            key
            for key in (
                "false_positive_rate",
                "family_wise_false_positive_rate",
                "false_negative_rate",
                "family_no_positive_rate",
            )
            if key in scenario
        )
        low, high = scenario["wilson_95_ci"]
        print(
            f"{scenario['scenario']}: {event_name}="
            f"{scenario[event_name]:.4f} (95% CI {low:.4f}-{high:.4f}), "
            f"target={scenario['target_rate']:.3f}, "
            f"mean N={scenario['average_stop_n']:.1f}, "
            f"max N={scenario['configured_max_n']}"
        )


if __name__ == "__main__":
    main()
