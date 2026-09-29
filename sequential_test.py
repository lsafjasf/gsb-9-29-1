"""Sequentially monitored Bernoulli tests with explicit stopping boundaries.

The module uses only the Python standard library.  It does not use a random
source itself; simulations and callers inject observations deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import math
from typing import Iterable, Mapping, Sequence


class Decision(str, Enum):
    CONTINUE = "continue"
    POSITIVE = "positive"
    NO_POSITIVE = "no_positive"


@dataclass(frozen=True)
class Snapshot:
    n: int
    log_e_value: float
    decision: Decision
    upper_log_boundary: float
    lower_log_boundary: float
    max_n: int


def _validate_probability(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not 0.0 < float(value) < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1")


def bernoulli_kl_divergence(observed_p: float, reference_p: float) -> float:
    """Return KL(Bern(observed_p) || Bern(reference_p))."""
    _validate_probability("observed_p", observed_p)
    _validate_probability("reference_p", reference_p)
    return (
        observed_p * math.log(observed_p / reference_p)
        + (1.0 - observed_p) * math.log((1.0 - observed_p) / (1.0 - reference_p))
    )


def conservative_max_n(
    p0: float,
    p1: float,
    alpha: float = 0.05,
    beta: float = 0.20,
    futility_budget: float | None = None,
) -> int:
    """Return a simple closed-form guaranteed maximum sample size.

    The upper rejection boundary is A=1/alpha.  The lower futility boundary
    uses B=futility_budget; the default is beta/2.  Hoeffding's inequality for
    bounded log-likelihood increments bounds the probability of reaching the
    truncation point without a positive decision by beta-futility_budget.
    """
    _validate_probability("p0", p0)
    _validate_probability("p1", p1)
    if p0 == p1:
        raise ValueError("p0 and p1 must differ")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be strictly between 0 and 1")
    if futility_budget is None:
        futility_budget = beta / 2.0
    if not 0.0 < futility_budget < beta:
        raise ValueError("futility_budget must be strictly between 0 and beta")

    increment_width = abs(math.log(p1 / p0) - math.log((1.0 - p1) / (1.0 - p0)))
    expected_alt_increment = bernoulli_kl_divergence(p1, p0)
    upper_log = math.log(1.0 / alpha)
    residual_budget = beta - futility_budget
    residual_log = math.log(1.0 / residual_budget)

    # Solve the equality from Hoeffding's inequality:
    # 2 N (D(p1||p0) - log(A)/N)^2 / width^2 = log(1/(beta-B)).
    linear_coefficient = (
        4.0 * expected_alt_increment * upper_log
        + residual_log * increment_width * increment_width
    )
    discriminant = (
        linear_coefficient * linear_coefficient
        - 16.0
        * expected_alt_increment
        * expected_alt_increment
        * upper_log
        * upper_log
    )
    root = (
        linear_coefficient + math.sqrt(max(0.0, discriminant))
    ) / (4.0 * expected_alt_increment * expected_alt_increment)
    return max(1, int(math.ceil(root - 1e-12)))


def _log_binomial_probability(n: int, k: int, p: float) -> float:
    if k < 0 or k > n:
        return float("-inf")
    log_combinations = (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    )
    if k == 0:
        success_term = 0.0
    else:
        success_term = k * math.log(p)
    if k == n:
        failure_term = 0.0
    else:
        failure_term = (n - k) * math.log(1.0 - p)
    return log_combinations + success_term + failure_term


def truncated_residual_probability(n: int, p0: float, p1: float, alpha: float) -> float:
    """Return P_H1(S_n < log(1/alpha), no earlier boundaries assumed)."""
    upper_log = math.log(1.0 / alpha)
    if p1 > p0:
        success_increment = math.log(p1 / p0)
        failure_increment = math.log((1.0 - p1) / (1.0 - p0))
        boundary_count = (upper_log - failure_increment * n) / (
            success_increment - failure_increment
        )
        max_successes = min(n, math.ceil(boundary_count) - 1)
        if max_successes < 0:
            return 0.0
        log_terms = [
            _log_binomial_probability(n, k, p1) for k in range(max_successes + 1)
        ]
    else:
        success_increment = math.log(p1 / p0)
        failure_increment = math.log((1.0 - p1) / (1.0 - p0))
        boundary_count = (upper_log - success_increment * n) / (
            failure_increment - success_increment
        )
        max_failures = min(n, math.ceil(boundary_count) - 1)
        if max_failures < 0:
            return 0.0
        log_terms = [
            _log_binomial_probability(n, n - k, p1) for k in range(max_failures + 1)
        ]

    max_log_term = max(log_terms)
    return min(1.0, math.exp(max_log_term) * sum(math.exp(term - max_log_term) for term in log_terms))


@lru_cache(maxsize=64)
def exact_max_n(
    p0: float,
    p1: float,
    alpha: float = 0.05,
    beta: float = 0.20,
    futility_budget: float | None = None,
    *,
    search_limit: int = 1_000_000,
) -> int:
    """Return the smallest finite truncation satisfying the residual power bound.

    At N the test can miss H1 only by reaching N below the rejection boundary.
    Enumerating the corresponding binomial tail gives the exact residual
    probability for the specified simple alternative.
    """
    _validate_probability("p0", p0)
    _validate_probability("p1", p1)
    if p0 == p1:
        raise ValueError("p0 and p1 must differ")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be strictly between 0 and 1")
    if futility_budget is None:
        futility_budget = beta / 2.0
    if not 0.0 < futility_budget < beta:
        raise ValueError("futility_budget must be strictly between 0 and beta")
    residual_budget = beta - futility_budget
    for n in range(1, search_limit + 1):
        if (
            truncated_residual_probability(n, p0, p1, alpha)
            <= residual_budget
        ):
            return n
    raise RuntimeError("exact max_n search reached search_limit")


class BernoulliSPRT:
    """One-sided simple-vs-simple sequential Bernoulli test.

    H0 is p=p0 and H1 is p=p1.  If p1>p0 the test detects an increase; if
    p1<p0 it detects a decrease.  The e-value is the H1/H0 likelihood ratio.
    """

    def __init__(
        self,
        p0: float,
        p1: float,
        alpha: float = 0.05,
        beta: float = 0.20,
        *,
        max_n: int | None = None,
        futility_budget: float | None = None,
    ) -> None:
        _validate_probability("p0", p0)
        _validate_probability("p1", p1)
        if p0 == p1:
            raise ValueError("p0 and p1 must differ")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be strictly between 0 and 1")
        if futility_budget is None:
            futility_budget = beta / 2.0
        if not 0.0 < futility_budget < beta:
            raise ValueError("futility_budget must be strictly between 0 and beta")

        self.p0 = float(p0)
        self.p1 = float(p1)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.futility_budget = float(futility_budget)
        self.direction = "greater" if self.p1 > self.p0 else "less"
        self.upper_log_boundary = math.log(1.0 / self.alpha)
        self.lower_log_boundary = math.log(self.futility_budget)
        self.max_n = int(max_n) if max_n is not None else exact_max_n(
            p0, p1, alpha, beta, futility_budget
        )
        if self.max_n < 1:
            raise ValueError("max_n must be at least 1")

        self.n = 0
        self.log_e_value = 0.0
        self.decision = Decision.CONTINUE

    @property
    def stopped(self) -> bool:
        return self.decision is not Decision.CONTINUE

    @property
    def e_value(self) -> float:
        return math.exp(self.log_e_value)

    def snapshot(self) -> Snapshot:
        return Snapshot(
            n=self.n,
            log_e_value=self.log_e_value,
            decision=self.decision,
            upper_log_boundary=self.upper_log_boundary,
            lower_log_boundary=self.lower_log_boundary,
            max_n=self.max_n,
        )

    def observation_log_likelihood_ratio(self, value: int) -> float:
        if value == 1:
            return math.log(self.p1 / self.p0)
        if value == 0:
            return math.log((1.0 - self.p1) / (1.0 - self.p0))
        raise ValueError("Bernoulli observations must be 0 or 1")

    def update(self, value: int) -> Snapshot:
        if self.stopped:
            return self.snapshot()
        if not isinstance(value, (int, float)) or not float(value).is_integer():
            raise ValueError("Bernoulli observations must be 0 or 1")
        integer_value = int(value)
        if integer_value not in (0, 1):
            raise ValueError("Bernoulli observations must be 0 or 1")

        self.n += 1
        self.log_e_value += self.observation_log_likelihood_ratio(integer_value)

        if self.log_e_value >= self.upper_log_boundary:
            self.decision = Decision.POSITIVE
        elif self.log_e_value <= self.lower_log_boundary:
            self.decision = Decision.NO_POSITIVE
        elif self.n >= self.max_n:
            self.decision = Decision.NO_POSITIVE
        return self.snapshot()

    def run(self, observations: Iterable[int]) -> Snapshot:
        for value in observations:
            self.update(value)
            if self.stopped:
                break
        return self.snapshot()


@dataclass(frozen=True)
class SuiteSnapshot:
    n: int
    decision: Decision
    positive_index: int | None
    stopped_indices: tuple[int, ...]
    max_n: int

    @property
    def stopped(self) -> bool:
        return self.decision is not Decision.CONTINUE


@dataclass(frozen=True)
class BernoulliTestConfig:
    name: str
    p0: float
    p1: float


class MultiBernoulliSPRT:
    """Family of simultaneous one-sided Bernoulli SPRTs.

    Per-test alpha is family_alpha / number_of_tests (Bonferroni).  The union
    bound controls family-wise false-positive probability even when metrics are
    correlated.
    """

    def __init__(
        self,
        configs: Sequence[BernoulliTestConfig | tuple[str, float, float]],
        family_alpha: float = 0.05,
        beta: float = 0.20,
        *,
        stop_on_first_positive: bool = True,
    ) -> None:
        if not configs:
            raise ValueError("at least one test configuration is required")
        if not 0.0 < family_alpha < 1.0:
            raise ValueError("family_alpha must be strictly between 0 and 1")
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be strictly between 0 and 1")

        normalized = []
        for config in configs:
            if isinstance(config, BernoulliTestConfig):
                normalized.append(config)
            else:
                name, p0, p1 = config
                normalized.append(BernoulliTestConfig(name, p0, p1))

        per_test_alpha = family_alpha / len(normalized)
        self.family_alpha = float(family_alpha)
        self.per_test_alpha = per_test_alpha
        self.beta = float(beta)
        self.stop_on_first_positive = stop_on_first_positive
        self.configs = tuple(normalized)
        self.tests = tuple(
            BernoulliSPRT(
                config.p0,
                config.p1,
                per_test_alpha,
                beta,
                max_n=exact_max_n(
                    config.p0,
                    config.p1,
                    per_test_alpha,
                    beta,
                    beta / 2.0,
                ),
            )
            for config in self.configs
        )
        self.max_n = max(test.max_n for test in self.tests)
        self.n = 0
        self.decision = Decision.CONTINUE
        self.positive_index: int | None = None

    @property
    def stopped(self) -> bool:
        return self.decision is not Decision.CONTINUE

    def _stopped_indices(self) -> tuple[int, ...]:
        return tuple(index for index, test in enumerate(self.tests) if test.stopped)

    def snapshot(self) -> SuiteSnapshot:
        return SuiteSnapshot(
            n=self.n,
            decision=self.decision,
            positive_index=self.positive_index,
            stopped_indices=self._stopped_indices(),
            max_n=self.max_n,
        )

    def update(self, observations: Sequence[int] | Mapping[str, int]) -> SuiteSnapshot:
        if self.stopped:
            return self.snapshot()
        if isinstance(observations, Mapping):
            ordered = [observations[config.name] for config in self.configs]
        else:
            ordered = list(observations)
        if len(ordered) != len(self.tests):
            raise ValueError("one observation is required for every metric")

        self.n += 1
        positive_indices = []
        new_positive_indices = []
        all_stopped = True
        for index, (test, value) in enumerate(zip(self.tests, ordered)):
            was_positive = test.decision is Decision.POSITIVE
            test.update(value)
            if test.decision is Decision.POSITIVE:
                positive_indices.append(index)
                if not was_positive:
                    new_positive_indices.append(index)
            if not test.stopped:
                all_stopped = False

        if positive_indices and self.positive_index is None:
            self.positive_index = positive_indices[0]

        if new_positive_indices and self.stop_on_first_positive:
            self.decision = Decision.POSITIVE
        elif all_stopped:
            self.decision = Decision.NO_POSITIVE

        if not self.stop_on_first_positive and all(test.stopped for test in self.tests):
            self.decision = (
                Decision.POSITIVE if positive_indices else Decision.NO_POSITIVE
            )
        return self.snapshot()

    def run(self, observations: Iterable[Sequence[int] | Mapping[str, int]]) -> SuiteSnapshot:
        for row in observations:
            self.update(row)
            if self.stopped:
                break
        return self.snapshot()
