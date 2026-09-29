from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Iterable, List, Optional, Sequence, Tuple, Union


TimeBaseLike = Union[Fraction, int, float, str, Tuple[int, int]]
TimeValue = Union[Fraction, int, float, str]


class RollbackPolicy(str, Enum):
    CLAMP = "clamp"
    GAP = "gap"
    RAISE = "raise"


class ClockEvent(str, Enum):
    FIRST = "first"
    FORWARD = "forward"
    DUPLICATE = "duplicate"
    WRAP = "wrap"
    CLAMPED = "clamped"
    GAP_FILLED = "gap_filled"


@dataclass(frozen=True)
class NormalizedFrame:
    index: int
    ticks: int
    time: Fraction
    event: ClockEvent
    segment: int
    previous_time: Optional[Fraction]


@dataclass(frozen=True)
class DriftModel:
    slope: Fraction
    intercept: Fraction

    def corrected(self, observed: TimeValue) -> Fraction:
        return self.slope * _as_time(observed) + self.intercept


@dataclass(frozen=True)
class StreamResult:
    name: str
    normalized: List[NormalizedFrame]
    corrected: List[Fraction]
    model: DriftModel


@dataclass(frozen=True)
class TwoStreamAlignment:
    first: StreamResult
    second: StreamResult
    model: DriftModel


@dataclass(frozen=True)
class ClockConfig:
    time_base: TimeBaseLike
    wrap_period_ticks: Optional[int] = None
    rollback_policy: RollbackPolicy = RollbackPolicy.CLAMP
    rollback_gap: TimeValue = 0


def _as_time_base(value: TimeBaseLike) -> Fraction:
    if isinstance(value, tuple):
        numerator, denominator = value
        return Fraction(numerator, denominator)
    return Fraction(value)


def _as_time(value: TimeValue) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def to_seconds(ticks: int, time_base: TimeBaseLike) -> Fraction:
    return Fraction(ticks) * _as_time_base(time_base)


def to_nanoseconds(value: TimeValue) -> int:
    return int(_as_time(value) * 1_000_000_000)


def assert_monotonic(values: Iterable[TimeValue], strict: bool = False) -> None:
    previous = None
    for position, value in enumerate(values):
        current = _as_time(value)
        if previous is not None:
            if strict and current <= previous:
                raise ValueError(f"timestamp at {position} is not strictly increasing")
            if current < previous:
                raise ValueError(f"timestamp at {position} moves backward")
        previous = current


class MonotonicClock:
    """Normalize one tick timeline into non-decreasing seconds on a common clock."""

    def __init__(
        self,
        time_base: TimeBaseLike,
        *,
        wrap_period_ticks: Optional[int] = None,
        rollback_policy: Union[RollbackPolicy, str] = RollbackPolicy.CLAMP,
        rollback_gap: TimeValue = 0,
    ) -> None:
        self.time_base = _as_time_base(time_base)
        if self.time_base <= 0:
            raise ValueError("time_base must be positive")
        self.wrap_period_ticks = wrap_period_ticks
        if wrap_period_ticks is not None and wrap_period_ticks <= 0:
            raise ValueError("wrap_period_ticks must be positive")
        self.rollback_policy = RollbackPolicy(rollback_policy)
        self.rollback_gap = _as_time(rollback_gap)
        if self.rollback_gap < 0:
            raise ValueError("rollback_gap must be non-negative")
        self.reset()

    def reset(self) -> None:
        self._last_ticks: Optional[int] = None
        self._last_time: Optional[Fraction] = None
        self._segment = 0
        self._segment_wrap_count = 0
        self._segment_time_offset = Fraction(0)
        self._segment_tick_offset = 0
        self._frame_count = 0

    def normalize_ticks(self, ticks: Iterable[int]) -> List[NormalizedFrame]:
        return [self.process(tick) for tick in ticks]

    def _apply_rollback(self, ticks: int, raw_time: Fraction) -> ClockEvent:
        if self.rollback_policy == RollbackPolicy.RAISE:
            raise ValueError(f"timestamp rolled back from {self._last_ticks} to {ticks}")
        if self.rollback_policy == RollbackPolicy.GAP:
            event = ClockEvent.GAP_FILLED
            target_time = self._last_time + self.rollback_gap
        else:
            event = ClockEvent.CLAMPED
            target_time = self._last_time

        self._segment_tick_offset = self._last_ticks - ticks
        self._segment_wrap_count = 0
        self._segment_time_offset = target_time - to_seconds(
            ticks + self._segment_tick_offset, self.time_base
        )
        self._segment += 1
        self._last_time = target_time
        return event

    def process(self, ticks: int) -> NormalizedFrame:
        if not isinstance(ticks, int):
            raise TypeError("ticks must be an integer")

        raw_time = to_seconds(ticks, self.time_base)
        event: ClockEvent

        if self._last_ticks is None:
            event = ClockEvent.FIRST
            output_time = raw_time + self._segment_time_offset
            previous_time = None
            self._last_ticks = ticks
            self._last_time = output_time
            frame = NormalizedFrame(
                self._frame_count, ticks, output_time, event, self._segment, previous_time
            )
            self._frame_count += 1
            return frame

        frame_index = self._frame_count
        previous_time = self._last_time
        delta_ticks = ticks - self._last_ticks
        wrap_seconds = (
            to_seconds(self.wrap_period_ticks, self.time_base)
            if self.wrap_period_ticks is not None
            else Fraction(0)
        )
        base_time = (
            to_seconds(ticks + self._segment_tick_offset, self.time_base)
            + self._segment_time_offset
        )
        wrapped_time = base_time + (self._segment_wrap_count + 1) * wrap_seconds

        if delta_ticks == 0:
            event = ClockEvent.DUPLICATE
            output_time = base_time + self._segment_wrap_count * wrap_seconds
        elif delta_ticks > 0:
            event = ClockEvent.FORWARD
            output_time = base_time + self._segment_wrap_count * wrap_seconds
        elif self.wrap_period_ticks is not None:
            if wrapped_time > self._last_time:
                event = ClockEvent.WRAP
                output_time = wrapped_time
                self._segment_wrap_count += 1
            else:
                event = self._apply_rollback(ticks, raw_time)
                output_time = self._last_time
        else:
            event = self._apply_rollback(ticks, raw_time)
            output_time = self._last_time

        frame = NormalizedFrame(
            frame_index,
            ticks,
            output_time,
            event,
            self._segment,
            previous_time,
        )
        self._last_ticks = ticks
        self._last_time = output_time
        self._frame_count += 1
        return frame


def _median(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise ValueError("cannot take median of empty values")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def fit_drift_model(
    samples: Iterable[Tuple[TimeValue, TimeValue]],
    *,
    method: str = "theil-sen",
) -> DriftModel:
    """Fit reference = slope * observed + intercept.

    "ols" is the exact least-squares model. "theil-sen" uses the median pair
    slope and median intercept; it tolerates the discontinuities/outliers that
    occur with real media clocks.
    """

    points = [(_as_time(x), _as_time(y)) for x, y in samples]
    if not points:
        raise ValueError("at least one calibration sample is required")

    if method in ("theil-sen", "theil_sen", "median"):
        if len(points) == 1:
            return DriftModel(Fraction(1), points[0][1] - points[0][0])
        slopes = [
            (right[1] - left[1]) / (right[0] - left[0])
            for pos, left in enumerate(points)
            for right in points[pos + 1 :]
            if right[0] != left[0]
        ]
        slope = _median(slopes) if slopes else Fraction(1)
        intercept = _median([reference - slope * observed for observed, reference in points])
        return DriftModel(slope, intercept)

    if method == "ols":
        count = len(points)
        sum_x = sum((point[0] for point in points), Fraction(0))
        sum_y = sum((point[1] for point in points), Fraction(0))
        mean_x = sum_x / count
        mean_y = sum_y / count
        denominator = sum((point[0] - mean_x) ** 2 for point in points)
        if denominator == 0:
            return DriftModel(Fraction(1), mean_y - mean_x)
        numerator = sum(
            (point[0] - mean_x) * (point[1] - mean_y) for point in points
        )
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
        return DriftModel(slope, intercept)

    raise ValueError(f"unknown method: {method}")


def percentile_fraction(values: Sequence[Fraction], percentile: float) -> Fraction:
    if not values:
        raise ValueError("cannot take percentile of empty values")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = Fraction(str(position - lower))
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def error_statistics(errors: Iterable[TimeValue]) -> dict:
    signed = [_as_time(value) for value in errors]
    if not signed:
        raise ValueError("at least one error is required")
    absolute = [abs(value) for value in signed]
    count = len(signed)
    mean = sum(signed, Fraction(0)) / count
    mean_abs = sum(absolute, Fraction(0)) / count
    return {
        "count": count,
        "mean_ns": to_nanoseconds(mean),
        "mean_abs_ns": to_nanoseconds(mean_abs),
        "min_abs_ns": to_nanoseconds(min(absolute)),
        "p50_abs_ns": to_nanoseconds(percentile_fraction(absolute, 50)),
        "p90_abs_ns": to_nanoseconds(percentile_fraction(absolute, 90)),
        "p95_abs_ns": to_nanoseconds(percentile_fraction(absolute, 95)),
        "p99_abs_ns": to_nanoseconds(percentile_fraction(absolute, 99)),
        "max_abs_ns": to_nanoseconds(max(absolute)),
    }


def align_to_reference(
    frames: Sequence[NormalizedFrame],
    calibration: Sequence[Tuple[TimeValue, TimeValue]],
    *,
    method: str = "theil-sen",
) -> List[Fraction]:
    model = fit_drift_model(calibration, method=method)
    corrected = [model.corrected(frame.time) for frame in frames]
    assert_monotonic(corrected)
    return corrected


def align_two_streams(
    first_ticks: Sequence[int],
    second_ticks: Sequence[int],
    *,
    first: ClockConfig,
    second: ClockConfig,
    sync_points: Sequence[Tuple[TimeValue, TimeValue]],
    method: str = "theil-sen",
) -> TwoStreamAlignment:
    """Map the second stream onto the first using matched synchronization points.

    Each sync point is ``(observed_first_seconds, observed_second_seconds)``.
    The first stream is treated as the master clock. If both devices are to be
    corrected to an external clock, use align_to_reference separately.
    """

    first_clock = MonotonicClock(
        first.time_base,
        wrap_period_ticks=first.wrap_period_ticks,
        rollback_policy=first.rollback_policy,
        rollback_gap=first.rollback_gap,
    )
    second_clock = MonotonicClock(
        second.time_base,
        wrap_period_ticks=second.wrap_period_ticks,
        rollback_policy=second.rollback_policy,
        rollback_gap=second.rollback_gap,
    )
    first_normalized = first_clock.normalize_ticks(first_ticks)
    second_normalized = second_clock.normalize_ticks(second_ticks)
    first_corrected = [frame.time for frame in first_normalized]
    model = fit_drift_model(
        [(second_time, first_time) for first_time, second_time in sync_points],
        method=method,
    )
    second_corrected = [model.corrected(frame.time) for frame in second_normalized]
    assert_monotonic(first_corrected)
    assert_monotonic(second_corrected)
    first_result = StreamResult(
        "first", first_normalized, first_corrected, DriftModel(Fraction(1), Fraction(0))
    )
    second_result = StreamResult(
        "second", second_normalized, second_corrected, model
    )
    return TwoStreamAlignment(first_result, second_result, model)


def merge_timelines(
    first: Sequence[TimeValue],
    second: Sequence[TimeValue],
    *,
    first_label: str = "first",
    second_label: str = "second",
) -> List[Tuple[Fraction, str]]:
    events = [(_as_time(value), first_label) for value in first]
    events.extend((_as_time(value), second_label) for value in second)
    events.sort(key=lambda item: (item[0], item[1]))
    assert_monotonic((item[0] for item in events))
    return events
