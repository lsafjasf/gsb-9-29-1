from __future__ import annotations

import unittest
from fractions import Fraction
from random import Random

from timestamp_sync import (
    ClockConfig,
    ClockEvent,
    DriftModel,
    MonotonicClock,
    RollbackPolicy,
    align_to_reference,
    align_two_streams,
    assert_monotonic,
    error_statistics,
    fit_drift_model,
    merge_timelines,
    to_seconds,
)


class TimestampNormalizationTests(unittest.TestCase):
    def test_exact_non_divisible_time_base(self) -> None:
        frames = MonotonicClock((1, 3)).normalize_ticks([0, 1, 2, 5])
        self.assertEqual(
            [frame.time for frame in frames],
            [Fraction(0), Fraction(1, 3), Fraction(2, 3), Fraction(5, 3)],
        )

    def test_single_frame_and_zero_duration(self) -> None:
        single = MonotonicClock((1, 90000)).normalize_ticks([123])
        self.assertEqual(single[0].time, Fraction(41, 30000))
        self.assertEqual(single[0].event, ClockEvent.FIRST)
        self.assertEqual(MonotonicClock((1, 90000)).normalize_ticks([]), [])

    def test_duplicate_timestamps_are_non_decreasing(self) -> None:
        frames = MonotonicClock((1, 1000)).normalize_ticks([0, 10, 10, 20])
        times = [frame.time for frame in frames]
        self.assertEqual(times, [Fraction(0), Fraction(1, 100), Fraction(1, 100), Fraction(1, 50)])
        self.assertEqual(frames[2].event, ClockEvent.DUPLICATE)
        assert_monotonic(times)
        with self.assertRaises(ValueError):
            assert_monotonic(times, strict=True)

    def test_known_modulus_wraps_are_unwrapped(self) -> None:
        frames = MonotonicClock((1, 3), wrap_period_ticks=10).normalize_ticks(
            [0, 1, 3, 9, 1, 4, 10, 2]
        )
        self.assertEqual(
            [frame.time for frame in frames],
            [
                Fraction(0),
                Fraction(1, 3),
                Fraction(1),
                Fraction(3),
                Fraction(11, 3),
                Fraction(14, 3),
                Fraction(20, 3),
                Fraction(22, 3),
            ],
        )
        self.assertEqual(frames[4].event, ClockEvent.WRAP)
        self.assertEqual(frames[7].event, ClockEvent.WRAP)
        assert_monotonic(frame.time for frame in frames)

    def test_wrap_boundary_from_modulus_minus_one_to_zero(self) -> None:
        frames = MonotonicClock((1, 3), wrap_period_ticks=10).normalize_ticks(
            [9, 0, 1]
        )
        self.assertEqual([frame.time for frame in frames], [Fraction(3), Fraction(10, 3), Fraction(11, 3)])
        self.assertEqual(frames[1].event, ClockEvent.WRAP)
        assert_monotonic(frame.time for frame in frames)

    def test_unknown_rollback_clamp_strategy(self) -> None:
        frames = MonotonicClock((1, 1000)).normalize_ticks(
            [0, 100, 150, 10, 20, 1000]
        )
        self.assertEqual(
            [frame.time for frame in frames],
            [
                Fraction(0),
                Fraction(1, 10),
                Fraction(3, 20),
                Fraction(3, 20),
                Fraction(4, 25),
                Fraction(57, 50),
            ],
        )
        self.assertEqual(frames[3].event, ClockEvent.CLAMPED)
        assert_monotonic(frame.time for frame in frames)

    def test_unknown_rollback_gap_strategy(self) -> None:
        frames = MonotonicClock(
            (1, 1000),
            rollback_policy=RollbackPolicy.GAP,
            rollback_gap=Fraction(1, 1000),
        ).normalize_ticks([0, 100, 150, 10, 20, 1000])
        self.assertEqual(
            [frame.time for frame in frames],
            [
                Fraction(0),
                Fraction(1, 10),
                Fraction(3, 20),
                Fraction(151, 1000),
                Fraction(161, 1000),
                Fraction(1141, 1000),
            ],
        )
        self.assertEqual(frames[3].event, ClockEvent.GAP_FILLED)
        assert_monotonic(frame.time for frame in frames)

    def test_raise_strategy_rejects_unknown_rollback(self) -> None:
        clock = MonotonicClock((1, 1000), rollback_policy=RollbackPolicy.RAISE)
        clock.process(100)
        with self.assertRaises(ValueError):
            clock.process(99)

    def test_variable_frame_intervals_are_preserved(self) -> None:
        frames = MonotonicClock((1, 1000)).normalize_ticks([0, 7, 100, 101, 5000])
        self.assertEqual(
            [frame.time - frames[index - 1].time for index, frame in enumerate(frames) if index],
            [Fraction(7, 1000), Fraction(93, 1000), Fraction(1, 1000), Fraction(4899, 1000)],
        )


class DriftAndAlignmentTests(unittest.TestCase):
    def test_exact_linear_drift_correction(self) -> None:
        calibration = [
            (Fraction(0), Fraction(1, 1000)),
            (Fraction(10), Fraction("10.011")),
            (Fraction(20), Fraction("20.021")),
        ]
        model = fit_drift_model(calibration)
        self.assertEqual(model.slope, Fraction(1001, 1000))
        self.assertEqual(model.intercept, Fraction(1, 1000))
        self.assertEqual(model.corrected(1000), Fraction(1001001, 1000))

    def test_robust_fit_tolerates_discontinuity_outlier(self) -> None:
        calibration = [(Fraction(index), Fraction(index)) for index in range(20)]
        calibration[19] = (Fraction(19), Fraction(19) + Fraction(1, 2))
        model = fit_drift_model(calibration)
        self.assertEqual(model, DriftModel(Fraction(1), Fraction(0)))

    def test_corrected_timeline_is_monotonic(self) -> None:
        frames = MonotonicClock((1, 1000)).normalize_ticks([0, 10, 10, 20, 100])
        calibration = [
            (Fraction(0), Fraction(1, 100)),
            (Fraction(1, 10), Fraction(11, 100)),
        ]
        corrected = align_to_reference(frames, calibration)
        self.assertEqual(corrected[1], corrected[2])
        assert_monotonic(corrected)

    def test_two_stream_non_divisible_time_bases(self) -> None:
        a_ticks = [0, 7, 14, 21]
        b_ticks = [0, 14, 28, 42]
        a_config = ClockConfig((1, 7))
        b_config = ClockConfig((1, 13))
        sync_points = [
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(14, 13)),
            (Fraction(2), Fraction(28, 13)),
            (Fraction(3), Fraction(42, 13)),
        ]
        result = align_two_streams(
            a_ticks,
            b_ticks,
            first=a_config,
            second=b_config,
            sync_points=sync_points,
            method="ols",
        )
        self.assertEqual(result.second.corrected, [Fraction(0), Fraction(1), Fraction(2), Fraction(3)])
        assert_monotonic(result.first.corrected)
        assert_monotonic(result.second.corrected)

    def test_extreme_duration_uses_exact_arithmetic(self) -> None:
        years = 1000
        seconds = years * 365 * 24 * 3600
        time_base = Fraction(1, 90000)
        ticks = [0, seconds * 90000]
        frames = MonotonicClock(time_base).normalize_ticks(ticks)
        self.assertEqual(frames[-1].time, Fraction(seconds))
        calibration = [
            (Fraction(0), Fraction(0)),
            (Fraction(seconds), Fraction(seconds) + 1),
        ]
        corrected = align_to_reference(frames, calibration, method="ols")
        self.assertEqual(corrected[-1] - corrected[0], Fraction(seconds) + 1)
        assert_monotonic(corrected)

    def test_merged_timeline_is_monotonic(self) -> None:
        merged = merge_timelines([1, 3], [1, 2, 4])
        self.assertEqual([time for time, _label in merged], [1, 1, 2, 3, 4])

    def test_error_statistics_use_integer_nanoseconds(self) -> None:
        stats = error_statistics([Fraction(0), Fraction(-1, 1_000_000_000)])
        self.assertEqual(stats["mean_abs_ns"], 0)
        self.assertEqual(stats["max_abs_ns"], 1)

    def test_random_stress_always_non_decreasing(self) -> None:
        random = Random(20260930)
        for wrap_period in (None, 101):
            clock = MonotonicClock(
                (1, 97),
                wrap_period_ticks=wrap_period,
                rollback_policy=RollbackPolicy.CLAMP,
            )
            frames = [clock.process(random.randrange(0, 101)) for _ in range(1000)]
            assert_monotonic(frame.time for frame in frames)


class ValidationTests(unittest.TestCase):
    def test_invalid_time_base(self) -> None:
        with self.assertRaises(ValueError):
            MonotonicClock((0, 1))

    def test_to_seconds_helper(self) -> None:
        self.assertEqual(to_seconds(3, (2, 7)), Fraction(6, 7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
