"""Self-tests for tsalign (stdlib unittest). Run: python3 -m unittest -v"""

import unittest
from fractions import Fraction

from tsalign import (
    AlignmentStats, DriftCorrector, MonotonicFilter, RollbackPolicy,
    TimestampRollbackError, align_stream, error_stats, pairing_errors,
    seconds_to_ticks, to_seconds, to_ticks, DEFAULT_TICK,
)

US = DEFAULT_TICK  # 1 microsecond tick


def assert_non_decreasing(case, seq):
    for i in range(1, len(seq)):
        case.assertGreaterEqual(
            seq[i], seq[i - 1],
            "output went backwards at index %d: %s < %s"
            % (i, seq[i], seq[i - 1]))


def assert_strictly_increasing(case, seq):
    for i in range(1, len(seq)):
        case.assertGreater(
            seq[i], seq[i - 1],
            "output not strictly increasing at index %d: %s <= %s"
            % (i, seq[i], seq[i - 1]))


class TimebaseConversionTest(unittest.TestCase):
    def test_exact_conversion(self):
        self.assertEqual(to_seconds(3000, Fraction(1, 90000)),
                         Fraction(1, 30))
        self.assertEqual(to_seconds(1024, Fraction(1, 48000)),
                         Fraction(1024, 48000))

    def test_nondivisible_timebases_exact(self):
        # NTSC 30000/1001 fps vs audio 44100 Hz: no common divisor with the
        # microsecond tick; Fraction math must stay exact.
        tb_video = Fraction(1001, 30000)
        tb_audio = Fraction(1, 44100)
        v = to_seconds(29, tb_video)
        a = to_seconds(44100, tb_audio)
        self.assertEqual(v, Fraction(29 * 1001, 30000))
        self.assertEqual(a, Fraction(1, 1))
        # rounding to ticks is within half a tick of the exact value
        for pts, tb in [(29, tb_video), (10**6, tb_audio), (7, Fraction(1, 90000))]:
            exact = to_seconds(pts, tb)
            got = to_ticks(pts, tb) * US
            self.assertLessEqual(abs(got - exact), US / 2)

    def test_tick_grid_unified(self):
        # same instant expressed in two timebases lands on the same tick
        t1 = to_ticks(90000, Fraction(1, 90000))
        t2 = to_ticks(48000, Fraction(1, 48000))
        self.assertEqual(t1, t2)


class MonotonicFilterTest(unittest.TestCase):
    def test_clamp_policy_is_explicit_and_monotone(self):
        stats = AlignmentStats()
        f = MonotonicFilter(policy=RollbackPolicy.CLAMP, min_step=US,
                            stats=stats)
        out = [f.push(Fraction(v)) for v in (1, 2, 3, 2, 2, 4)]
        assert_strictly_increasing(self, out)
        self.assertEqual(stats.rollbacks, 2)
        # clamped values continue from last + min_step
        self.assertEqual(out[3], out[2] + US)
        self.assertEqual(out[4], out[3] + US)

    def test_drop_policy(self):
        f = MonotonicFilter(policy=RollbackPolicy.DROP)
        out = [f.push(Fraction(v)) for v in (1, 2, 1, 3)]
        self.assertEqual(out, [Fraction(1), Fraction(2), None, Fraction(3)])

    def test_raise_policy(self):
        f = MonotonicFilter(policy=RollbackPolicy.RAISE)
        f.push(Fraction(5))
        with self.assertRaises(TimestampRollbackError):
            f.push(Fraction(4))

    def test_jump_recorded_not_modified(self):
        stats = AlignmentStats()
        f = MonotonicFilter(jump_threshold=Fraction(1), stats=stats)
        f.push(Fraction(0))
        got = f.push(Fraction(10))
        self.assertEqual(got, Fraction(10))  # jump passed through untouched
        self.assertEqual(len(stats.jumps), 1)


class DriftCorrectorTest(unittest.TestCase):
    def test_skew_estimation(self):
        skew_true = 1.0001  # 100 ppm fast clock
        n = 5000
        expected = [Fraction(i, 25) for i in range(n)]
        observed = [Fraction(3) + e * Fraction(10001, 10000) for e in expected]
        c = DriftCorrector().fit(observed, expected)
        self.assertAlmostEqual(c.skew, skew_true, places=8)
        self.assertAlmostEqual(c.offset, 3.0, places=6)

    def test_robust_to_rollback_and_jump(self):
        n = 2000
        expected = [Fraction(i, 25) for i in range(n)]
        observed = [e * Fraction(100005, 100000) for e in expected]
        observed[500] = observed[100]          # rollback
        observed[1200] = observed[1200] + 30   # 30 s jump
        c = DriftCorrector().fit(observed, expected)
        self.assertAlmostEqual(c.skew, 1.00005, places=7)
        # the rollback sample and the whole post-jump segment are flagged
        self.assertGreaterEqual(c.outliers, 2)
        # correction of a clean (pre-jump) sample lands on the true grid
        self.assertAlmostEqual(float(c.correct(observed[100])),
                               float(expected[100]), places=6)

    def test_single_sample_identity(self):
        c = DriftCorrector().fit([Fraction(42)], [Fraction(0)])
        self.assertEqual(c.skew, 1.0)
        self.assertEqual(c.correct(Fraction(42)), Fraction(0))

    def test_correction_residual(self):
        n = 1000
        expected = [Fraction(i, 25) for i in range(n)]
        observed = [Fraction(1, 10) + e * Fraction(10002, 10000)
                    for e in expected]
        c = DriftCorrector().fit(observed, expected)
        residuals = [abs(float(c.correct(o)) - float(e))
                     for o, e in zip(observed, expected)]
        self.assertLess(max(residuals), 1e-3)


class AlignStreamTest(unittest.TestCase):
    def test_monotonic_output_with_rollbacks(self):
        tb = Fraction(1, 90000)
        pts = [i * 3600 for i in range(200)]      # 25 fps CFR
        pts[100] = pts[90]                        # rollback
        pts[150] = pts[149]                       # duplicate
        res = align_stream(pts, tb, nominal_interval=Fraction(1, 25))
        assert_strictly_increasing(self, res.ticks)
        self.assertEqual(res.stats.rollbacks, 2)

    def test_zero_duration_stream(self):
        # all frames stamped identically -> zero-duration stream
        res = align_stream([100, 100, 100, 100], Fraction(1, 90000),
                           nominal_interval=Fraction(1, 25))
        self.assertEqual(len(res.ticks), 4)
        assert_non_decreasing(self, res.ticks)
        assert_strictly_increasing(self, res.ticks)  # min_step keeps progress

    def test_single_frame_stream(self):
        res = align_stream([12345], Fraction(1, 48000),
                           nominal_interval=Fraction(1024, 48000))
        self.assertEqual(len(res.ticks), 1)
        self.assertEqual(res.ticks[0],
                         seconds_to_ticks(to_seconds(12345, Fraction(1, 48000))))
        self.assertEqual(res.stats.skew, 1.0)  # no drift fit possible

    def test_empty_stream(self):
        res = align_stream([], Fraction(1, 90000),
                           nominal_interval=Fraction(1, 25))
        self.assertEqual(len(res.ticks), 0)

    def test_extreme_long_duration(self):
        # pts near 2**62 in a 1/90000 timebase ~= 5.1e13 seconds; float64
        # cannot hold microsecond precision there, Fraction must.
        tb = Fraction(1, 90000)
        base = 2**62
        n = 9000  # 360 s of 25 fps at the far end of the pts range
        pts = [base + i * 3600 for i in range(n)]
        res = align_stream(pts, tb, nominal_interval=Fraction(1, 25))
        assert_strictly_increasing(self, res.ticks)
        # exact microsecond placement preserved at the extreme
        for i in (0, n // 2, n - 1):
            exact = to_seconds(pts[i], tb)
            self.assertLessEqual(abs(res.seconds[i] - exact), US)
        # spacing between consecutive frames is exactly 40 ms in ticks
        step = to_ticks(3600, tb)
        self.assertEqual(res.ticks[-1] - res.ticks[-2], step)

    def test_vfr_stream(self):
        # variable frame rate: 24/30/60 fps mix, timestamps follow the
        # declared per-frame durations with a 40 ppm clock skew
        durations = [Fraction(1, 24), Fraction(1, 30), Fraction(1, 60)] * 400
        tb = Fraction(1, 90000)
        t = Fraction(0)
        pts = []
        for d in durations:
            pts.append(int(t * Fraction(100004, 100000) / tb))
            t += d
        res = align_stream(pts, tb, nominal_durations=durations)
        assert_strictly_increasing(self, res.ticks)
        self.assertAlmostEqual(res.stats.skew, 1.00004, delta=2e-5)
        # residual vs nominal grid is small after correction
        acc = Fraction(0)
        worst = Fraction(0)
        for s, d in zip(res.seconds, durations):
            worst = max(worst, abs(s - acc))
            acc += d
        self.assertLess(float(worst), 0.002)  # < 2 ms over 1200 VFR frames

    def test_two_stream_end_to_end(self):
        # video 25 fps @1/90000 with +50 ppm skew, audio 1024-sample frames
        # @1/48000 with -30 ppm skew; both start at the same true instant.
        n_v, n_a = 4500, 8438  # 180 s
        tb_v, tb_a = Fraction(1, 90000), Fraction(1, 48000)
        true_v = [Fraction(i, 25) for i in range(n_v)]
        true_a = [Fraction(i * 1024, 48000) for i in range(n_a)]
        pts_v = [int(true_v[i] * Fraction(100005, 100000) / tb_v)
                 for i in range(n_v)]
        pts_a = [int(true_a[i] * Fraction(99997, 100000) / tb_a)
                 for i in range(n_a)]
        rv = align_stream(pts_v, tb_v, nominal_interval=Fraction(1, 25))
        ra = align_stream(pts_a, tb_a, nominal_interval=Fraction(1024, 48000))
        assert_strictly_increasing(self, rv.ticks)
        assert_strictly_increasing(self, ra.ticks)
        self.assertAlmostEqual(rv.stats.skew, 1.00005, delta=2e-5)
        self.assertAlmostEqual(ra.stats.skew, 0.99997, delta=2e-5)

        # Alignment error vs the simulated ground truth. (Note: nearest-frame
        # pairing alone cannot measure drift -- a slow stream just re-pairs to
        # a neighbour -- so residual against the true grid is the metric.)
        raw_v = [to_seconds(p, tb_v) for p in pts_v]
        raw_a = [to_seconds(p, tb_a) for p in pts_a]
        before_v = [float(raw_v[i] - true_v[i]) for i in range(n_v)]
        before_a = [float(raw_a[i] - true_a[i]) for i in range(n_a)]
        after_v = [float(rv.seconds[i] - true_v[i]) for i in range(n_v)]
        after_a = [float(ra.seconds[i] - true_a[i]) for i in range(n_a)]
        rms = lambda xs: (sum(x * x for x in xs) / len(xs)) ** 0.5
        # uncorrected drift accumulates to milliseconds over 180 s
        self.assertGreater(rms(before_v), 0.005)   # ~9 ms for +50 ppm
        self.assertGreater(rms(before_a), 0.003)   # ~5 ms for -30 ppm
        # corrected residual is at the frame-quantisation level (<= ~25 us)
        self.assertLess(rms(after_v), 25e-6)
        self.assertLess(rms(after_a), 25e-6)

        # cross-stream check: both streams now share one timescale, and frame
        # pairing error stays within half the shorter frame interval
        paired = error_stats(pairing_errors(rv.ticks, ra.ticks))
        self.assertLess(paired["rms"], 0.011)      # half of 21.3 ms audio frame


if __name__ == "__main__":
    unittest.main()
