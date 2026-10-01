"""tsalign -- multi-stream timestamp alignment library (Python 3, stdlib only).

Problem: two media streams each carry their own timebase; over long durations
their clocks drift apart (audio/video desync), and raw timestamps may roll
back or jump.

Pipeline per stream:
  1. Timebase conversion: pts * timebase -> exact Fraction seconds, then
     rounded to a unified integer tick grid (default 1 us).
  2. Drift (clock skew) estimation against the nominal timing grid:
     a robust median-of-pair-slopes fit (Theil-Sen style, tolerates
     contiguous contaminated segments such as everything after a jump),
     refined by least squares over the inlier set, then affine correction.
  3. Monotonic filter with an explicit rollback policy.

Rollback policy (explicit, chosen strategy):
  CLAMP (default): a timestamp that would move the output backwards is
      replaced by last_output + min_step; the event is counted in
      stats.rollbacks. Forward progress of the stream is preserved and
      output never goes backwards.
  DROP:  the offending frame is dropped from the output entirely.
  RAISE: raise TimestampRollbackError and let the caller decide.

Forward jumps are NOT modified (they may be real gaps in the media); jumps
larger than `jump_threshold` are recorded in stats.jumps.

All internal math on timestamps uses fractions.Fraction, so non-divisible
timebases (e.g. 1001/30000 vs 1/48000) and extreme durations (pts near 2**62)
are handled exactly, with no float precision loss.
"""

import random
from fractions import Fraction
from enum import Enum

DEFAULT_TICK = Fraction(1, 1_000_000)  # unified timescale: 1 microsecond


class RollbackPolicy(Enum):
    CLAMP = "clamp"
    DROP = "drop"
    RAISE = "raise"


class TimestampRollbackError(Exception):
    """Raised under RollbackPolicy.RAISE when a timestamp moves backwards."""


class AlignmentStats:
    def __init__(self):
        self.rollbacks = 0          # number of rollback events handled
        self.jumps = []             # list of (prev_seconds, new_seconds)
        self.skew = 1.0             # estimated clock skew (observed/nominal)
        self.offset = 0.0           # estimated clock offset in seconds
        self.drift_corrected = False
        self.fit_outliers = 0       # samples rejected during robust fit

    def __repr__(self):
        return ("AlignmentStats(rollbacks=%d, jumps=%d, skew=%.9f, "
                "offset=%.6f, drift_corrected=%s, fit_outliers=%d)" % (
                    self.rollbacks, len(self.jumps), self.skew, self.offset,
                    self.drift_corrected, self.fit_outliers))


def to_seconds(pts, timebase):
    """Exact conversion of pts in `timebase` to seconds (Fraction)."""
    return Fraction(pts) * Fraction(timebase)


def to_ticks(pts, timebase, tick=DEFAULT_TICK):
    """Convert pts in `timebase` to integer ticks on the unified scale."""
    scaled = Fraction(pts) * Fraction(timebase) / Fraction(tick)
    return int(scaled + Fraction(1, 2))  # round half up


def seconds_to_ticks(seconds, tick=DEFAULT_TICK):
    scaled = Fraction(seconds) / Fraction(tick)
    return int(scaled + Fraction(1, 2))


class MonotonicFilter:
    """Enforces the explicit rollback policy; guarantees non-decreasing output
    (strictly increasing when min_step > 0)."""

    def __init__(self, policy=RollbackPolicy.CLAMP, min_step=Fraction(0),
                 jump_threshold=None, stats=None):
        self.policy = policy
        self.min_step = Fraction(min_step)
        self.jump_threshold = (Fraction(jump_threshold)
                               if jump_threshold is not None else None)
        self.stats = stats if stats is not None else AlignmentStats()
        self._last = None

    def push(self, ts):
        """Feed a timestamp (Fraction seconds). Returns the emitted timestamp,
        or None when the frame is dropped under RollbackPolicy.DROP."""
        ts = Fraction(ts)
        if self._last is None:
            self._last = ts
            return ts
        floor = self._last + self.min_step
        if ts < floor:
            self.stats.rollbacks += 1
            if self.policy is RollbackPolicy.CLAMP:
                self._last = floor
                return floor
            if self.policy is RollbackPolicy.DROP:
                return None
            raise TimestampRollbackError(
                "timestamp rollback: %s < %s" % (ts, floor))
        if (self.jump_threshold is not None
                and ts - self._last > self.jump_threshold):
            self.stats.jumps.append((self._last, ts))
        self._last = ts
        return ts


def _median(sorted_vals):
    n = len(sorted_vals)
    mid = n // 2
    if n % 2:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2


class DriftCorrector:
    """Estimates clock skew/offset of observed timestamps against a nominal
    reference grid, using least squares with one MAD-based outlier rejection
    pass (robust to rollbacks and jumps in the raw data).

    Model:  observed ~= offset + skew * reference
    Correct: reference_estimate = (observed - offset) / skew
    """

    def __init__(self):
        self.offset = 0.0
        self.skew = 1.0
        self.outliers = 0
        self._fitted = False

    def fit(self, observed, expected):
        """observed/expected: parallel sequences of seconds (Fraction or float).
        Fewer than 2 usable points -> no correction (skew=1, offset anchored
        at the first sample so identity mapping is preserved).

        Robust estimation: median slope of many randomly sampled point pairs
        (Theil-Sen style), median intercept. Tolerates contiguous contaminated
        segments below ~50% -- e.g. timestamps after a forward jump -- which a
        single least-squares + MAD pass cannot handle.
        """
        n = min(len(observed), len(expected))
        if n == 0:
            self.offset, self.skew = 0.0, 1.0
            self._fitted = True
            return self
        if n == 1:
            self.offset = float(observed[0]) - float(expected[0])
            self.skew = 1.0
            self._fitted = True
            return self

        xs = [float(expected[i]) for i in range(n)]
        ys = [float(observed[i]) for i in range(n)]

        rng = random.Random(0x5A116)
        max_pairs = n * (n - 1) // 2
        k = min(2001, max_pairs)
        slopes = []
        tries = 0
        while len(slopes) < k and tries < k * 10:
            tries += 1
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j or xs[i] == xs[j]:
                continue
            slopes.append((ys[j] - ys[i]) / (xs[j] - xs[i]))
        if not slopes:  # zero-duration: no time progress observable
            self.offset, self.skew = ys[0] - xs[0], 1.0
            self._fitted = True
            return self
        slope = _median(sorted(slopes))
        intercept = _median(sorted(ys[i] - slope * xs[i] for i in range(n)))

        # refine on inliers with least squares: the median slope is robust
        # but quantisation/jitter-limited; LS over the inlier set recovers
        # full precision so residuals stay at the jitter floor even over
        # multi-hour durations.
        residuals = [ys[i] - (intercept + slope * xs[i]) for i in range(n)]
        med = _median(sorted(residuals))
        mad = _median(sorted(abs(r - med) for r in residuals))
        if mad > 0:
            thresh = 5.0 * mad
            inliers = [i for i in range(n)
                       if abs(residuals[i] - med) <= thresh]
            if len(inliers) >= max(2, n // 4):
                slope, intercept = self._least_squares(
                    [xs[i] for i in inliers], [ys[i] for i in inliers])
                residuals = [ys[i] - (intercept + slope * xs[i])
                             for i in range(n)]
                med = _median(sorted(residuals))
                mad = _median(sorted(abs(r - med) for r in residuals))
                if mad > 0:
                    self.outliers = sum(
                        1 for r in residuals if abs(r - med) > 5.0 * mad)

        if abs(slope) < 1e-12:
            slope, intercept = 1.0, ys[0] - xs[0]
        self.skew = slope
        self.offset = intercept
        self._fitted = True
        return self

    @staticmethod
    def _least_squares(xs, ys):
        n = len(xs)
        xm = sum(xs) / n
        ym = sum(ys) / n
        sxx = sum((x - xm) ** 2 for x in xs)
        if sxx == 0.0:
            return 1.0, ys[0] - xs[0]
        sxy = sum((xs[i] - xm) * (ys[i] - ym) for i in range(n))
        slope = sxy / sxx
        return slope, ym - slope * xm

    def correct(self, ts):
        """Map an observed timestamp (Fraction seconds) onto the reference
        grid. Affine with positive skew, so it preserves ordering."""
        if not self._fitted:
            raise RuntimeError("DriftCorrector.fit() must be called first")
        return (Fraction(ts) - Fraction(self.offset)) / Fraction(self.skew)


class AlignmentResult:
    def __init__(self, ticks, seconds, stats, tick):
        self.ticks = ticks          # list[int] on the unified tick grid
        self.seconds = seconds      # list[Fraction] corrected seconds
        self.stats = stats
        self.tick = tick

    def __len__(self):
        return len(self.ticks)


def align_stream(pts_list, timebase, nominal_interval=None,
                 nominal_durations=None, tick=DEFAULT_TICK,
                 policy=RollbackPolicy.CLAMP, min_step=None,
                 jump_threshold=None, correct_drift=True):
    """Align one stream onto the unified tick grid.

    pts_list:          raw presentation timestamps (ints) in `timebase`.
    nominal_interval:  declared per-frame interval in seconds (Fraction or
                       float), e.g. Fraction(1, 25) or Fraction(1024, 48000).
                       Required for drift estimation of CFR streams.
    nominal_durations: per-frame nominal durations in seconds for VFR streams
                       (len == len(pts_list)); overrides nominal_interval.
    min_step:          minimum forward step enforced by the monotonic filter
                       (default: one tick -> strictly increasing output).
    jump_threshold:    forward jumps larger than this (seconds) are recorded.
    correct_drift:     set False to skip drift estimation (skew stays 1).

    Returns AlignmentResult with monotone (strictly increasing) ticks.
    """
    tick = Fraction(tick)
    if min_step is None:
        min_step = tick
    stats = AlignmentStats()

    # 1. exact timebase conversion
    observed = [to_seconds(pts, timebase) for pts in pts_list]
    n = len(observed)

    # 2. drift estimation against the nominal grid
    corrector = DriftCorrector()
    if correct_drift and n > 0:
        # The nominal grid is anchored at the first observed timestamp: only
        # the clock *rate* (skew) is corrected, the stream's absolute start
        # position is never moved (no unobservable offset removal).
        anchor = observed[0]
        if nominal_durations is not None:
            if len(nominal_durations) != n:
                raise ValueError("nominal_durations must match pts_list")
            expected = []
            acc = anchor
            for d in nominal_durations:
                expected.append(acc)
                acc += Fraction(d)
        elif nominal_interval is not None:
            interval = Fraction(nominal_interval)
            expected = [anchor + interval * i for i in range(n)]
        else:
            # No nominal timing known: drift is unobservable, skip correction.
            expected = None
        if expected is not None:
            corrector.fit(observed, expected)
            stats.skew = corrector.skew
            stats.offset = corrector.offset
            stats.drift_corrected = True
            stats.fit_outliers = corrector.outliers
            corrected = [corrector.correct(t) for t in observed]
        else:
            corrector.fit([], [])
            corrected = observed
    else:
        corrector.fit([], [])
        corrected = observed

    # 3. monotonic filter (final guarantee on the output)
    mono = MonotonicFilter(policy=policy, min_step=min_step,
                           jump_threshold=jump_threshold, stats=stats)
    seconds_out = []
    for ts in corrected:
        emitted = mono.push(ts)
        if emitted is not None:
            seconds_out.append(emitted)

    ticks = [seconds_to_ticks(s, tick) for s in seconds_out]
    return AlignmentResult(ticks, seconds_out, stats, tick)


def pairing_errors(ticks_a, ticks_b):
    """For each tick in stream A, the signed distance to the nearest tick in
    stream B (two-pointer merge, O(n)). Both inputs must be sorted."""
    errors = []
    j = 0
    nb = len(ticks_b)
    for ta in ticks_a:
        while j + 1 < nb and abs(ticks_b[j + 1] - ta) <= abs(ticks_b[j] - ta):
            j += 1
        if nb:
            errors.append(ta - ticks_b[j])
    return errors


def error_stats(errors, tick=DEFAULT_TICK):
    """Summary statistics (in seconds) for a list of tick errors."""
    tick = float(Fraction(tick))
    if not errors:
        return {"count": 0, "mean": 0.0, "rms": 0.0, "max_abs": 0.0, "p95": 0.0}
    vals = [e * tick for e in errors]
    abs_sorted = sorted(abs(v) for v in vals)
    n = len(vals)
    mean = sum(vals) / n
    rms = (sum(v * v for v in vals) / n) ** 0.5
    p95 = abs_sorted[min(n - 1, int(0.95 * n))]
    return {"count": n, "mean": mean, "rms": rms,
            "max_abs": abs_sorted[-1], "p95": p95}
