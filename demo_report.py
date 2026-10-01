"""Long-duration drift experiment: two streams, two timebases, clock skew,
jitter, an injected timestamp rollback and a forward jump.

Generates alignment-error data before/after drift correction and verifies
output monotonicity. Writes drift_report.txt.

Run: python3 demo_report.py
"""

import random
from fractions import Fraction

from tsalign import (align_stream, error_stats, pairing_errors, to_seconds,
                     DEFAULT_TICK)

DURATION_S = 3600          # one hour
VIDEO_FPS = 25
AUDIO_SAMPLES = 1024
AUDIO_RATE = 48000
TB_VIDEO = Fraction(1, 90000)
TB_AUDIO = Fraction(1, 48000)
SKEW_VIDEO = Fraction(100005, 100000)   # +50 ppm clock
SKEW_AUDIO = Fraction(99997, 100000)    # -30 ppm clock
ROLLBACK_AT = 0.40         # fraction of the stream
JUMP_AT = 0.70
JUMP_SECONDS = 2


def rms(vals):
    return (sum(v * v for v in vals) / len(vals)) ** 0.5 if vals else 0.0


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def simulate(n_frames, interval, tb, skew, jitter_s, rollback_at=None,
             jump_at=None, seed=0):
    """Returns (pts_list, true_grid). true_grid includes the injected jump so
    residuals can be measured against ground truth."""
    rng = random.Random(seed)
    pts, true_grid = [], []
    gap = Fraction(0)
    for i in range(n_frames):
        t = interval * i
        if jump_at is not None and i == jump_at:
            gap += JUMP_SECONDS
        true_grid.append(t + gap)
        observed = (t + gap) * skew
        observed += Fraction(int(rng.uniform(-1, 1) * jitter_s / DEFAULT_TICK)) * DEFAULT_TICK
        if rollback_at is not None and i == rollback_at:
            observed = (interval * (i - 20) + gap) * skew  # stamped 20 frames back
        pts.append(int(observed / tb))
    return pts, true_grid


def residual_stats(seconds_out, true_grid, skip=()):
    res = [float(seconds_out[i] - true_grid[i])
           for i in range(len(true_grid)) if i not in skip]
    return {"rms": rms(res), "max_abs": max(abs(r) for r in res),
            "p95": pct([abs(r) for r in res], 0.95)}


def fmt_s(x):
    return "%.3f ms" % (x * 1e3) if abs(x) < 1 else "%.3f s" % x


def main():
    n_v = DURATION_S * VIDEO_FPS
    n_a = DURATION_S * AUDIO_RATE // AUDIO_SAMPLES
    iv_v = Fraction(1, VIDEO_FPS)
    iv_a = Fraction(AUDIO_SAMPLES, AUDIO_RATE)
    rb_v, rb_a = int(n_v * ROLLBACK_AT), int(n_a * ROLLBACK_AT)
    jp_v, jp_a = int(n_v * JUMP_AT), int(n_a * JUMP_AT)

    pts_v, true_v = simulate(n_v, iv_v, TB_VIDEO, SKEW_VIDEO, 300e-6,
                             rollback_at=rb_v, jump_at=jp_v, seed=1)
    pts_a, true_a = simulate(n_a, iv_a, TB_AUDIO, SKEW_AUDIO, 150e-6,
                             rollback_at=rb_a, jump_at=jp_a, seed=2)

    rv = align_stream(pts_v, TB_VIDEO, nominal_interval=iv_v,
                      jump_threshold=Fraction(1))
    ra = align_stream(pts_a, TB_AUDIO, nominal_interval=iv_a,
                      jump_threshold=Fraction(1))

    # --- monotonicity assertion (must never go backwards) ---
    for res, name in ((rv, "video"), (ra, "audio")):
        for i in range(1, len(res.ticks)):
            assert res.ticks[i] > res.ticks[i - 1], \
                "%s output not monotone at %d" % (name, i)

    # --- drift before correction (raw pts vs ground truth) ---
    raw_v = [to_seconds(p, TB_VIDEO) for p in pts_v]
    raw_a = [to_seconds(p, TB_AUDIO) for p in pts_a]
    skip_v, skip_a = {rb_v}, {rb_a}
    before_v = residual_stats(raw_v, true_v, skip_v)
    before_a = residual_stats(raw_a, true_a, skip_a)
    desync_before = abs(float(raw_v[-1] - true_v[-1])
                        - float(raw_a[-1] - true_a[-1]))

    # --- residual after correction ---
    after_v = residual_stats(rv.seconds, true_v, skip_v)
    after_a = residual_stats(ra.seconds, true_a, skip_a)
    desync_after = abs(float(rv.seconds[-1] - true_v[-1])
                       - float(ra.seconds[-1] - true_a[-1]))

    paired = error_stats(pairing_errors(rv.ticks, ra.ticks))

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 68)
    out("tsalign drift experiment: %d s, video %d fps @1/90000, audio %d/%d Hz"
        % (DURATION_S, VIDEO_FPS, AUDIO_SAMPLES, AUDIO_RATE))
    out("injected: 1 rollback/stream @%d%%, +%d s jump @%d%%, jitter 300/150 us"
        % (ROLLBACK_AT * 100, JUMP_SECONDS, JUMP_AT * 100))
    out("=" * 68)
    out("")
    out("[clock skew estimation]   true        estimated")
    out("  video                   %+d ppm     %+.2f ppm"
        % (round(float(SKEW_VIDEO - 1) * 1e6), (rv.stats.skew - 1) * 1e6))
    out("  audio                   %+d ppm     %+.2f ppm"
        % (round(float(SKEW_AUDIO - 1) * 1e6), (ra.stats.skew - 1) * 1e6))
    out("  fit outliers rejected:  video=%d audio=%d (rollback+jump segment)"
        % (rv.stats.fit_outliers, ra.stats.fit_outliers))
    out("")
    out("[alignment error vs ground truth]   rms          p95          max")
    out("  video BEFORE correction      %10s   %10s   %10s"
        % (fmt_s(before_v["rms"]), fmt_s(before_v["p95"]), fmt_s(before_v["max_abs"])))
    out("  video AFTER  correction      %10s   %10s   %10s"
        % (fmt_s(after_v["rms"]), fmt_s(after_v["p95"]), fmt_s(after_v["max_abs"])))
    out("  audio BEFORE correction      %10s   %10s   %10s"
        % (fmt_s(before_a["rms"]), fmt_s(before_a["p95"]), fmt_s(before_a["max_abs"])))
    out("  audio AFTER  correction      %10s   %10s   %10s"
        % (fmt_s(after_a["rms"]), fmt_s(after_a["p95"]), fmt_s(after_a["max_abs"])))
    out("")
    out("[A/V desync at end of %d s]" % DURATION_S)
    out("  BEFORE correction: %s" % fmt_s(desync_before))
    out("  AFTER  correction: %s" % fmt_s(desync_after))
    out("")
    out("[cross-stream frame pairing after correction]")
    out("  rms=%s  p95=%s  max=%s  (audio frame = %.3f ms)"
        % (fmt_s(paired["rms"]), fmt_s(paired["p95"]), fmt_s(paired["max_abs"]),
           float(iv_a) * 1e3))
    out("")
    out("[event handling]")
    out("  rollbacks clamped: video=%d audio=%d (policy=CLAMP, min_step=1 tick)"
        % (rv.stats.rollbacks, ra.stats.rollbacks))
    out("  forward jumps passed through: video=%d audio=%d"
        % (len(rv.stats.jumps), len(ra.stats.jumps)))
    out("")
    out("[monotonicity] ASSERTION PASSED: %d video + %d audio output ticks "
        "strictly increasing" % (len(rv.ticks), len(ra.ticks)))

    with open("drift_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
