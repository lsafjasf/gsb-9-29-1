from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path

from timestamp_sync import (
    MonotonicClock,
    RollbackPolicy,
    assert_monotonic,
    error_statistics,
    fit_drift_model,
    to_nanoseconds,
)


DAYS = 30
SECONDS_PER_DAY = 24 * 60 * 60
DURATION = DAYS * SECONDS_PER_DAY
VIDEO_TIME_BASE = Fraction(1, 90000)
AUDIO_TIME_BASE = Fraction(1, 48000)
VIDEO_WRAP_TICKS = 2**33
VIDEO_SLOPE = Fraction(1000030, 1000000)
AUDIO_SLOPE = Fraction(999985, 1000000)
VIDEO_OFFSET = Fraction(1, 5)
AUDIO_OFFSET = Fraction(2, 25)
AUDIO_RESET = SECONDS_PER_DAY
VIDEO_JUMP = DURATION - 6 * 3600
VIDEO_JUMP_SECONDS = Fraction(1, 10)


def observed_seconds(ideal: Fraction, *, video: bool) -> Fraction:
    slope = VIDEO_SLOPE if video else AUDIO_SLOPE
    offset = VIDEO_OFFSET if video else AUDIO_OFFSET
    return ideal * slope + offset


def video_tick(ideal: Fraction) -> int:
    observed = observed_seconds(ideal, video=True)
    tick = observed // VIDEO_TIME_BASE
    if ideal >= VIDEO_JUMP:
        tick += VIDEO_JUMP_SECONDS // VIDEO_TIME_BASE
    return tick % VIDEO_WRAP_TICKS


def audio_tick(ideal: Fraction) -> int:
    if ideal >= AUDIO_RESET:
        observed = (ideal - AUDIO_RESET) * AUDIO_SLOPE
    else:
        observed = observed_seconds(ideal, video=False)
    return observed // AUDIO_TIME_BASE


def build_samples() -> list[Fraction]:
    samples = [Fraction(hour * 3600) for hour in range(DAYS * 24 + 1)]
    samples.append(Fraction(AUDIO_RESET) - Fraction(1, 1000))
    samples.append(Fraction(AUDIO_RESET) + Fraction(1, 1000))
    return sorted(set(samples))


def model_summary(model) -> dict:
    return {
        "slope_exact": str(model.slope),
        "intercept_exact": str(model.intercept),
        "slope_ppm": float((model.slope - 1) * 1_000_000),
        "intercept_seconds": float(model.intercept),
    }


def main() -> None:
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    ideal_times = build_samples()

    video_clock = MonotonicClock(
        VIDEO_TIME_BASE,
        wrap_period_ticks=VIDEO_WRAP_TICKS,
        rollback_policy=RollbackPolicy.CLAMP,
    )
    audio_clock = MonotonicClock(
        AUDIO_TIME_BASE,
        rollback_policy=RollbackPolicy.CLAMP,
    )
    video_frames = video_clock.normalize_ticks(video_tick(ideal) for ideal in ideal_times)
    audio_frames = audio_clock.normalize_ticks(audio_tick(ideal) for ideal in ideal_times)

    video_model = fit_drift_model(
        [(frame.time, ideal) for frame, ideal in zip(video_frames, ideal_times)]
    )
    audio_model = fit_drift_model(
        [(frame.time, ideal) for frame, ideal in zip(audio_frames, ideal_times)]
    )
    assert_monotonic(frame.time for frame in video_frames)
    assert_monotonic(frame.time for frame in audio_frames)

    video_corrected_times = [video_model.corrected(frame.time) for frame in video_frames]
    audio_corrected_times = [audio_model.corrected(frame.time) for frame in audio_frames]
    assert_monotonic(video_corrected_times)
    assert_monotonic(audio_corrected_times)

    rows = []
    for ideal, video_frame, audio_frame, video_corrected, audio_corrected in zip(
        ideal_times,
        video_frames,
        audio_frames,
        video_corrected_times,
        audio_corrected_times,
    ):
        video_raw = video_frame.time
        audio_raw = audio_frame.time
        rows.append(
            {
                "ideal_seconds": str(ideal),
                "video_raw_seconds": str(video_raw),
                "audio_raw_seconds": str(audio_raw),
                "video_corrected_seconds": str(video_corrected),
                "audio_corrected_seconds": str(audio_corrected),
                "video_raw_error_ns": to_nanoseconds(video_raw - ideal),
                "audio_raw_error_ns": to_nanoseconds(audio_raw - ideal),
                "video_residual_ns": to_nanoseconds(video_corrected - ideal),
                "audio_residual_ns": to_nanoseconds(audio_corrected - ideal),
                "raw_av_diff_ns": to_nanoseconds(video_raw - audio_raw),
                "corrected_av_diff_ns": to_nanoseconds(video_corrected - audio_corrected),
            }
        )

    csv_path = output_dir / "alignment_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "duration_days": DAYS,
        "duration_seconds": DURATION,
        "sample_count": len(rows),
        "time_bases": {
            "video_seconds_per_tick": str(VIDEO_TIME_BASE),
            "audio_seconds_per_tick": str(AUDIO_TIME_BASE),
            "video_wrap_period_ticks": VIDEO_WRAP_TICKS,
        },
        "synthetic_clock_truth_ppm": {"video": 30, "audio": -15},
        "estimated_models": {
            "video": model_summary(video_model),
            "audio": model_summary(audio_model),
        },
        "events": {
            "video_wraps": sum(frame.event.value == "wrap" for frame in video_frames),
            "audio_reset_clamps": sum(
                frame.event.value == "clamped" for frame in audio_frames
            ),
            "video_forward_jump_at_seconds": VIDEO_JUMP,
            "audio_reset_at_seconds": AUDIO_RESET,
        },
        "raw_error": {
            "video_vs_reference": error_statistics(
                Fraction(row["video_raw_error_ns"], 1_000_000_000) for row in rows
            ),
            "audio_vs_reference": error_statistics(
                Fraction(row["audio_raw_error_ns"], 1_000_000_000) for row in rows
            ),
            "av_difference": error_statistics(
                Fraction(row["raw_av_diff_ns"], 1_000_000_000) for row in rows
            ),
        },
        "corrected_residual": {
            "video_vs_reference": error_statistics(
                Fraction(row["video_residual_ns"], 1_000_000_000) for row in rows
            ),
            "audio_vs_reference": error_statistics(
                Fraction(row["audio_residual_ns"], 1_000_000_000) for row in rows
            ),
            "av_difference": error_statistics(
                Fraction(row["corrected_av_diff_ns"], 1_000_000_000) for row in rows
            ),
        },
    }

    json_path = output_dir / "alignment_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {csv_path} ({len(rows)} rows)")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
