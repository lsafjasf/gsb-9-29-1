"""Shared helpers for the decaying-counter test suite."""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decaying_counter import DecayingCounter  # noqa: E402
from buggy_counter import BuggyCounter        # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def admit(events, max_lateness):
    """Reference admission policy, recomputed from scratch.

    Applies dedup (first occurrence of an id wins) and the bounded
    lateness rule against the running watermark, in arrival order.
    Returns (kept_events, dropped_records).
    """
    seen = set()
    watermark = None
    kept = []
    dropped = []
    for event_id, t, value in events:
        if event_id in seen:
            dropped.append(("duplicate", event_id, t))
            continue
        if watermark is not None and t < watermark - max_lateness:
            dropped.append(("late", event_id, t))
            continue
        seen.add(event_id)
        kept.append((event_id, t, value))
        if watermark is None or t > watermark:
            watermark = t
    return kept, dropped


def oracle_query(kept_events, start, end, half_life):
    """Full per-event recomputation (the ground truth).

    Deliberately uses arrival order and math.pow so that floating-point
    behaviour differs from the implementation under test.
    """
    total = 0.0
    for _event_id, t, value in kept_events:
        if start <= t <= end:
            total += value * math.pow(2.0, -(end - t) / half_life)
    return total


def make_events(rng, count, span, out_of_order=0.3, duplicate_rate=0.1,
                base_time=1_000.0, jitter=None):
    """Random event stream with out-of-order arrivals and duplicates.

    ``jitter`` is the typical out-of-order displacement in time units;
    events are swapped with partners up to ~2*jitter away, so a mix of
    in-bound and beyond-bound lateness results when jitter is comparable
    to the lateness limit.  Returns (event_id, timestamp, value) tuples
    in arrival order.
    """
    events = []
    next_id = 0
    for i in range(count):
        # Roughly in-order stream: event times increase with arrival
        # index, with per-event noise below the mean spacing.
        t = base_time + (i + rng.random()) * (span / count)
        value = round(rng.uniform(0.5, 5.0), 6)
        events.append([f"e{next_id}", t, value])
        next_id += 1
    # Inject duplicates of already-seen events.
    for _ in range(int(count * duplicate_rate)):
        source = events[rng.randrange(len(events))]
        events.append([source[0], source[1], source[2]])
    # Bounded-displacement shuffling for out-of-order arrivals.
    if jitter is None:
        jitter = span * 0.05
    max_step = max(1, int(2.0 * jitter / span * count))
    for i in range(len(events)):
        if rng.random() < out_of_order:
            j = min(len(events) - 1, max(0, i + rng.randint(-max_step, max_step)))
            events[i], events[j] = events[j], events[i]
    return [tuple(e) for e in events]


def random_queries(rng, count, time_lo, time_hi, half_life):
    """Random query intervals, including empty / point / huge ranges."""
    queries = []
    for _ in range(count):
        kind = rng.random()
        if kind < 0.1:
            # Range outside all data (empty result).
            start = time_hi + rng.random() * half_life * 10
            end = start + rng.random() * half_life * 10
        elif kind < 0.2:
            # Point query.
            start = rng.uniform(time_lo, time_hi)
            end = start
        elif kind < 0.3:
            # Huge range spanning far beyond the data.
            start = time_lo - rng.random() * half_life * 1000
            end = time_hi + rng.random() * half_life * 1000
        else:
            a = rng.uniform(time_lo, time_hi)
            b = rng.uniform(time_lo, time_hi)
            start, end = min(a, b), max(a, b)
        queries.append((start, end))
    return queries


def rel_error(actual, expected):
    if expected == 0.0:
        return abs(actual)
    return abs(actual - expected) / abs(expected)
