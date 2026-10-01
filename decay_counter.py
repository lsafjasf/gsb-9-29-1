"""Time-decaying counter with arbitrary sliding-window queries.

Semantic contract (the reference oracle in ``oracle.py`` implements the
identical contract by brute force):

* Decay is exponential in *event time*. The weight of an event ``(t, v)`` at
  evaluation time ``T`` is ``v * 2 ** (-max(0, T - t) / half_life)``.
* ``query(start, end)`` returns the sum of event weights evaluated at
  ``T = end`` over all accepted events with ``start <= t <= end`` (both
  endpoints inclusive).
* Events are de-duplicated by ``event_id``: the first copy wins, later copies
  are ignored and observable via ``stats["duplicates"]``.
* Out-of-order arrival is bounded: after an event with timestamp ``m`` arrived,
  an event with ``t < m - max_lateness`` is rejected ("dropped late"). Drops
  are observable via ``stats["dropped_late"]``, ``dropped_events`` and the
  ``on_drop`` callback. Events arriving at the boundary exactly are accepted.
* The wall clock only moves forward: a backwards clock is clamped to the last
  reading, the rewind is recorded in ``stats["clock_rewinds"]`` and reported
  through ``on_clock_rewind``. No accumulated quantity can go backwards.
* ``snapshot()``/``restore()`` persist every accepted event, so a process
  restart yields exactly the same state and answers as an uninterrupted run.
Only the Python standard library is used.
"""

from __future__ import annotations

import json
import math
import time
from bisect import bisect_left, bisect_right, insort

__all__ = ["DecayCounter"]


class DecayCounter:
    def __init__(self, half_life, bucket_width=None, max_lateness=0.0,
                 clock=time.monotonic, on_drop=None, on_clock_rewind=None):
        half_life = float(half_life)
        if half_life <= 0.0:
            raise ValueError("half_life must be positive")
        if max_lateness < 0.0:
            raise ValueError("max_lateness must be non-negative")
        bucket_width = float(bucket_width) if bucket_width is not None else half_life
        if bucket_width <= 0.0:
            raise ValueError("bucket_width must be positive")

        self.half_life = half_life
        self.bucket_width = bucket_width
        self.max_lateness = float(max_lateness)
        self._clock = clock
        self._on_drop = on_drop
        self._on_clock_rewind = on_clock_rewind

        self._buckets = {}
        self._agg = {}
        self._keys = []
        self._seen = set()
        self._max_event_ts = None
        self._last_now = None

        self.stats = {
            "accepted": 0,
            "duplicates": 0,
            "dropped_late": 0,
            "clock_rewinds": 0,
        }
        self.dropped_events = []

    def _weight(self, dt):
        return math.exp2(-max(0.0, float(dt)) / self.half_life)

    def add(self, event_id, ts, value=1.0):
        """Ingest one event; returns "accepted", "duplicate" or "dropped"."""
        ts = float(ts)
        value = float(value)
        if not math.isfinite(ts) or not math.isfinite(value):
            raise ValueError("ts and value must be finite")

        if event_id in self._seen:
            self.stats["duplicates"] += 1
            return "duplicate"

        if (self._max_event_ts is not None
                and ts < self._max_event_ts - self.max_lateness):
            record = (event_id, ts, value)
            self.dropped_events.append(record)
            self.stats["dropped_late"] += 1
            if self._on_drop is not None:
                self._on_drop(event_id, ts, value)
            return "dropped"

        self._seen.add(event_id)
        self.stats["accepted"] += 1
        if self._max_event_ts is None or ts > self._max_event_ts:
            self._max_event_ts = ts

        k = math.floor(ts / self.bucket_width)
        bucket = self._buckets.get(k)
        if bucket is None:
            bucket = []
            self._buckets[k] = bucket
            self._agg[k] = 0.0
            insort(self._keys, k)
        bucket.append((ts, value))
        bucket_end = (k + 1) * self.bucket_width
        self._agg[k] += value * self._weight(bucket_end - ts)
        return "accepted"

    def query(self, start, end):
        """Decayed weighted sum over events in the inclusive interval."""
        start = float(start)
        end = float(end)
        if end < start:
            raise ValueError("query end precedes start")
        if not self._keys:
            return 0.0

        g = self.bucket_width
        first = bisect_left(self._keys, math.floor(start / g))
        last = bisect_right(self._keys, math.floor(end / g))

        total = 0.0
        for k in self._keys[first:last]:
            bucket_lo = k * g
            bucket_hi = bucket_lo + g
            if bucket_lo >= start and bucket_hi <= end:
                total += self._agg[k] * self._weight(end - bucket_hi)
            else:
                for ts, value in self._buckets[k]:
                    if start <= ts <= end:
                        total += value * self._weight(end - ts)
        return total

    def value_at(self, T=None):
        """Decayed weighted sum of all events with ``t <= T``."""
        if T is None:
            T = self._now()
        T = float(T)
        if not self._keys or self._max_event_ts is None:
            return 0.0

        g = self.bucket_width
        upper = bisect_right(self._keys, math.floor(T / g))

        total = 0.0
        for k in self._keys[:upper]:
            bucket_hi = (k + 1) * g
            if bucket_hi <= T:
                total += self._agg[k] * self._weight(T - bucket_hi)
            else:
                for ts, value in self._buckets[k]:
                    if ts <= T:
                        total += value * self._weight(T - ts)
        return total

    def _now(self):
        raw = float(self._clock())
        if self._last_now is None:
            self._last_now = raw
        elif raw < self._last_now:
            self.stats["clock_rewinds"] += 1
            if self._on_clock_rewind is not None:
                self._on_clock_rewind(raw, self._last_now)
        else:
            self._last_now = raw
        return self._last_now

    def snapshot(self):
        """JSON-serialisable, event-level state of the counter."""
        return {
            "version": 1,
            "half_life": self.half_life,
            "bucket_width": self.bucket_width,
            "max_lateness": self.max_lateness,
            "buckets": {
                str(k): [[ts, value] for ts, value in self._buckets[k]]
                for k in self._keys
            },
            "seen_ids": sorted(self._seen),
            "max_event_ts": self._max_event_ts,
            "last_now": self._last_now,
            "stats": dict(self.stats),
            "dropped_events": [list(record) for record in self.dropped_events],
        }

    def snapshot_json(self):
        return json.dumps(self.snapshot(), sort_keys=True)

    @classmethod
    def restore(cls, snapshot, clock=time.monotonic,
                on_drop=None, on_clock_rewind=None):
        if isinstance(snapshot, str):
            snapshot = json.loads(snapshot)
        if snapshot.get("version") != 1:
            raise ValueError("unsupported snapshot version")

        counter = cls(
            snapshot["half_life"],
            bucket_width=snapshot["bucket_width"],
            max_lateness=snapshot["max_lateness"],
            clock=clock,
            on_drop=on_drop,
            on_clock_rewind=on_clock_rewind,
        )
        for key, events in snapshot["buckets"].items():
            bucket_index = int(key)
            events = [(float(ts), float(value)) for ts, value in events]
            counter._buckets[bucket_index] = events
            bucket_end = (bucket_index + 1) * counter.bucket_width
            counter._agg[bucket_index] = sum(
                value * counter._weight(bucket_end - ts)
                for ts, value in events
            )
        counter._keys = sorted(counter._buckets)
        counter._seen = set(snapshot["seen_ids"])
        counter._max_event_ts = snapshot["max_event_ts"]
        counter._last_now = snapshot["last_now"]
        counter.stats = dict(snapshot["stats"])
        counter.dropped_events = [
            tuple(record) for record in snapshot["dropped_events"]
        ]
        return counter
