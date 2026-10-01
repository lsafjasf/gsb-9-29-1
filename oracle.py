"""Brute-force reference implementation of the decay-counter contract.

This is the full-recompute oracle used for differential testing. It keeps
every accepted event and scans all of them per query. Its contract is
identical to ``decay_counter.DecayCounter``: de-dup by event id, bounded
lateness watermark (events with ``t < max_ts - max_lateness`` are dropped),
exponential event-time decay and inclusive interval semantics.
"""

from __future__ import annotations

__all__ = ["Oracle"]


class Oracle:
    def __init__(self, half_life, max_lateness=0.0):
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        self.half_life = float(half_life)
        self.max_lateness = float(max_lateness)
        self.events = []
        self.seen = set()
        self.max_event_ts = None
        self.stats = {
            "accepted": 0,
            "duplicates": 0,
            "dropped_late": 0,
            "clock_rewinds": 0,
        }
        self.dropped_events = []

    @staticmethod
    def weight(half_life, ts, T):
        dt = T - ts
        if dt < 0.0:
            dt = 0.0
        return 2.0 ** (-dt / half_life)

    def add(self, event_id, ts, value=1.0):
        ts = float(ts)
        value = float(value)
        if event_id in self.seen:
            self.stats["duplicates"] += 1
            return "duplicate"
        if (self.max_event_ts is not None
                and ts < self.max_event_ts - self.max_lateness):
            self.dropped_events.append((event_id, ts, value))
            self.stats["dropped_late"] += 1
            return "dropped"
        self.seen.add(event_id)
        self.stats["accepted"] += 1
        self.events.append((ts, value))
        if self.max_event_ts is None or ts > self.max_event_ts:
            self.max_event_ts = ts
        return "accepted"

    def query(self, start, end):
        if end < start:
            raise ValueError("query end precedes start")
        return sum(
            value * self.weight(self.half_life, ts, end)
            for ts, value in self.events
            if start <= ts <= end
        )

    def value_at(self, T):
        return sum(
            value * self.weight(self.half_life, ts, T)
            for ts, value in self.events
            if ts <= T
        )
