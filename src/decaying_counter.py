"""Time-decaying counter with sliding-window interval queries.

Model
-----
Each event has a unique id, an event time ``t`` and a value (default 1).
The weight of an event inside a query interval ``[start, end]`` is

    value * 2 ** (-(end - t) / half_life)          if start <= t <= end

and zero otherwise.  Query results therefore depend only on event data
and the requested interval: they never depend on "current window" state,
so crossing a window boundary cannot produce a jump.

Bounded out-of-order ingestion
------------------------------
The watermark is the maximum event time accepted so far.  An event with
time ``t`` is accepted iff ``t >= watermark - max_lateness`` (the
boundary itself is accepted).  Older events are rejected and recorded.
Duplicate ids are ignored and recorded as well; both rejection types
appear in ``self.journal`` and are counted in ``self.stats``.

Clock
-----
``now_fn`` is injectable (e.g. ``time.monotonic`` in production, a
scripted function in tests).  If the clock moves backwards the
effective "now" is clamped to the last observed value; the regression is
recorded.  State and cumulative quantities therefore never move
backwards or become negative because of a clock regression.

Restart / persistence
---------------------
``snapshot()`` returns a JSON-safe dict containing the full state.
``DecayingCounter.restore(snapshot)`` rebuilds an identical counter, so
a process restart followed by continued ingestion produces exactly the
same answers as an uninterrupted run.
"""

from bisect import bisect_left, bisect_right


class DecayingCounter:
    def __init__(self, half_life, max_lateness, now_fn=None):
        if half_life <= 0:
            raise ValueError("half_life must be positive")
        if max_lateness < 0:
            raise ValueError("max_lateness must be non-negative")
        self.half_life = float(half_life)
        self.max_lateness = float(max_lateness)
        self._now_fn = now_fn

        self._times = []            # sorted unique event times
        self._values = {}           # event time -> aggregated value
        self._seen_ids = set()      # accepted event ids (dedup)
        self._watermark = None      # max accepted event time
        self._now = None            # monotone effective clock value

        # Observable records of everything the counter refuses or clamps.
        self.journal = []
        self.stats = {
            "accepted": 0,
            "duplicates": 0,
            "late": 0,
            "clock_regressions": 0,
        }

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------
    def now(self):
        """Return the effective current time; never moves backwards."""
        if self._now_fn is None:
            if self._now is not None:
                return self._now
            return self._watermark
        t = float(self._now_fn())
        if t != t:  # NaN
            raise ValueError("clock returned NaN")
        if self._now is not None and t < self._now:
            self.stats["clock_regressions"] += 1
            self.journal.append({
                "kind": "clock_regression",
                "observed": t,
                "clamped_to": self._now,
            })
            return self._now
        self._now = t
        return t

    @property
    def watermark(self):
        return self._watermark

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def add(self, event_id, timestamp=None, value=1.0):
        """Ingest one event; return True if accepted, False if rejected.

        Duplicates are checked first, then the bounded-lateness rule.
        """
        if timestamp is None:
            t = self.now()
        else:
            t = float(timestamp)
        if t != t:
            raise ValueError("event timestamp is NaN")
        value = float(value)

        if event_id in self._seen_ids:
            self.stats["duplicates"] += 1
            self.journal.append({
                "kind": "duplicate",
                "id": event_id,
                "time": t,
                "value": value,
            })
            return False

        if self._watermark is not None and t < self._watermark - self.max_lateness:
            self.stats["late"] += 1
            self.journal.append({
                "kind": "late",
                "id": event_id,
                "time": t,
                "value": value,
                "watermark": self._watermark,
                "max_lateness": self.max_lateness,
                "limit": self._watermark - self.max_lateness,
            })
            return False

        self._seen_ids.add(event_id)
        index = bisect_left(self._times, t)
        if index < len(self._times) and self._times[index] == t:
            self._values[t] += value
        else:
            self._times.insert(index, t)
            self._values[t] = value
        if self._watermark is None or t > self._watermark:
            self._watermark = t
        self.stats["accepted"] += 1
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def query(self, start, end):
        """Decayed total of events with start <= t <= end, evaluated at end."""
        start = float(start)
        end = float(end)
        if end < start:
            raise ValueError("query end must be >= start")
        lo = bisect_left(self._times, start)
        hi = bisect_right(self._times, end)
        half_life = self.half_life
        total = 0.0
        # Ascending event-time order; the oracle in the tests sums in a
        # different order so float drift is actually exercised.
        for index in range(lo, hi):
            event_time = self._times[index]
            total += self._values[event_time] * (
                2.0 ** (-(end - event_time) / half_life)
            )
        return total

    def total(self, at=None):
        """Decayed total of all events at time ``at`` (default: now()).

        Events later than ``at`` are excluded, and the effective clock is
        monotone, so this never decreases or turns negative on a clock
        regression.
        """
        if at is None:
            at = self.now()
        return self.query(float("-inf"), at)

    # ------------------------------------------------------------------
    # Persistence (exact, JSON-safe)
    # ------------------------------------------------------------------
    def snapshot(self):
        return {
            "version": 1,
            "half_life": self.half_life,
            "max_lateness": self.max_lateness,
            "events": [[t, self._values[t]] for t in self._times],
            "seen_ids": sorted(self._seen_ids, key=repr),
            "watermark": self._watermark,
            "now": self._now,
            "stats": dict(self.stats),
            "journal": list(self.journal),
        }

    @classmethod
    def restore(cls, snapshot, now_fn=None):
        data = snapshot
        counter = cls(data["half_life"], data["max_lateness"], now_fn=now_fn)
        for event_time, value in data["events"]:
            event_time = float(event_time)
            counter._times.append(event_time)
            counter._values[event_time] = float(value)
        counter._seen_ids = set(data["seen_ids"])
        counter._watermark = data["watermark"]
        counter._now = data["now"]
        counter.stats = dict(data["stats"])
        counter.journal = list(data["journal"])
        return counter
