"""Legacy tumbling-window decaying counter (intentionally flawed).

Kept ONLY so the reproduction tests can demonstrate the production
incidents:

1. The in-progress bucket is excluded from queries; when a bucket seals
   (window switch) all of its events appear at once -> a jump.
2. Events belonging to an already sealed bucket are misattributed to
   the current bucket -> out-of-order drift.
3. Event ids are ignored -> duplicates are counted multiple times.
4. Points inside a bucket are stored by raw timestamp -> several events
   at the same timestamp overwrite each other.
5. A bucket overlapping a query interval is counted wholesale, including
   points outside the interval -> cross-window queries disagree with a
   per-event recomputation.
6. snapshot() forgets the watermark/current bucket -> restart changes
   the answers.
"""


class BuggyCounter:
    def __init__(self, half_life, window=60.0):
        self.half_life = float(half_life)
        self.window = float(window)
        self.buckets = {}   # bucket index -> {"anchor", "points": {ts: v}}
        self.current = None

    def add(self, event_id, timestamp, value=1.0):
        t = float(timestamp)
        idx = int(t // self.window)
        # Flaw 2: late events are forced into the current bucket.
        if self.current is not None and idx < self.current:
            idx = self.current
        if self.current is None or idx > self.current:
            self.current = idx
        bucket = self.buckets.get(idx)
        if bucket is None:
            bucket = {"anchor": (idx + 1) * self.window, "points": {}}
            self.buckets[idx] = bucket
        # Flaw 3: event_id is never consulted.
        # Flaw 4: same-timestamp events overwrite one another.
        bucket["points"][t] = float(value)

    def query(self, start, end):
        start = float(start)
        end = float(end)
        total = 0.0
        for idx, bucket in self.buckets.items():
            # Flaw 1: the in-progress bucket is invisible until it seals.
            if self.current is not None and idx >= self.current:
                continue
            window_start = idx * self.window
            window_end = window_start + self.window
            # Flaw 5: bucket overlap includes every point in the bucket.
            if window_end < start or window_start > end:
                continue
            anchor = bucket["anchor"]
            for event_time, value in bucket["points"].items():
                total += (
                    value
                    * (2.0 ** (-(anchor - event_time) / self.half_life))
                    * (2.0 ** (-(end - anchor) / self.half_life))
                )
        return total

    def snapshot(self):
        # Flaw 6: the current bucket / watermark is not persisted.
        return {"half_life": self.half_life, "window": self.window,
                "buckets": self.buckets}

    @classmethod
    def restore(cls, data):
        counter = cls(data["half_life"], data["window"])
        counter.buckets = data["buckets"]
        return counter
