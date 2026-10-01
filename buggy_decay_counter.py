"""Original monitoring implementation, kept only to reproduce the bugs.

Defects (fixed in ``decay_counter.py``):

1. Tumbling windows: only the previous and current window are retained and the
   decay is applied in one lumpy step at rotation based on wall-clock time.
   Queries crossing a rotation boundary jump, and an event whose timestamp is
   exactly on a window boundary falls into the next window where a closed
   interval ending at that boundary gives it zero overlap.
2. No de-duplication: redelivered events are counted again.
3. The running total decays events in *arrival* order using arrival gaps; with
   out-of-order arrival the gaps are negative, so weights exceed one and the
   total drifts above the true value.
4. Snapshots persist only window sums and the running total; after restore the
   decay clock restarts at zero, so recovered answers disagree with a run that
   never restarted.
"""

from __future__ import annotations


class BuggyDecayCounter:
    def __init__(self, half_life, window):
        self.half_life = float(half_life)
        self.window = float(window)
        self._prev = 0.0
        self._cur = 0.0
        self._cur_index = 0
        self._total = 0.0
        self._last_event_ts = None

    def _rotate(self, index):
        while self._cur_index < index:
            self._prev = self._cur * (0.5 ** (self.window / self.half_life))
            self._cur = 0.0
            self._cur_index += 1

    def add(self, event_id, ts, value=1.0):
        ts = float(ts)
        value = float(value)

        if self._last_event_ts is not None:
            gap = ts - self._last_event_ts
            self._total *= 0.5 ** (gap / self.half_life)
        self._last_event_ts = ts
        self._total += value

        index = int(ts // self.window)
        self._rotate(index)
        self._cur += value

    def query(self, start, end):
        w0 = int(start // self.window)
        w1 = int(end // self.window)
        total = 0.0
        for index in range(w0, w1 + 1):
            if index == self._cur_index:
                raw = self._cur
            elif index == self._cur_index - 1:
                raw = self._prev
            else:
                raw = 0.0
            lo = index * self.window
            hi = lo + self.window
            overlap = max(0.0, min(hi, end) - max(lo, start)) / self.window
            total += raw * overlap
        return total

    def value_at(self):
        return self._total

    def snapshot(self):
        return {
            "half_life": self.half_life,
            "window": self.window,
            "prev": self._prev,
            "cur": self._cur,
            "cur_index": self._cur_index,
            "total": self._total,
        }

    @classmethod
    def restore(cls, snapshot):
        counter = cls(snapshot["half_life"], snapshot["window"])
        counter._prev = snapshot["prev"]
        counter._cur = snapshot["cur"]
        counter._cur_index = snapshot["cur_index"]
        counter._total = snapshot["total"]
        return counter
