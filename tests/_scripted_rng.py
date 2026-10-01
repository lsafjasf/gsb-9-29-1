"""Injectable deterministic RNG for unit tests: replays scripted values."""
import random


class ScriptedRNG:
    def __init__(self, values=(), seed=0):
        self._values = list(values)
        self._i = 0
        self._fallback = random.Random(seed)

    def random(self):
        if self._i < len(self._values):
            v = self._values[self._i]
            self._i += 1
            return v
        return self._fallback.random()

    def randrange(self, n):
        if self._i < len(self._values):
            v = self._values[self._i]
            self._i += 1
            return int(v * n) % n
        return self._fallback.randrange(n)

    def sample(self, population, k):
        pop = list(population)
        if k == 0:
            return []
        if k > len(pop):
            raise ValueError("sample larger than population")
        return self._fallback.sample(pop, k)
