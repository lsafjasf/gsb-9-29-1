"""Edge-case and correctness tests for lsh.py. Run: python3 test_lsh.py"""

import random
import unittest

from lsh import LSHIndex, cosine_similarity


def random_unit_vector(rng, dim):
    vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def noisy_copy(rng, vec, rel_noise):
    # rel_noise is the noise-to-signal ratio: per-coordinate sigma is scaled
    # by 1/sqrt(dim) so that ||noise|| / ||vec|| ~= rel_noise.
    sigma = rel_noise / len(vec) ** 0.5
    out = [x + rng.gauss(0.0, sigma) for x in vec]
    norm = sum(x * x for x in out) ** 0.5
    return [x / norm for x in out]


class TestEmptyIndex(unittest.TestCase):
    def test_query_on_empty_index(self):
        index = LSHIndex()
        self.assertEqual(len(index), 0)
        self.assertEqual(index.query([1.0, 0.0, 0.0]), set())
        self.assertEqual(index.query_topk([1.0, 0.0, 0.0]), [])

    def test_remove_missing_key_raises(self):
        index = LSHIndex()
        with self.assertRaises(KeyError):
            index.remove("nope")
        index.discard("nope")  # must be a no-op

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            LSHIndex(num_hashes=0)
        with self.assertRaises(ValueError):
            LSHIndex(num_tables=0)


class TestSinglePoint(unittest.TestCase):
    def test_single_point_found(self):
        index = LSHIndex(num_hashes=6, num_tables=8, seed=1)
        vec = random_unit_vector(random.Random(7), 64)
        index.add("only", vec)
        self.assertEqual(len(index), 1)
        self.assertEqual(index.query(vec), {"only"})
        top = index.query_topk(vec, k=5)
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0][1], "only")
        self.assertAlmostEqual(top[0][0], 1.0, places=9)

    def test_delete_only_point_returns_to_empty(self):
        index = LSHIndex()
        index.add("a", [1.0, 2.0, 3.0])
        index.remove("a")
        self.assertEqual(len(index), 0)
        self.assertEqual(index.query([1.0, 2.0, 3.0]), set())
        self.assertEqual(index.num_buckets(), 0)


class TestAllIdentical(unittest.TestCase):
    def test_identical_vectors_always_collide(self):
        index = LSHIndex(num_hashes=10, num_tables=12, seed=3)
        vec = random_unit_vector(random.Random(11), 100)
        for i in range(50):
            index.add(i, vec)
        # All identical vectors land in one bucket per table.
        self.assertEqual(index.num_buckets(), index.num_tables)
        self.assertEqual(index.query(vec), set(range(50)))
        # Deleting some keeps the rest findable.
        for i in range(25):
            index.remove(i)
        self.assertEqual(index.query(vec), set(range(25, 50)))


class TestDynamicUpdates(unittest.TestCase):
    def test_insert_delete_reinsert(self):
        rng = random.Random(5)
        index = LSHIndex(num_hashes=6, num_tables=10, seed=9)
        vecs = {i: random_unit_vector(rng, 32) for i in range(100)}
        for i, v in vecs.items():
            index.add(i, v)
        self.assertEqual(len(index), 100)
        stats = index.memory_stats()
        self.assertEqual(stats["bucket_memberships"], 100 * index.num_tables)

        for i in range(0, 100, 2):
            index.remove(i)
        self.assertEqual(len(index), 50)
        for i in range(0, 100, 2):
            self.assertNotIn(i, index)
            self.assertNotIn(i, index.query(vecs[i]))
        stats = index.memory_stats()
        self.assertEqual(stats["bucket_memberships"], 50 * index.num_tables)

        # Re-adding a key with a new vector replaces the old signature.
        v_old = random_unit_vector(rng, 32)
        v_new = random_unit_vector(rng, 32)
        index.add("x", v_old)
        index.add("x", v_new)
        self.assertEqual(len(index), 51)
        self.assertEqual(index.memory_stats()["bucket_memberships"], 51 * index.num_tables)
        self.assertIn("x", index.query(v_new))

    def test_empty_buckets_are_freed(self):
        index = LSHIndex(num_hashes=4, num_tables=6, seed=2)
        index.add("a", [1.0, 0.0])
        peak = index.num_buckets()
        self.assertEqual(peak, 6)
        index.remove("a")
        self.assertEqual(index.num_buckets(), 0)


class TestHighDimSparse(unittest.TestCase):
    def test_sparse_million_dim(self):
        dim = 1_000_000
        rng = random.Random(13)
        index = LSHIndex(num_hashes=8, num_tables=12, seed=17)

        def sparse_vec(seed_rng, support):
            return {d: seed_rng.gauss(0.0, 1.0) for d in support}

        shared_support = rng.sample(range(dim), 60)
        base = sparse_vec(rng, shared_support)
        # Near-duplicate: same support, slightly perturbed values.
        dup = {d: v + rng.gauss(0.0, 0.05) for d, v in base.items()}
        # Distractors on disjoint supports.
        for i in range(30):
            support = rng.sample(range(dim), 60)
            index.add(f"noise-{i}", sparse_vec(rng, support))
        index.add("base", base)
        index.add("dup", dup)

        self.assertGreater(cosine_similarity(base, dup), 0.99)
        candidates = index.query(base)
        self.assertIn("base", candidates)
        self.assertIn("dup", candidates)
        top = index.query_topk(base, k=2)
        self.assertEqual({key for _, key in top}, {"base", "dup"})

        # Weight cache only materializes dimensions actually seen, and stays
        # far below num_tables * num_hashes * dim.
        stats = index.memory_stats()
        self.assertLess(stats["weight_cache_entries"], 12 * 8 * 2000)

    def test_zero_vector(self):
        index = LSHIndex()
        index.add("zero", {})
        index.add("dense-zero", [0.0, 0.0, 0.0])
        self.assertEqual(len(index), 2)
        self.assertIsInstance(index.query({}), set)
        self.assertEqual(cosine_similarity({}, {1: 1.0}), 0.0)


class TestRecallSanity(unittest.TestCase):
    def test_recall_on_clustered_data(self):
        rng = random.Random(23)
        dim = 64
        index = LSHIndex(num_hashes=8, num_tables=16, seed=29)
        bases = [random_unit_vector(rng, dim) for _ in range(20)]
        items = {}
        for b, base in enumerate(bases):
            for c in range(5):
                vec = noisy_copy(rng, base, 0.15)
                items[(b, c)] = vec
                index.add((b, c), vec)
        # For a fresh noisy copy of base 0, the other copies of base 0 are
        # the true neighbours (cosine ~0.98); they must be candidates.
        query = noisy_copy(rng, bases[0], 0.15)
        truth = {(0, c) for c in range(5)}
        candidates = index.query(query)
        self.assertEqual(len(truth & candidates), len(truth))


if __name__ == "__main__":
    unittest.main(verbosity=2)
