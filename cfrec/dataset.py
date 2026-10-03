"""Deterministic synthetic sparse-rating dataset (standard library only).

Simulates a long-tail, category-driven domain (e.g. video/music):
* Zipf-ish item popularity -> a few items absorb most interactions.
* Users have per-category affinities and a few liked tags.
* 25% heavy users generate most ratings, the rest contribute very few
  (realistic power-law sparsity and near-zero rows).
* make_split holds out brand-new users (0 train ratings), brand-new items
  (0 train ratings), semi-cold users (1-3 train ratings) and warm users.
"""

import math
import random

from .data import Item, ItemCatalog, Ratings, Split

N_CATEGORIES = 8
TAGS_PER_CATEGORY = 5
GLOBAL_TAGS = 4
N_ITEMS = 200
N_USERS = 300
N_COLD_USERS = 30
N_COLD_ITEMS = 20
N_SEMI_COLD_USERS = 40


def _beta(rng, a, b):
    # Sanyal-style: ratio of two gamma(1/scale) draws.
    x = rng.gammavariate(a, 1.0)
    y = rng.gammavariate(b, 1.0)
    return x / (x + y)


def generate(seed=20261003):
    rng = random.Random(seed)
    catalog = ItemCatalog()
    categories = ["cat%d" % i for i in range(N_CATEGORIES)]
    tags_by_cat = {c: ["%s_t%d" % (c, t) for t in range(TAGS_PER_CATEGORY)]
                   for c in categories}
    global_tags = ["g%d" % t for t in range(GLOBAL_TAGS)]

    item_quality = {}
    item_category = {}
    for idx in range(N_ITEMS):
        item_id = "item%03d" % idx
        category = categories[idx % N_CATEGORIES]
        item_category[item_id] = category
        n_tags = rng.randint(2, 4)
        features = set(rng.sample(tags_by_cat[category],
                                  min(n_tags, TAGS_PER_CATEGORY)))
        if rng.random() < 0.6:
            features.add(rng.choice(global_tags))
        catalog.add(Item(item_id, category, features))
        item_quality[item_id] = rng.gauss(0.0, 0.30)

    # Zipf-ish global popularity weights.
    pop_weight = {}
    for rank, item_id in enumerate(sorted(catalog.ids())):
        pop_weight[item_id] = 1.0 / ((rank + 1) ** 0.9)

    users = ["user%03d" % i for i in range(N_USERS)]
    affinity = {}
    liked_tags = {}
    user_bias = {}
    heavy = set(rng.sample(users, int(N_USERS * 0.25)))
    for user_id in users:
        # Each user has 2-3 favorite categories (strong taste clusters) and
        # likes tags mostly drawn from those categories.
        # sorted(): set iteration order is process-dependent and would make
        # the generated dataset non-deterministic across runs.
        favorites = sorted(rng.sample(categories, rng.randint(2, 3)))
        affinity[user_id] = {}
        for c in categories:
            affinity[user_id][c] = (_beta(rng, 5.0, 2.0) if c in favorites
                                    else _beta(rng, 1.5, 4.0))
        fav_pool = [t for c in favorites for t in tags_by_cat[c]]
        n_liked = rng.randint(3, 5)
        liked = set(rng.sample(fav_pool, min(n_liked, len(fav_pool))))
        if rng.random() < 0.5:
            liked.add(rng.choice(global_tags))
        liked_tags[user_id] = liked
        user_bias[user_id] = rng.gauss(0.0, 0.4)

    raw = {u: [] for u in users}
    for user_id in users:
        n_inter = rng.gauss(38, 8) if user_id in heavy else rng.gauss(7.0, 3.0)
        n_inter = max(1, int(n_inter))
        weights = {}
        for item_id, w in pop_weight.items():
            aff = affinity[user_id][item_category[item_id]]
            weights[item_id] = (w ** 0.5) * (0.10 + 2.5 * aff)
        picked = []
        pool = list(weights)
        for _ in range(min(n_inter, len(pool))):
            total = sum(weights[i] for i in pool)
            r = rng.random() * total
            acc = 0.0
            for idx, item_id in enumerate(pool):
                acc += weights[item_id]
                if r <= acc:
                    picked.append(item_id)
                    pool.pop(idx)
                    break
        for item_id in picked:
            category = item_category[item_id]
            item = catalog.get(item_id)
            tag_hits = len(item.features & liked_tags[user_id])
            tag_term = 0.45 * tag_hits / max(len(item.features), 1)
            score = (3.0
                     + 2.0 * (affinity[user_id][category] - 0.5)
                     + item_quality[item_id]
                     + tag_term
                     + user_bias[user_id]
                     + rng.gauss(0.0, 0.25))
            score = max(1.0, min(5.0, score))
            rating = round(score * 2.0) / 2.0
            raw[user_id].append((item_id, rating))

    return catalog, raw, heavy


def make_split(seed=20261003):
    catalog, raw, _ = generate(seed)
    rng = random.Random(seed ^ 0x5)
    all_users = [u for u in raw if raw[u]]
    all_items = catalog.ids()
    cold_users = set(rng.sample(all_users, N_COLD_USERS))
    cold_items = set(rng.sample(all_items, N_COLD_ITEMS))

    train = Ratings()
    warm_test = {}
    semi_cold_test = {}
    cold_user_test = {}
    cold_item_test = {}

    semi_pool = [u for u in all_users if u not in cold_users
                 and len([1 for i, _ in raw[u] if i not in cold_items]) >= 4]
    semi_cold_users = set(rng.sample(semi_pool, N_SEMI_COLD_USERS))

    for user_id, pairs in raw.items():
        pairs = list(pairs)
        rng.shuffle(pairs)
        if user_id in cold_users:
            cold_user_test[user_id] = dict(pairs)
            continue
        cold_item_pairs = [(i, r) for i, r in pairs if i in cold_items]
        eligible = [(i, r) for i, r in pairs if i not in cold_items]
        if cold_item_pairs:
            cold_item_test[user_id] = dict(cold_item_pairs)
        if not eligible:
            continue
        if user_id in semi_cold_users:
            n_train = min(3, len(eligible) - 1)
            for item_id, rating in eligible[:n_train]:
                train.add(user_id, item_id, rating)
            semi_cold_test[user_id] = dict(eligible[n_train:])
        else:
            n_test = max(1, int(round(len(eligible) * 0.25)))
            test_pairs = eligible[:n_test]
            for item_id, rating in eligible[n_test:]:
                train.add(user_id, item_id, rating)
            warm_test[user_id] = dict(test_pairs)

    split = Split(train, warm_test, semi_cold_test, cold_user_test,
                  cold_item_test, cold_users, cold_items)
    split.catalog = catalog
    return split
