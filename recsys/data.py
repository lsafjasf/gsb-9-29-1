"""Synthetic sparse rating data with item content features.

The data mimics a MovieLens-style domain:

* each item has one primary category and a set of content tags;
* each user has a small set of preferred categories (taste);
* a rating is generated only when item affinity crosses a random threshold,
  which keeps the matrix extremely sparse instead of a dense random grid;
* a few users and items are held out as *cold* entities: part of their
  interactions go to the test set so offline metrics can be measured for
  new-user / new-item scenarios.

Everything is deterministic given ``seed``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

Rating = Tuple[int, int, float]  # (user_id, item_id, rating)


@dataclass
class Dataset:
    num_users: int
    num_items: int
    categories: List[str]
    item_categories: Dict[int, str]
    item_tags: Dict[int, List[str]]
    train: List[Rating]
    test: List[Rating]
    cold_users: Set[int]
    cold_items: Set[int]

    def train_matrix(self) -> Dict[int, Dict[int, float]]:
        matrix: Dict[int, Dict[int, float]] = {}
        for user, item, rating in self.train:
            matrix.setdefault(user, {})[item] = rating
        return matrix

    def test_matrix(self) -> Dict[int, Dict[int, float]]:
        matrix: Dict[int, Dict[int, float]] = {}
        for user, item, rating in self.test:
            matrix.setdefault(user, {})[item] = rating
        return matrix

    @property
    def density(self) -> float:
        cells = self.num_users * self.num_items
        return len(self.train) / cells if cells else 0.0


CATEGORIES = [
    "action", "comedy", "drama", "documentary", "horror",
    "sci_fi", "romance", "kids", "music", "thriller",
]

TAG_POOL = [
    "classic", "foreign", "indie", "blockbuster", "slow_paced",
    "dark", "uplifting", "visual", "dialogue", "true_story",
    "franchise", "award", "short", "ensemble", "minimalist",
]


def generate_dataset(
    num_users: int = 300,
    num_items: int = 200,
    num_cold_users: int = 20,
    num_cold_items: int = 20,
    seed: int = 42,
) -> Dataset:
    """Generate train/test ratings.

    Cold users/items keep at most one interaction in training (often zero),
    while roughly half of their ratings go to the test set.
    """

    rng = random.Random(seed)

    # Zipf-ish category mix: a few head categories, a long tail.
    weights = [1.0 / (rank + 1) for rank in range(len(CATEGORIES))]
    item_categories: Dict[int, str] = {}
    item_tags: Dict[int, List[str]] = {}
    for item in range(num_items):
        category = rng.choices(CATEGORIES, weights=weights, k=1)[0]
        item_categories[item] = category
        tag_count = rng.randint(1, 3)
        item_tags[item] = rng.sample(TAG_POOL, tag_count)

    cold_users = set(range(num_users - num_cold_users, num_users))
    cold_items = set(range(num_items - num_cold_items, num_items))

    # Per-user taste: 1-3 preferred categories.
    tastes: Dict[int, Set[str]] = {}
    user_bias: Dict[int, float] = {}
    for user in range(num_users):
        taste_size = rng.randint(1, 3)
        tastes[user] = set(rng.sample(CATEGORIES, taste_size))
        user_bias[user] = rng.uniform(-0.4, 0.4)

    train: List[Rating] = []
    test: List[Rating] = []

    for user in range(num_users):
        for item in range(num_items):
            category = item_categories[item]
            in_taste = category in tastes[user]

            # Sparse observation process: in-taste items are far more likely
            # to be rated, but most cells stay empty.
            observe_prob = 0.16 if in_taste else 0.008
            if item in cold_items:
                observe_prob *= 0.7
            if user in cold_users:
                observe_prob *= 0.35
            if rng.random() > observe_prob:
                continue

            # Rating process: taste match + tag noise + user bias.
            affinity = 1.3 if in_taste else -0.9
            affinity += rng.gauss(0.0, 0.75) + user_bias[user]
            rating = 3.0 + affinity
            rating = max(1.0, min(5.0, round(rating * 2) / 2.0))  # half-star grid

            cold_entity = user in cold_users or item in cold_items
            to_test = rng.random() < (0.6 if cold_entity else 0.2)
            if cold_entity and not to_test:
                # Keep at most one training interaction for cold entities.
                existing = sum(1 for u, i, _ in train if u == user or i == item)
                if existing >= 1:
                    to_test = True
            (test if to_test else train).append((user, item, rating))

    # Guarantee every cold user/item has at least one test interaction.
    def has_rating(rows: List[Rating], user: int | None = None, item: int | None = None) -> bool:
        return any(
            (user is None or u == user) and (item is None or i == item)
            for u, i, _ in rows
        )

    for user in list(cold_users):
        if not has_rating(test, user=user):
            item = rng.randrange(num_items)
            rating = rng.choice([3.5, 4.0, 4.5, 5.0])
            test.append((user, item, rating))
    for item in list(cold_items):
        if not has_rating(test, item=item):
            user = rng.randrange(num_users)
            rating = rng.choice([3.5, 4.0, 4.5, 5.0])
            test.append((user, item, rating))

    return Dataset(
        num_users=num_users,
        num_items=num_items,
        categories=list(CATEGORIES),
        item_categories=item_categories,
        item_tags=item_tags,
        train=train,
        test=test,
        cold_users=cold_users,
        cold_items=cold_items,
    )
