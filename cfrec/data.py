"""Data containers: item catalog, rating matrix (sparse dict-of-dicts), split."""

from collections import defaultdict


class Item(object):
    """An item with a category and binary content features (tags)."""

    __slots__ = ("item_id", "category", "features")

    def __init__(self, item_id, category, features=()):
        self.item_id = item_id
        self.category = category
        self.features = frozenset(features)

    def __repr__(self):
        return "Item(%r, %r, %r)" % (self.item_id, self.category, sorted(self.features))


class ItemCatalog(object):
    """Registry of items; also exposes the feature vocabulary."""

    def __init__(self):
        self._items = {}
        self.feature_vocabulary = set()

    def add(self, item):
        if item.item_id in self._items:
            raise ValueError("duplicate item id: %r" % (item.item_id,))
        self._items[item.item_id] = item
        self.feature_vocabulary |= item.features

    def get(self, item_id):
        return self._items.get(item_id)

    def ids(self):
        return list(self._items.keys())

    def __len__(self):
        return len(self._items)

    def __contains__(self, item_id):
        return item_id in self._items

    def __iter__(self):
        return iter(self._items.values())


class Ratings(object):
    """Sparse rating matrix stored as dict-of-dicts.

    Missing values are simply absent entries; every downstream computation
    only ever iterates over observed entries (see similarity.py).
    """

    def __init__(self):
        self.by_user = defaultdict(dict)
        self.by_item = defaultdict(dict)
        self.min_rating = float("inf")
        self.max_rating = float("-inf")

    def add(self, user_id, item_id, rating):
        rating = float(rating)
        self.by_user[user_id][item_id] = rating
        self.by_item[item_id][user_id] = rating
        self.min_rating = min(self.min_rating, rating)
        self.max_rating = max(self.max_rating, rating)

    def __len__(self):
        return sum(len(v) for v in self.by_user.values())

    @property
    def n_users(self):
        return len(self.by_user)

    @property
    def n_items(self):
        return len(self.by_item)

    def density(self):
        if self.n_users == 0 or self.n_items == 0:
            return 0.0
        return len(self) / float(self.n_users * self.n_items)

    def user_ratings(self, user_id):
        return self.by_user.get(user_id, {})

    def item_ratings(self, item_id):
        return self.by_item.get(item_id, {})

    def get(self, user_id, item_id):
        return self.by_user.get(user_id, {}).get(item_id)

    def global_mean(self, default=3.0):
        total = 0.0
        count = 0
        for items in self.by_user.values():
            for rating in items.values():
                total += rating
                count += 1
        return total / count if count else default

    def user_mean(self, user_id, default=None):
        items = self.by_user.get(user_id)
        if not items:
            return self.global_mean() if default is None else default
        return sum(items.values()) / len(items)

    def item_mean(self, item_id, default=None):
        users = self.by_item.get(item_id)
        if not users:
            return self.global_mean() if default is None else default
        return sum(users.values()) / len(users)

    def item_popularity(self):
        return {item_id: len(users) for item_id, users in self.by_item.items()}


class Split(object):
    """Container for the evaluation split produced by dataset.make_split."""

    def __init__(self, train, warm_test, semi_cold_test, cold_user_test,
                 cold_item_test, cold_users, cold_items):
        self.train = train
        self.warm_test = warm_test                # {user: {item: rating}}
        self.semi_cold_test = semi_cold_test      # {user: {item: rating}}
        self.cold_user_test = cold_user_test      # {user: {item: rating}}
        self.cold_item_test = cold_item_test      # {user: {item: rating}}
        self.cold_users = cold_users
        self.cold_items = cold_items
