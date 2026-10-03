#!/usr/bin/env python3
"""Minimal walkthrough: one warm, one semi-cold and one cold user.

Run:  python3 demo.py
"""

from benchmark import build_models
from cfrec.dataset import make_split


def show(title, user_id, split, models):
    print("\n=== %s  user=%s ===" % (title, user_id))
    history = split.train.user_ratings(user_id)
    print("train history (%d ratings): %s"
          % (len(history),
             sorted(history.items(), key=lambda kv: -kv[1])[:4] or "(empty)"))
    for model in models:
        recs = model.recommend(user_id, k=5)
        items = ", ".join(item_id for item_id, _ in recs) or "(no recs)"
        print("  %-13s %s" % (model.name, items))


def main():
    split = make_split()
    models = build_models(split.train, split.catalog)
    print("catalog: %d items / %d categories; train density %.2f%%"
          % (len(split.catalog),
             len({i.category for i in split.catalog}),
             split.train.density() * 100))

    warm = max(split.warm_test,
               key=lambda u: len(split.train.user_ratings(u)))
    semi = next(iter(split.semi_cold_test))
    cold = next(iter(split.cold_user_test))
    show("WARM USER (many ratings)", warm, split, models)
    show("SEMI-COLD USER (1-3 ratings)", semi, split, models)
    show("COLD USER (0 ratings)", cold, split, models)

    # Rating prediction comparison for one held-out pair.
    user_id = warm
    item_id, true_rating = next(iter(split.warm_test[user_id].items()))
    print("\n=== rating prediction on held-out pair: %s -> %s (true=%.1f) ==="
          % (user_id, item_id, true_rating))
    for model in models:
        print("  %-13s pred=%.2f" % (model.name, model.predict(user_id, item_id)))


if __name__ == "__main__":
    main()
