"""Shared random-scenario generator for differential and restart tests."""

import random


def build_scenario(seed, kind):
    rng = random.Random(seed)
    half_life = rng.choice([0.5, 1.0, 3.0, 10.0, 40.0])
    bucket_width = rng.choice([0.25, 0.5, 1.0, 3.0, 7.5])

    if kind == "uniform":
        span = 60.0
        count = 400
        max_lateness = rng.choice([0.0, 2.0, span])
        stamps = [rng.uniform(0.0, span) for _ in range(count)]
    elif kind == "dense":
        span = 2.0
        count = 800
        max_lateness = rng.choice([0.0, 0.5, 10.0])
        stamps = [rng.uniform(0.0, span) for _ in range(count)]
    elif kind == "boundary":
        span = 50.0
        count = 300
        max_lateness = rng.choice([0.0, 1.0, 50.0])
        stamps = []
        for i in range(count):
            bucket_index = rng.randint(0, int(span / bucket_width))
            edge = bucket_index * bucket_width
            offset = rng.choice([0.0, 0.0, bucket_width * 0.5,
                                 bucket_width - 1e-9, bucket_width])
            stamps.append(edge + offset)
    elif kind == "same_time":
        span = 40.0
        count = 300
        max_lateness = rng.choice([0.0, 3.0, 40.0])
        stamps = []
        for _ in range(count // 5):
            common = rng.uniform(0.0, span)
            stamps.extend([common] * 5)
    elif kind == "reversed":
        span = 30.0
        count = 200
        max_lateness = span + 1.0
        stamps = sorted(rng.uniform(0.0, span) for _ in range(count))
        stamps.reverse()
    else:
        raise ValueError(kind)

    records = []
    for i, ts in enumerate(stamps):
        value = rng.choice([0.5, 1.0, 1.0, 1.0, 2.0])
        records.append(("e%d" % i, ts, value))

    duplicate_count = 0
    if kind != "reversed" and rng.random() < 0.8:
        duplicate_count = count // 10
        for _ in range(duplicate_count):
            i = rng.randrange(count)
            records.append(("e%d" % i, stamps[i] + rng.uniform(-0.01, 0.01),
                            rng.choice([0.5, 1.0, 3.0])))

    rng.shuffle(records)
    if kind == "reversed":
        records.sort(key=lambda record: -record[1])

    query_times = [0.0]
    query_times += [rng.uniform(0.0, span) for _ in range(20)]
    query_times += sorted(rng.choice(stamps) for _ in range(10))
    query_times += [span, span + 10.0, span + 1000.0]
    query_edges = sorted({round(t, 6) for t in query_times})
    queries = []
    for end in query_edges:
        start = rng.choice([0.0, end, max(0.0, end - rng.uniform(0.0, span)),
                            -10.0, end - span])
        if start <= end:
            queries.append((start, end))
    for edge in [k * bucket_width for k in range(int(span / bucket_width) + 1)]:
        queries.append((edge, edge))
        if edge + bucket_width <= span + 1.0:
            queries.append((edge, edge + bucket_width))

    value_times = sorted({round(t, 6) for t in query_edges})

    return {
        "kind": kind,
        "seed": seed,
        "half_life": half_life,
        "bucket_width": bucket_width,
        "max_lateness": max_lateness,
        "records": records,
        "queries": queries,
        "value_times": value_times,
        "duplicates_injected": duplicate_count,
    }


KINDS = ("uniform", "dense", "boundary", "same_time", "reversed")
