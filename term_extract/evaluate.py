"""Evaluation against human annotation.

Gold file format (one concept per line):
    canonical_form<TAB>surface1|surface2|surface3
The TAB section is optional; without it the line itself is the only surface.

Metrics
-------
precision / recall / f1 : computed at the *concept* level -- a system term
    counts as correct if its canonical key matches any gold surface's
    canonical key.
redundancy_rate         : share of extracted items that duplicate an
    already-extracted concept (1 - unique_concepts / extracted_items).
    Computed both before merging (raw candidates) and after.
"""
from .merging import canonical_key


def load_gold(path):
    """Return (concept_keys, surface_to_concept)."""
    concepts, surface_map = [], {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                canon, surfaces = line.split("\t", 1)
                surfaces = surfaces.split("|")
            else:
                canon, surfaces = line, [line]
            ckey = canonical_key(canon)
            concepts.append(ckey)
            surface_map[canonical_key(canon)] = ckey
            for s in surfaces:
                surface_map[canonical_key(s)] = ckey
    return concepts, surface_map


def concept_keys_of(system_terms):
    """Map system output (list of term dicts) to canonical concept keys."""
    keys = []
    for t in system_terms:
        keys.append(canonical_key(t["term"]))
    return keys


def evaluate(system_terms, gold_path, raw_candidate_count=None):
    concepts, surface_map = load_gold(gold_path)
    gold_set = set(concepts)

    sys_keys = concept_keys_of(system_terms)
    # resolve each system term to a gold concept via any of its variants
    resolved = set()
    for t in system_terms:
        for surface in [t["term"]] + t.get("variants", []):
            ck = canonical_key(surface)
            if ck in surface_map:
                resolved.add(surface_map[ck])
                break

    n_sys = len(system_terms)
    precision = len(resolved) / n_sys if n_sys else 0.0
    recall = len(resolved & gold_set) / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)

    unique_concepts = len(set(sys_keys))
    redundancy = 1 - unique_concepts / n_sys if n_sys else 0.0

    result = {
        "extracted": n_sys,
        "gold_concepts": len(gold_set),
        "matched_concepts": len(resolved & gold_set),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "redundancy_rate_after_merge": round(redundancy, 4),
    }
    if raw_candidate_count:
        result["raw_candidates"] = raw_candidate_count
        result["redundancy_removed_by_merge"] = round(
            1 - n_sys / raw_candidate_count, 4) if raw_candidate_count else 0.0
    return result
