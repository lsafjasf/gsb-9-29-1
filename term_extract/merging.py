"""Redundancy merging.

Three merge families, each recording an explicit, human-readable reason:

  * case variants        -- "BERT" / "Bert" / "bert"
  * inflection variants  -- English plural/singular ("neural networks" /
                            "neural network"), hyphen/space variants
  * abbreviation pairs   -- full form <-> acronym, detected either from
                            explicit "Long (Short)" / "Short (Long)" patterns
                            in the text (incl. Chinese parentheses) or from
                            an acronym letter match between two candidates
"""
import re
from collections import Counter

from .tokenizer import is_cjk

# ---------------------------------------------------------------------------
# canonicalization
# ---------------------------------------------------------------------------

def singularize(word):
    """Conservative English singularization."""
    if len(word) <= 3 or not word.isalpha():
        return word
    low = word.lower()
    if low.endswith(("ss", "us", "is")):
        return low
    if low.endswith("ies") and len(low) > 4:
        return low[:-3] + "y"
    if low.endswith(("ches", "shes", "xes", "zes", "sses")):
        return low[:-2]
    if low.endswith("ses") and not low.endswith("sses"):
        return low[:-1]
    if low.endswith("s"):
        return low[:-1]
    return low


def canonical_key(text):
    """Case- and inflection-insensitive key: lowercase, singularize each
    English token, normalize hyphens to spaces, drop spaces around CJK."""
    text = text.strip().lower().replace("-", " ").replace("_", " ")
    tokens = text.split()
    out = []
    for tok in tokens:
        if all(is_cjk(c) for c in tok):
            out.append(tok)
        else:
            out.append(singularize(tok))
    return " ".join(out)


# ---------------------------------------------------------------------------
# abbreviation detection
# ---------------------------------------------------------------------------

# "Natural Language Processing (NLP)" / "natural language processing (NLP)"
RE_LONG_SHORT = re.compile(
    r"([A-Za-z][A-Za-z0-9]*(?:[ \-][A-Za-z0-9]+){1,6})\s*[（(]\s*([A-Z][A-Za-z0-9]{1,11})\s*[)）]")
# "NLP (natural language processing)"
RE_SHORT_LONG = re.compile(
    r"\b([A-Z][A-Z0-9]{1,11})\s*[（(]\s*([A-Za-z][A-Za-z0-9 \-]{2,60}?)\s*[)）]")
# "卷积神经网络（Convolutional Neural Network）" / "（Convolutional Neural Network, CNN）"
RE_ZH_EN = re.compile(
    r"([一-鿿]{2,12})\s*[（(]\s*([A-Za-z][A-Za-z0-9 ]{1,50}?)(?:\s*[,，]\s*([A-Z][A-Za-z0-9]{1,11}))?\s*[)）]")


def find_abbrev_pairs(text):
    """Scan raw text for explicit abbreviation definitions.

    Returns list of (long_form, short_form, evidence)."""
    pairs = []
    for m in RE_LONG_SHORT.finditer(text):
        long, short = m.group(1).strip(), m.group(2).strip()
        if len(short) >= 2 and _acronym_ok(long, short):
            pairs.append((long, short, f"pattern 'Long (Abbr)' at offset {m.start()}"))
    for m in RE_SHORT_LONG.finditer(text):
        short, long = m.group(1).strip(), m.group(2).strip()
        if _acronym_ok(long, short):
            pairs.append((long, short, f"pattern 'Abbr (Long)' at offset {m.start()}"))
    for m in RE_ZH_EN.finditer(text):
        zh, en, abbr = m.group(1), m.group(2).strip(), m.group(3)
        pairs.append((zh, en, f"pattern '中文 (English)' at offset {m.start()}"))
        if abbr:
            pairs.append((en, abbr, f"pattern '(English, ABBR)' at offset {m.start()}"))
            pairs.append((zh, abbr, f"pattern '中文 (…, ABBR)' at offset {m.start()}"))
    return pairs


def _acronym_ok(long, short):
    letters = [w[0] for w in re.split(r"[ \-]+", long) if w and w[0].isalpha()]
    acro = "".join(letters).lower()
    s = short.lower().replace("-", "")
    # full acronym, or acronym with skipped words (e.g. "SVM" from
    # "support vector machine"): short must be a subsequence of letters
    if acro == s:
        return True
    it = iter(letters)
    return all(any(c == ch for c in it) for ch in s) and len(s) >= 2


def acronym_of(phrase):
    return "".join(w[0] for w in re.split(r"[ \-]+", phrase)
                   if w and w[0].isalpha()).lower()


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------

class TermGroup:
    """A merged, non-redundant term with its variants and merge log."""

    def __init__(self, cand):
        self.candidates = [cand]          # Candidate objects
        self.variants = Counter()         # surface -> freq
        self.reasons = []                 # human-readable merge evidence
        self.has_abbrev_pair = False
        self.forced_representative = None
        self.representative = None
        self.score = 0.0
        self.freq = 0
        self._absorb(cand)
        self._note_case_variants(cand)

    def _absorb(self, cand):
        for surface, n in cand.surfaces.items():
            self.variants[surface] += n

    def _note_case_variants(self, cand):
        """Case variants collapse at candidate level already; record the
        evidence so the merge log stays complete."""
        lowers = {}
        for surface in cand.surfaces:
            lowers.setdefault(surface.lower(), set()).add(surface)
        for forms in lowers.values():
            if len(forms) > 1:
                self.reasons.append(
                    "case_variant: surfaces "
                    + ", ".join(f"'{f}'" for f in sorted(forms))
                    + " share the same lowercase form")

    def absorb(self, other):
        self.candidates.extend(other.candidates)
        for surface, n in other.variants.items():
            self.variants[surface] += n
        self.reasons.extend(other.reasons)
        self.has_abbrev_pair = self.has_abbrev_pair or other.has_abbrev_pair

    def finalize(self):
        self.score = max(c.score for c in self.candidates)
        self.freq = sum(c.freq for c in self.candidates)
        if self.forced_representative:
            self.representative = self.forced_representative
            return self
        # representative: when an abbreviation pair was merged, display
        # the long (multi-token) form; otherwise the highest-scoring one
        if self.has_abbrev_pair:
            best = max(self.candidates,
                       key=lambda c: (len(c.tokens), c.score, c.freq))
        else:
            best = max(self.candidates, key=lambda c: (c.score, c.freq))
        self.representative = best.surface()
        return self

    def as_dict(self):
        return {
            "term": self.representative,
            "score": round(self.score, 4),
            "freq": self.freq,
            "variants": [s for s, _ in self.variants.most_common()],
            "merge_reasons": self.reasons,
            "complete": all(c.complete for c in self.candidates),
        }


def _variant_reason(surfaces_a, surfaces_b):
    la, lb = surfaces_a.lower(), surfaces_b.lower()
    if la == lb:
        return "case_variant"
    if la.replace("-", " ") == lb.replace("-", " "):
        return "hyphenation_variant"
    return "inflection_variant"   # singular/plural etc.


def merge_candidates(ranked, raw_text=""):
    """Merge ranked Candidate list into TermGroups.

    Returns (groups, merge_log) where merge_log lists every merge action
    with its justification -- this is the '归并依据'.
    """
    # ---- pass 1: canonical key grouping (case / plural / hyphen) --------
    groups = {}
    order = []
    for cand in ranked:
        key = canonical_key(cand.surface())
        if key in groups:
            g = groups[key]
            reason = _variant_reason(g.candidates[0].surface(), cand.surface())
            g.reasons.append(
                f"{reason}: '{cand.surface()}' merged into "
                f"'{g.candidates[0].surface()}' (shared canonical form '{key}')")
            g._absorb(cand)
            g.candidates.append(cand)
        else:
            groups[key] = TermGroup(cand)
            order.append(key)

    # ---- pass 2: abbreviation pairs -------------------------------------
    key_of_group = {k: k for k in order}

    def find_group(surface):
        return groups.get(canonical_key(surface))

    def merge_groups(g_long, g_short, why, merge_log):
        if g_long is g_short:
            return
        # keep the long form as the surviving group
        g_long.absorb(g_short)
        g_long.reasons.append(why)
        merge_log.append(why)
        for k, g in list(groups.items()):
            if g is g_short:
                del groups[k]

    merge_log = []
    for long, short, evidence in find_abbrev_pairs(raw_text):
        g_long, g_short = find_group(long), find_group(short)
        if g_long and g_short and g_long is not g_short:
            why = (f"abbreviation: '{short}' is the short form of '{long}' "
                   f"({evidence}); merged '{g_short.representative or short}' "
                   f"into '{g_long.representative or long}'")
            g_long.has_abbrev_pair = True
            merge_groups(g_long, g_short, why, merge_log)
        elif g_long or g_short:
            # one side fell below min_freq: keep the pair as variants of
            # the surviving group so the link is not lost
            g = g_long or g_short
            missing = short if g_long else long
            why = (f"abbreviation: '{missing}' paired with "
                   f"'{long if g_long else short}' ({evidence}); "
                   f"recorded as variant (below min_freq)")
            g.variants[missing] += 0
            g.reasons.append(why)
            g.has_abbrev_pair = True
            if g_short and len(long) > len(short):
                g.forced_representative = long
            merge_log.append(why)

    # acronym letter-match fallback (no explicit pattern in text)
    group_list = list(groups.values())
    for g in group_list:
        if g not in groups.values():
            continue
        rep = g.candidates[0].surface()
        if len(rep.split()) < 2:
            continue
        acro = acronym_of(rep)
        if len(acro) < 2:
            continue
        other = groups.get(canonical_key(acro))
        if other and other is not g:
            why = (f"abbreviation: '{other.candidates[0].surface()}' matches "
                   f"the acronym of '{rep}'; merged")
            g.has_abbrev_pair = True
            merge_groups(g, other, why, merge_log)

    result = [g.finalize() for g in groups.values()]
    result.sort(key=lambda g: (-g.score, -g.freq))
    return result, merge_log
