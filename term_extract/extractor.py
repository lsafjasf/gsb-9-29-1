"""End-to-end extraction pipeline."""
from .tokenizer import tokenize, sentences
from .candidates import generate_candidates, build_containment
from .scoring import score_candidates, rank
from .merging import merge_candidates


class TermExtractor:
    """Unsupervised domain-term extractor (stdlib only).

    Parameters
    ----------
    max_n : int          max tokens per candidate (CJK chars count as tokens)
    min_freq : int       minimum corpus frequency to keep a candidate
    window : int         context-window size (tokens each side) used for
                         boundary entropy and the completeness test
    top_k : int | None   keep only the top-k merged terms (None = all)
    min_score : float    score floor applied before merging
    """

    def __init__(self, max_n=6, min_freq=2, window=2, top_k=None,
                 min_score=0.05, unigram_min_freq=3, zh2_min_freq=3):
        self.max_n = max_n
        self.min_freq = min_freq
        self.window = window
        self.top_k = top_k
        self.min_score = min_score
        self.unigram_min_freq = unigram_min_freq
        self.zh2_min_freq = zh2_min_freq

    # -- internal ---------------------------------------------------------
    def _prepare(self, docs):
        docs_tokens = []
        for doc in docs:
            docs_tokens.append(sentences(tokenize(doc)))
        cands = generate_candidates(docs_tokens, max_n=self.max_n,
                                    window=self.window)
        contains = build_containment(cands)
        score_candidates(cands, contains, min_freq=self.min_freq)
        return cands

    # -- public API -------------------------------------------------------
    def extract_candidates(self, docs):
        """Ranked, pre-merge candidate list (useful for inspection)."""
        if isinstance(docs, str):
            docs = [docs]
        cands = self._prepare(docs)
        return rank(cands, min_freq=self.min_freq, top_k=None,
                    min_score=self.min_score,
                    unigram_min_freq=self.unigram_min_freq,
                    zh2_min_freq=self.zh2_min_freq)

    def extract(self, docs):
        """Full pipeline: returns (terms, merge_log).

        terms is a list of dicts (see TermGroup.as_dict), sorted by score.
        """
        if isinstance(docs, str):
            docs = [docs]
        raw_text = "\n".join(docs)
        cands = self._prepare(docs)
        ranked = rank(cands, min_freq=self.min_freq, top_k=None,
                      min_score=self.min_score,
                      unigram_min_freq=self.unigram_min_freq,
                      zh2_min_freq=self.zh2_min_freq)
        groups, merge_log = merge_candidates(ranked, raw_text=raw_text)
        if self.top_k:
            groups = groups[: self.top_k]
        return [g.as_dict() for g in groups], merge_log
