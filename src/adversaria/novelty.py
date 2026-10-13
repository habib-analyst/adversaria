"""Argument-novelty critic: pure-numpy TF-IDF + cosine dissimilarity.

The critic scores each proposal on NOVELTY — how different its argument is
from every proposal that came before it in the same debate — rather than on
correctness (which needs ground truth the critic doesn't have). Repeated
angles are discounted even when correct, fighting consensus hallucination
and parrot double-counting.

Implemented in pure numpy (no sklearn) to keep the dependency footprint at
zero: tokenise -> TF-IDF with smoothed IDF -> L2-normalised rows ->
novelty_i = 1 - max cosine similarity to any earlier proposal.
"""

from __future__ import annotations

import re

import numpy as np


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def tfidf_matrix(docs: list[str]) -> np.ndarray:
    """Row-normalised TF-IDF matrix, shape (n_docs, vocab)."""
    tokenised = [tokenize(d) for d in docs]
    vocab: dict[str, int] = {}
    for toks in tokenised:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    n, v = len(docs), len(vocab)
    if v == 0:
        return np.zeros((n, 1))
    tf = np.zeros((n, v))
    for i, toks in enumerate(tokenised):
        for t in toks:
            tf[i, vocab[t]] += 1.0
        if toks:
            tf[i] /= len(toks)
    df = np.count_nonzero(tf > 0, axis=0)
    idf = np.log((n + 1.0) / (df + 1.0)) + 1.0  # smoothed
    mat = tf * idf
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def novelty_scores(rationales: list[str]) -> list[float]:
    """Novelty of each rationale vs all earlier ones, in [0, 1].

    First proposal always scores 1.0 (nothing to repeat yet).
    """
    if not rationales:
        return []
    mat = tfidf_matrix(rationales)
    sims = mat @ mat.T  # cosine similarities (rows are unit norm)
    out: list[float] = []
    for i in range(len(rationales)):
        if i == 0:
            out.append(1.0)
        else:
            best = float(np.max(sims[i, :i]))
            out.append(float(min(1.0, max(0.0, 1.0 - best))))
    return out


class NoveltyCritic:
    """Stateful critic: score each incoming proposal against the debate so far."""

    def __init__(self) -> None:
        self._rationales: list[str] = []

    def reset(self) -> None:
        self._rationales = []

    def score(self, rationale: str) -> float:
        """Novelty of `rationale` vs everything seen so far; then remembers it."""
        novelty = novelty_scores(self._rationales + [rationale])[-1]
        self._rationales.append(rationale)
        return float(novelty)

    def batch_score(self, rationales: list[str]) -> list[float]:
        self.reset()
        return [self.score(r) for r in rationales]
