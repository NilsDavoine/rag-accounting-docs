"""Okapi BM25, in pure Python. This is the reference implementation: the production path
computes the same formula in SQL (rag/store.py), and a test checks both agree.

score(D, Q) = sum_{t in Q} idf(t) * tf(t, D) * (k1 + 1) / (tf(t, D) + k1 * (1 - b + b * |D| / avgdl))
idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
"""
from __future__ import annotations

import math
from collections import Counter

K1 = 1.2
B = 0.75


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = K1, b: float = B):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.n = len(docs)
        self.avgdl = sum(self.lens) / self.n if self.n else 0.0
        self.df = Counter(t for tf in self.tfs for t in tf)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query: list[str], i: int) -> float:
        tf, dl = self.tfs[i], self.lens[i]
        s = 0.0
        for t in set(query):
            f = tf.get(t, 0)
            if f:
                s += self.idf(t) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def top(self, query: list[str], k: int) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(self.n)]
        scores = [x for x in scores if x[1] > 0]
        return sorted(scores, key=lambda x: -x[1])[:k]
