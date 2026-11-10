"""Retrieval: the five configurations compared in the ablation table.

  bm25           BM25 (SQL) only
  vector         pgvector cosine only
  rrf            both lists fused with Reciprocal Rank Fusion
  rerank         RRF candidates re-scored by the multilingual cross-encoder
  rerank+nature  cross-encoder score + weight * P(wanted nature | passage)

The company is passed to every SQL query; this module never sees another tenant's rows."""
from __future__ import annotations

from rag import store
from rag.config import CANDIDATES, FINAL_K, RRF_K
from rag.models import embed_query, rerank_scores
from rag.nature import wanted_nature
from rag.text import tokenize

MODES = ["bm25", "vector", "rrf", "rerank", "rerank+nature"]


def rrf(lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    fused: dict[int, dict] = {}
    for lst in lists:
        for rank, hit in enumerate(lst, start=1):
            entry = fused.setdefault(hit["id"], {**hit, "score": 0.0})
            entry["score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda h: (-h["score"], h["id"]))


def search(conn, company: str, question: str, mode: str = "rerank+nature", k: int = FINAL_K,
           nature_weight: float = 0.0) -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "bm25":
        return store.bm25_search(conn, company, tokenize(question), k)
    if mode == "vector":
        return store.vector_search(conn, company, embed_query(question), k)
    lexical = store.bm25_search(conn, company, tokenize(question), CANDIDATES)
    dense = store.vector_search(conn, company, embed_query(question), CANDIDATES)
    fused = rrf([lexical, dense])
    if mode == "rrf":
        return fused[:k]
    candidates = fused[:CANDIDATES]
    ce = rerank_scores(question, [h["prefixed"] for h in candidates])
    wanted = wanted_nature(question) if mode == "rerank+nature" else None
    for h, s in zip(candidates, ce):
        h["ce"] = float(s)
        boost = nature_weight * h["nature_proba"].get(wanted, 0.0) if wanted else 0.0
        h["score"] = float(s) + boost
    return sorted(candidates, key=lambda h: (-h["score"], h["id"]))[:k]
