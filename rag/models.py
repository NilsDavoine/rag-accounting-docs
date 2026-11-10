"""Lazy-loaded local models (CPU is enough; CUDA is used if available)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from rag.config import EMBED_MODEL, EMBED_REVISION, RERANK_MODEL, RERANK_REVISION


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, revision=EMBED_REVISION, device=_device())


@lru_cache(maxsize=1)
def cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, revision=RERANK_REVISION, max_length=256, device=_device())


def embed_passages(texts: list[str]) -> np.ndarray:
    # e5 models expect "passage: " / "query: " prefixes
    return embedder().encode([f"passage: {t}" for t in texts], normalize_embeddings=True, batch_size=32)


def embed_query(text: str) -> np.ndarray:
    return embedder().encode([f"query: {text}"], normalize_embeddings=True)[0]


_CACHE: dict[tuple[str, str], float] = {}
USE_CACHE = False   # the benchmark turns it on while tuning thresholds, off while timing


def clear_cache() -> None:
    _CACHE.clear()


def rerank_scores(query: str, texts: list[str]) -> np.ndarray:
    """Relevance in [0, 1]: the model outputs one logit per pair, we apply the sigmoid."""
    if not texts:
        return np.zeros(0)
    todo = [t for t in dict.fromkeys(texts) if not (USE_CACHE and (query, t) in _CACHE)]
    if todo:
        import torch
        logits = cross_encoder().predict([(query, t) for t in todo], batch_size=32, show_progress_bar=False,
                                         activation_fct=torch.nn.Identity())
        fresh = dict(zip(todo, (1 / (1 + np.exp(-np.asarray(logits, dtype=float)))).tolist()))
        if USE_CACHE:
            _CACHE.update({(query, t): v for t, v in fresh.items()})
    else:
        fresh = {}
    return np.array([fresh[t] if t in fresh else _CACHE[(query, t)] for t in texts])
