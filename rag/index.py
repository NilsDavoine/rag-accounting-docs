"""Build the index: chunk the corpus, classify each chunk's nature, embed, store.

Run:  python -m rag.index"""
from __future__ import annotations

import json
import time

from rag import nature, store
from rag.chunking import load_corpus
from rag.config import ARTIFACTS, CORPUS_DIR, LABELS_FILE
from rag.models import embed_passages
from rag.text import tokenize


def build() -> dict:
    t0 = time.time()
    chunks = load_corpus(CORPUS_DIR)
    model, report = nature.train(chunks, LABELS_FILE)
    nature.save(model, ARTIFACTS / "nature.pkl")
    preds = nature.predict(model, chunks)
    vectors = embed_passages([c.prefixed for c in chunks])
    conn = store.connect()
    store.reset(conn)
    for c, (lab, proba), vec in zip(chunks, preds, vectors):
        store.insert_chunk(conn, c, lab, proba, tokenize(c.prefixed), vec)
    stats = {"documents": len({c.doc_id for c in chunks}), "chunks": len(chunks),
             "companies": sorted({c.company for c in chunks}), "nature_classifier": report,
             "seconds": round(time.time() - t0, 1)}
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "index_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
