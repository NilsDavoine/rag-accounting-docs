"""Build the index: chunk the corpus, classify each chunk's nature, embed, store.

Run:  python -m rag.index            main corpus (trains the nature classifier)
      python -m rag.index --hard     held-out hard corpus, in its own schema; reuses the
                                     classifier trained on the main corpus and only measures it"""
from __future__ import annotations

import argparse
import json
import time

from rag import nature, store
from rag.chunking import load_corpus
from rag.config import ARTIFACTS, CORPUS_DIR, HARD_CORPUS_DIR, HARD_LABELS_FILE, HARD_SCHEMA, LABELS_FILE
from rag.models import embed_passages
from rag.text import tokenize


def build(hard: bool = False) -> dict:
    t0 = time.time()
    chunks = load_corpus(HARD_CORPUS_DIR if hard else CORPUS_DIR)
    if hard:
        model = nature.load(ARTIFACTS / "nature.pkl")
        report = nature.evaluate(model, chunks, HARD_LABELS_FILE)
    else:
        model, report = nature.train(chunks, LABELS_FILE)
        nature.save(model, ARTIFACTS / "nature.pkl")
    preds = nature.predict(model, chunks)
    vectors = embed_passages([c.prefixed for c in chunks])
    conn = store.connect(HARD_SCHEMA if hard else None)
    store.reset(conn)
    for c, (lab, proba), vec in zip(chunks, preds, vectors):
        store.insert_chunk(conn, c, lab, proba, tokenize(c.prefixed), vec)
    stats = {"documents": len({c.doc_id for c in chunks}), "chunks": len(chunks),
             "companies": sorted({c.company for c in chunks}), "nature_classifier": report,
             "seconds": round(time.time() - t0, 1)}
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / ("index_stats_hard.json" if hard else "index_stats.json")).write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", action="store_true", help="index the held-out hard corpus instead of the main one")
    print(json.dumps(build(ap.parse_args().hard), indent=2))
