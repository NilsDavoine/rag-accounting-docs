"""Tenant isolation, against the real database and the real retrieval code.

These tests do NOT skip when the database is down: an isolation test that silently skips
is an isolation test that silently passes. Start it with `docker compose up -d` and build
the index with `python -m rag.index` first.

Mutation-tested: removing the company filter from either SQL query turns this file red
(see README, "How the isolation test was checked")."""
import json

import pytest

from rag import store
from rag.config import QUESTIONS_FILE
from rag.retrieve import search

COMPANIES = ["ombrelune", "quillfern", "halvardt"]


@pytest.fixture(scope="module")
def conn():
    return store.connect()


@pytest.fixture(scope="module")
def probes(conn):
    """Every bench question, plus one probe per foreign document title and invoice number:
    queries written to pull another company's passages."""
    qs = [json.loads(l)["question"] for l in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()]
    rows = conn.execute("SELECT DISTINCT company, title FROM chunks ORDER BY 1, 2").fetchall()
    return qs, rows


def _check(conn, company, hits):
    owners = store.owner_of(conn, [h["id"] for h in hits])
    foreign = [(h["doc_id"], owners[h["id"]]) for h in hits if owners[h["id"]] != company]
    assert not foreign, f"{company} received passages of another company: {foreign}"


@pytest.mark.parametrize("mode", ["bm25", "vector"])
def test_no_company_ever_receives_another_companys_passage(conn, probes, mode):
    questions, titles = probes
    n_hits = 0
    for company in COMPANIES:
        foreign_titles = [t for c, t in titles if c != company]
        for q in questions + foreign_titles:
            hits = search(conn, company, q, mode, k=50)
            n_hits += len(hits)
            _check(conn, company, hits)
    assert n_hits > 1000, "the probes returned almost nothing: the test would pass vacuously"


def test_full_pipeline_with_reranking_stays_inside_the_company(conn, probes):
    _, titles = probes
    for company in COMPANIES:
        for _, t in [x for x in titles if x[0] != company][:6]:
            _check(conn, company, search(conn, company, t, "rerank+nature", k=8, nature_weight=0.1))


def test_positive_control_the_probes_do_find_the_owners_documents(conn, probes):
    """Without this, an empty result would look like perfect isolation."""
    _, titles = probes
    for company, title in titles[::7]:
        hits = search(conn, company, title, "bm25", k=5)
        assert title in [h["title"] for h in hits], f"{company} cannot find its own document {title!r}"


def test_sql_bm25_matches_the_python_reference(conn):
    from rag.bm25 import BM25
    from rag.chunking import load_corpus
    from rag.config import CORPUS_DIR
    from rag.text import tokenize
    chunks = [c for c in load_corpus(CORPUS_DIR) if c.company == "quillfern"]
    ref = BM25([tokenize(c.prefixed) for c in chunks])
    for q in ["What is the total of the invoice from Pellow Office Goods?", "notice period termination agreement",
              "VAT second quarter 2025"]:
        expected = {(chunks[i].doc_id, chunks[i].heading, chunks[i].part): s for i, s in ref.top(tokenize(q), 10)}
        got = {(h["doc_id"], h["heading"], h["part"]): h["score"] for h in store.bm25_search(conn, "quillfern", tokenize(q), 10)}
        assert set(got) == set(expected)
        for key, s in got.items():
            assert s == pytest.approx(expected[key], rel=1e-9)
