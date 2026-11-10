"""PostgreSQL + pgvector storage.

Tenant isolation lives here, inside every SQL statement that returns passages:
`WHERE c.company = %(company)s`. Nothing downstream re-filters, on purpose: if the SQL
filter is wrong, tests/test_isolation.py must catch it, not a Python safety net that
would hide it.

BM25 is computed in SQL from a postings table (term, chunk, tf). Document frequencies
and the average length are computed over the asking company's chunks only, so another
tenant's vocabulary does not even influence the ranking."""
from __future__ import annotations

import json

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from rag.bm25 import B, K1
from rag.config import DATABASE_URL, EMBED_DIM

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS postings;
DROP TABLE IF EXISTS chunks;
CREATE TABLE chunks (
    id           serial PRIMARY KEY,
    company      text NOT NULL,
    doc_id       text NOT NULL,
    title        text NOT NULL,
    doc_type     text NOT NULL,
    source       text NOT NULL,
    lang         text NOT NULL,
    doc_date     date NOT NULL,
    heading      text NOT NULL,
    part         int  NOT NULL,
    text         text NOT NULL,
    prefixed     text NOT NULL,
    nature       text NOT NULL,
    nature_proba jsonb NOT NULL,
    n_tokens     int  NOT NULL,
    embedding    vector({EMBED_DIM}) NOT NULL
);
CREATE INDEX chunks_company ON chunks (company);
CREATE INDEX chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE TABLE postings (
    term     text NOT NULL,
    chunk_id int  NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    tf       int  NOT NULL,
    PRIMARY KEY (term, chunk_id)
);
"""

COLUMNS = "c.id, c.company, c.doc_id, c.title, c.heading, c.part, c.text, c.prefixed, c.nature, c.nature_proba, c.doc_type, c.doc_date"

VECTOR_SQL = f"""
SELECT {COLUMNS}, 1 - (c.embedding <=> %(qvec)s) AS score
FROM chunks c
WHERE c.company = %(company)s
ORDER BY c.embedding <=> %(qvec)s
LIMIT %(k)s
"""

BM25_SQL = f"""
WITH allowed AS (
    SELECT c.id, c.n_tokens FROM chunks c WHERE c.company = %(company)s
), stats AS (
    SELECT count(*)::float AS n, avg(n_tokens)::float AS avgdl FROM allowed
), q AS (
    SELECT DISTINCT unnest(%(terms)s::text[]) AS term
), df AS (
    SELECT p.term, count(*)::float AS df
    FROM postings p JOIN allowed a ON a.id = p.chunk_id JOIN q ON q.term = p.term
    GROUP BY p.term
), scored AS (
    SELECT a.id,
           sum(ln(1 + (s.n - df.df + 0.5) / (df.df + 0.5))
               * p.tf * ({K1} + 1) / (p.tf + {K1} * (1 - {B} + {B} * a.n_tokens / s.avgdl))) AS score
    FROM postings p
    JOIN allowed a ON a.id = p.chunk_id
    JOIN df ON df.term = p.term
    CROSS JOIN stats s
    GROUP BY a.id
)
SELECT {COLUMNS}, sc.score
FROM scored sc JOIN chunks c ON c.id = sc.id
WHERE c.company = %(company)s
ORDER BY sc.score DESC, c.id
LIMIT %(k)s
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    # pgvector >= 0.8: keep scanning the HNSW graph until enough rows pass the WHERE filter
    conn.execute("SET hnsw.iterative_scan = relaxed_order")
    return conn


def _rows(cur) -> list[dict]:
    names = [d.name for d in cur.description]
    out = []
    for r in cur.fetchall():
        row = dict(zip(names, r))
        if isinstance(row.get("nature_proba"), str):
            row["nature_proba"] = json.loads(row["nature_proba"])
        out.append(row)
    return out


def reset(conn) -> None:
    conn.execute(SCHEMA)


def insert_chunk(conn, chunk, nature: str, proba: dict, tokens: list[str], embedding: np.ndarray) -> int:
    cid = conn.execute(
        """INSERT INTO chunks (company, doc_id, title, doc_type, source, lang, doc_date, heading, part, text,
                               prefixed, nature, nature_proba, n_tokens, embedding)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
        (chunk.company, chunk.doc_id, chunk.title, chunk.doc_type, chunk.source, chunk.lang, chunk.date,
         chunk.heading, chunk.part, chunk.text, chunk.prefixed, nature, json.dumps(proba), len(tokens), embedding),
    ).fetchone()[0]
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO postings (term, chunk_id, tf) VALUES (%s,%s,%s)",
                        [(t, cid, n) for t, n in tf.items()])
    return cid


def vector_search(conn, company: str, qvec: np.ndarray, k: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(VECTOR_SQL, {"company": company, "qvec": qvec, "k": k})
        return _rows(cur)


def bm25_search(conn, company: str, terms: list[str], k: int) -> list[dict]:
    if not terms:
        return []
    with conn.cursor() as cur:
        cur.execute(BM25_SQL, {"company": company, "terms": terms, "k": k})
        return _rows(cur)


def owner_of(conn, chunk_ids: list[int]) -> dict[int, str]:
    """Audit helper (tests and benchmark only): who owns these chunks, read independently."""
    rows = conn.execute("SELECT id, company FROM chunks WHERE id = ANY(%s)", (chunk_ids,)).fetchall()
    return dict(rows)
