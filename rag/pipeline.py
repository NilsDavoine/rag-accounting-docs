"""One call from question to guarded answer, used by the CLI and the web page."""
from __future__ import annotations

import json
import time

from rag import store
from rag.answer import Answer, answer
from rag.config import ARTIFACTS
from rag.retrieve import search

DEFAULTS = {"tau": 0.1, "margin": 0.1, "nature_weight": 0.1}


def settings() -> dict:
    p = ARTIFACTS / "settings.json"
    return {**DEFAULTS, **(json.loads(p.read_text()) if p.exists() else {})}


def ask(conn, company: str, question: str, mode: str = "rerank+nature", generator: str = "extractive"):
    s = settings()
    t0 = time.perf_counter()
    hits = search(conn, company, question, mode, nature_weight=s["nature_weight"])
    t1 = time.perf_counter()
    ans: Answer = answer(question, hits, s["tau"], s["margin"], generator)
    t2 = time.perf_counter()
    return ans, hits, {"retrieval_ms": round(1000 * (t1 - t0)), "answer_ms": round(1000 * (t2 - t1))}


if __name__ == "__main__":
    conn = store.connect()
    print(ask(conn, "ombrelune", "Quel taux horaire avons-nous convenu avec Cabinet Vauclerc Conseil ?")[0])
