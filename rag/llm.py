"""Optional generators. Neither is needed to run the benchmark.

  ollama     a local model served by Ollama (default qwen2.5:1.5b-instruct)
  anthropic  Claude, only if the user sets ANTHROPIC_API_KEY

Whatever the model writes, every number in its answer must exist in the passages it was
given; otherwise the caller falls back to quoting the passage."""
from __future__ import annotations

import json
import os
import urllib.request

from rag.config import ANTHROPIC_MODEL, OLLAMA_MODEL, OLLAMA_URL
from rag.text import extract_values

SYSTEM = ("You answer questions about a company's accounting documents. Use ONLY the numbered passages. "
          "Answer in the language of the question, in one or two sentences, and cite passages as [1], [2]. "
          "Copy amounts and dates exactly as written; never round. If the passages do not contain the answer, "
          "say you cannot answer.")


def _prompt(question: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i}] {p['prefixed']}" for i, p in enumerate(passages, 1))
    return f"Passages:\n{ctx}\n\nQuestion: {question}"


def _ollama(prompt: str) -> str:
    body = json.dumps({"model": OLLAMA_MODEL, "system": SYSTEM, "prompt": prompt, "stream": False, "think": False,
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"].strip()


def _anthropic(prompt: str) -> str:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; use --generator extractive or ollama")
    import anthropic
    msg = anthropic.Anthropic().messages.create(model=ANTHROPIC_MODEL, max_tokens=300, temperature=0,
                                                system=SYSTEM, messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def grounded_answer(question: str, passages: list[dict], backend: str, types) -> tuple[str, bool]:
    text = {"ollama": _ollama, "anthropic": _anthropic}[backend](_prompt(question, passages))
    allowed = {v["value"] for p in passages for v in extract_values(p["text"])}
    written = [v for v in extract_values(text) if types is None or v["type"] in types]
    return text, any(v["value"] not in allowed for v in written)
