"""Optional generators. Neither is needed to run the benchmark.

  ollama     a local model served by Ollama (default qwen2.5:1.5b-instruct)
  anthropic  Claude (default claude-haiku-5-5), only if the user sets ANTHROPIC_API_KEY

Whatever the model writes, every number in its answer must exist in the passages it was
given; otherwise the caller falls back to quoting the passage. A model that finds no answer
replies with the sentinel NO_ANSWER, which the caller turns into a refusal."""
from __future__ import annotations

import json
import os
import urllib.request

from rag.config import ANTHROPIC_MODEL, OLLAMA_MODEL, OLLAMA_URL
from rag.text import extract_values

NO_ANSWER = "NO_ANSWER"
SYSTEM = ("You answer questions about a company's accounting documents. Use ONLY the numbered passages. "
          "Answer in the language of the question, in one or two sentences, and cite passages as [1], [2]. "
          "Copy amounts and dates exactly as written; never round. If the passages do not contain the answer, "
          f"reply with exactly {NO_ANSWER} and nothing else.")
MAX_TOKENS = 4000   # room for the model's own thinking; a grounded answer is a few dozen tokens

# Token counts of every Claude call in this process, for the cost line of the benchmark.
USAGE = {"requests": 0, "input_tokens": 0, "output_tokens": 0}


def prompt(question: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(f"[{i}] {p['prefixed']}" for i, p in enumerate(passages, 1))
    return f"Passages:\n{ctx}\n\nQuestion: {question}"


def _ollama(user: str) -> str:
    body = json.dumps({"model": OLLAMA_MODEL, "system": SYSTEM, "prompt": user, "stream": False, "think": False,
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"].strip()


def _client():
    """The real Anthropic client. Tests replace this function with a fake (no network)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; use --generator extractive or ollama")
    import anthropic
    return anthropic.Anthropic()


def _anthropic(user: str) -> str:
    # Current Claude models reject sampling parameters (temperature, top_p, top_k) and an assistant
    # prefill, so neither is sent. "low" effort keeps a short grounded answer fast and cheap.
    msg = _client().messages.create(model=ANTHROPIC_MODEL, max_tokens=MAX_TOKENS, output_config={"effort": "low"},
                                    system=SYSTEM, messages=[{"role": "user", "content": user}])
    USAGE["requests"] += 1
    USAGE["input_tokens"] += msg.usage.input_tokens
    USAGE["output_tokens"] += msg.usage.output_tokens
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def grounded_answer(question: str, passages: list[dict], backend: str, types) -> tuple[str, bool]:
    text = {"ollama": _ollama, "anthropic": _anthropic}[backend](prompt(question, passages))
    allowed = {v["value"] for p in passages for v in extract_values(p["text"])}
    written = [v for v in extract_values(text) if types is None or v["type"] in types]
    return text, any(v["value"] not in allowed for v in written)
