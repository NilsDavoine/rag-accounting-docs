"""Answer generation: the same guarded pipeline, with the sentence written by an LLM instead of
quoted. Each run adds or replaces one row per question set in eval/generation.json, and the
"Answer generation" table of eval/results.md is rebuilt from that file.

  python -m eval.run_generation --generator extractive    no model, no key (run by `make eval`)
  python -m eval.run_generation --generator anthropic     Claude Haiku 5.5, needs ANTHROPIC_API_KEY
  python -m eval.run_generation --generator ollama        a local model through Ollama

Before the first Claude call it prints the number of requests and an estimated cost, then asks
for confirmation (--yes skips the question).

Configuration: what the app runs (cross-encoder rerank with the tuned nature weight; tau and the
conflict margin from artifacts/settings.json, tuned on the main dev split). Questions: the test
split of the main bench and the whole held-out hard set.

Scoring reads the written answer: the expected value must appear in it as a typed value (so
"26 EUR" and "EUR 26.00" both count). A question without an answer must be refused, by the
guardrails or by the model replying NO_ANSWER. Conflicts and ambiguous dates are scored as in
run_eval: those warnings are added by deterministic code after the model has written."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import sys
import time

from rag import llm, models, store
from rag.answer import answer
from rag.config import ANTHROPIC_MODEL, HARD_QUESTIONS_FILE, HARD_SCHEMA, OLLAMA_MODEL, QUESTIONS_FILE, ROOT
from rag.pipeline import settings as load_settings
from rag.retrieve import search
from rag.text import extract_values

RESULTS = ROOT / "eval" / "results.md"
STORE = ROOT / "eval" / "generation.json"
# USD per million tokens (input, output), for prompts up to 100,000 tokens.
PRICES = {"claude-haiku-5-5": (0.10, 0.50)}
CHARS_PER_TOKEN = 3.0          # conservative for FR/DE text: overestimates the input tokens
EXPECTED_OUTPUT_TOKENS = 300   # a two-sentence answer plus some low-effort thinking (a guess, shown as such)
PRICE_TIER_LIMIT = 100_000


def load_sets():
    main = [json.loads(l) for l in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()]
    hard = [json.loads(l) for l in HARD_QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()]
    return [("main (test split)", None, [q for q in main if q["split"] == "test"]), ("hard (held out)", HARD_SCHEMA, hard)]


def collect(s: dict) -> dict[str, list[tuple[dict, list[dict]]]]:
    """Retrieval once per question, with the app's configuration."""
    out = {}
    for name, schema, qs in load_sets():
        conn = store.connect(schema)
        out[name] = [(q, search(conn, q["company"], q["question"], "rerank+nature", nature_weight=s["nature_weight"])) for q in qs]
    return out


def plan(items: dict, s: dict) -> list[str]:
    """The exact prompts an LLM would receive: one per question that passes the guardrails."""
    prompts = []
    for pairs in items.values():
        for q, hits in pairs:
            a = answer(q["question"], [dict(h) for h in hits], s["tau"], s["margin"], "extractive")
            if not a.refused:
                prompts.append(llm.SYSTEM + "\n" + llm.prompt(q["question"], a.passages))
    return prompts


def estimate(prompts: list[str], model: str) -> dict:
    tokens = [math.ceil(len(p) / CHARS_PER_TOKEN) for p in prompts]
    pin, pout = PRICES.get(model, (None, None))
    est = {"requests": len(prompts), "input_tokens": sum(tokens), "largest_prompt": max(tokens, default=0),
           "output_expected": EXPECTED_OUTPUT_TOKENS * len(prompts), "output_max": llm.MAX_TOKENS * len(prompts)}
    if pin is not None:
        est["cost_expected"] = (est["input_tokens"] * pin + est["output_expected"] * pout) / 1e6
        est["cost_max"] = (est["input_tokens"] * pin + est["output_max"] * pout) / 1e6
    return est


def text_correct(q: dict, a) -> bool:
    if not q["answerable"]:
        return a.refused
    if a.refused:
        return False
    if q["kind"] == "conflict":
        return set(q["expected"]) <= {c["value"] for c in a.conflict}
    if q["kind"] == "ambiguous_date":
        return "ambiguous_date" in a.flags and q["expected"][0] in a.text
    return any(v["value"] == q["expected"][0] and v["type"] == q["type"] for v in extract_values(a.text))


def evaluate(pairs, s: dict, generator: str) -> dict:
    before = dict(llm.USAGE)
    ok, lat, fallbacks = [], [], 0
    tp = fp = fn = 0
    for q, hits in pairs:
        t0 = time.perf_counter()
        a = answer(q["question"], [dict(h) for h in hits], s["tau"], s["margin"], generator)
        lat.append(1000 * (time.perf_counter() - t0))
        ok.append(text_correct(q, a))
        fallbacks += "llm_number_not_in_sources" in a.flags
        tp += a.refused and not q["answerable"]
        fp += a.refused and q["answerable"]
        fn += (not a.refused) and not q["answerable"]
    used = {k: llm.USAGE[k] - before[k] for k in llm.USAGE}
    row = {"questions": len(pairs), "accuracy": statistics.mean(ok), "refusal_precision": tp / (tp + fp) if tp + fp else 0.0,
           "refusal_recall": tp / (tp + fn) if tp + fn else 0.0, "fallbacks": fallbacks,
           "p50_ms": statistics.median(lat), "requests": used["requests"],
           "input_tokens": used["input_tokens"], "output_tokens": used["output_tokens"]}
    if generator == "anthropic" and ANTHROPIC_MODEL in PRICES:
        pin, pout = PRICES[ANTHROPIC_MODEL]
        row["cost_usd"] = (used["input_tokens"] * pin + used["output_tokens"] * pout) / 1e6
    return row


def model_name(generator: str) -> str:
    return {"extractive": "none (quotes the best clause)", "anthropic": ANTHROPIC_MODEL, "ollama": OLLAMA_MODEL}[generator]


def render() -> None:
    rows = json.loads(STORE.read_text(encoding="utf-8")) if STORE.exists() else {}
    L = ["| Generator | Model | Questions | Accuracy (written answer) | Refusal precision | Refusal recall | "
         "Answers replaced by the quote | p50 answer time | API cost | Measured |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(rows, key=lambda k: (not k.split("|")[2].startswith("main"), k.split("|")[0] != "extractive", k)):
        r = rows[key]
        gen, model, qset = key.split("|")
        cost = f"${r['cost_usd']:.4f} ({r['input_tokens']:,} in / {r['output_tokens']:,} out)" if "cost_usd" in r else "–"
        L.append(f"| {gen} | `{model}` | {qset}: {r['questions']} | {100 * r['accuracy']:.1f} % | {100 * r['refusal_precision']:.1f} % | "
                 f"{100 * r['refusal_recall']:.1f} % | {r['fallbacks']} | {r['p50_ms']:.0f} ms | {cost} | {r['date']} |")
    if not any(k.startswith("anthropic|") for k in rows):
        L += ["", "Claude has not been measured yet: it needs an API key. "
              "`ANTHROPIC_API_KEY=... make eval-claude` adds its rows here, after printing the estimated cost."]
    L += ["", "Same retrieval and guardrails for every generator. \"Answers replaced by the quote\": the model wrote a number "
          "that is in none of its passages, so the app showed the quoted clause instead. Answer time excludes retrieval."]
    if RESULTS.exists():
        from eval.run_eval import copy_block
        RESULTS.write_text(copy_block(RESULTS.read_text(encoding="utf-8"), "generation", L), encoding="utf-8")


def confirm(est: dict, model: str, assume_yes: bool) -> None:
    print(f"\n{est['requests']} requests to {model}. Estimated input: {est['input_tokens']:,} tokens "
          f"(largest prompt {est['largest_prompt']:,}; price tier up to {PRICE_TIER_LIMIT:,}).")
    if "cost_expected" in est:
        print(f"Estimated cost: about ${est['cost_expected']:.4f} with ~{EXPECTED_OUTPUT_TOKENS} output tokens per answer; "
              f"at most ${est['cost_max']:.4f} if every answer used max_tokens={llm.MAX_TOKENS}.")
    else:
        print(f"No price table for {model}; check its price before going on.")
    if est["largest_prompt"] > PRICE_TIER_LIMIT:
        sys.exit("a prompt exceeds the 100,000-token price tier; aborting")
    if assume_yes:
        return
    if not sys.stdin.isatty():
        sys.exit("not confirmed: re-run with --yes")
    if input("Proceed? [y/N] ").strip().lower() != "y":
        sys.exit("cancelled")


def main(argv=None, collect_fn=None) -> None:
    ap = argparse.ArgumentParser(description="Measure an answer generator on the app's configuration.")
    ap.add_argument("--generator", default="extractive", choices=["extractive", "anthropic", "ollama"])
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation before paid API calls")
    args = ap.parse_args(argv)
    if args.generator == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Nothing was called.")
    s = load_settings()
    items = (collect_fn or collect)(s)
    if args.generator == "anthropic":
        confirm(estimate(plan(items, s), ANTHROPIC_MODEL), ANTHROPIC_MODEL, args.yes)
    models.USE_CACHE = False
    rows = json.loads(STORE.read_text(encoding="utf-8")) if STORE.exists() else {}
    today = dt.date.today().isoformat()
    for qset, pairs in items.items():
        r = evaluate(pairs, s, args.generator)
        r["date"] = today
        rows[f"{args.generator}|{model_name(args.generator)}|{qset}"] = r
        print(qset, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    STORE.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    render()


if __name__ == "__main__":
    main()
