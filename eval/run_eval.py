"""Benchmark: ablation over the five retrieval configurations, guardrail metrics, isolation
audit and latency. Writes eval/results.md (generated; never edit it by hand).

Protocol
  * questions carry a split: dev (1/3) is used to tune the nature weight, the support
    threshold tau and the conflict margin; every number in the tables is on test (2/3).
  * the cross-encoder score cache is ON while tuning (same pairs
    are scored many times) and OFF during the measured pass, so latencies are real.
  * isolation is audited on every retrieved passage of every configuration and question
    (dev + test), by reading the owner of each chunk id with a separate query.

Run:  python -m eval.run_eval [--generator extractive|ollama|anthropic]"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections import defaultdict

from rag import models, store
from rag.answer import answer
from rag.config import ARTIFACTS, EMBED_MODEL, EMBED_REVISION, QUESTIONS_FILE, RERANK_MODEL, RERANK_REVISION, ROOT
from rag.retrieve import search

WEIGHTS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
TAUS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
MARGINS = [0.02, 0.05, 0.1, 0.2, 0.3]


def load_questions():
    return [json.loads(l) for l in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()]


def ranks(q, hits) -> list[int]:
    want = {tuple(s) for s in q["sources"]}
    return [i for i, h in enumerate(hits, 1) if (h["doc_id"], h["heading"]) in want]


def recall_at(q, hits, k=5) -> float:
    want = {tuple(s) for s in q["sources"]}
    got = {(h["doc_id"], h["heading"]) for h in hits[:k]}
    return len(want & got) / len(want)


def rr(q, hits) -> float:
    r = ranks(q, hits)
    return 1.0 / r[0] if r else 0.0


def correct(q, a) -> bool:
    if not q["answerable"]:
        return a.refused
    if a.refused:
        return False
    if q["kind"] == "conflict":
        return set(q["expected"]) <= {c["value"] for c in a.conflict}
    if q["kind"] == "ambiguous_date":
        return "ambiguous_date" in a.flags and q["expected"][0] in a.text
    return a.primary == q["expected"][0] and not a.conflict


def tune(conn, dev):
    """Pick the nature weight (dev MRR), then tau and margin (dev answer accuracy)."""
    models.USE_CACHE = True
    answerable = [q for q in dev if q["answerable"]]
    mrr = {}
    for w in WEIGHTS:
        mrr[w] = statistics.mean(rr(q, search(conn, q["company"], q["question"], "rerank+nature", nature_weight=w))
                                 for q in answerable)
    best_w = max(WEIGHTS, key=lambda w: (round(mrr[w], 4), -w))
    hits = {q["id"]: search(conn, q["company"], q["question"], "rerank+nature", nature_weight=best_w) for q in dev}
    acc = {}
    for t in TAUS:
        for m in MARGINS:
            acc[(t, m)] = statistics.mean(correct(q, answer(q["question"], [dict(h) for h in hits[q["id"]]], t, m))
                                          for q in dev)
    best_t, best_m = max(acc, key=lambda k: (round(acc[k], 4), k[0], -k[1]))
    models.USE_CACHE = False
    models.clear_cache()
    return {"nature_weight": best_w, "tau": best_t, "margin": best_m,
            "dev_mrr_by_weight": {str(k): round(v, 3) for k, v in mrr.items()},
            "dev_accuracy": round(acc[(best_t, best_m)], 3)}


def pct(x):
    return f"{100 * x:.1f} %"


def p(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="extractive", choices=["extractive", "ollama", "anthropic"])
    args = ap.parse_args()

    conn = store.connect()
    qs = load_questions()
    dev = [q for q in qs if q["split"] == "dev"]
    test = [q for q in qs if q["split"] == "test"]
    print(f"tuning on {len(dev)} dev questions...")
    settings = tune(conn, dev)
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "settings.json").write_text(json.dumps({k: settings[k] for k in ("tau", "margin", "nature_weight")}, indent=2))
    print("settings:", settings)

    w = settings["nature_weight"]
    configs = [("BM25 only", "bm25", 0.0), ("Vectors only (pgvector)", "vector", 0.0),
               ("Hybrid: BM25 + vectors, RRF", "rrf", 0.0), ("+ cross-encoder rerank", "rerank", 0.0)]
    if w > 0:
        configs.append((f"+ passage-nature signal (tuned weight {w})", "rerank+nature", w))
    configs.append(("+ passage-nature signal (weight 0.1, not tuned)", "rerank+nature", 0.1))
    app_config = configs[-2][0]   # what the app runs: rerank with the tuned nature weight

    rows, per_kind, leaks, audited, latencies = {}, defaultdict(lambda: [0, 0]), 0, 0, {}
    for name, mode, weight in configs:
        rec, mrr, acc, lat_r, lat_a = [], [], [], [], []
        tp = fp = fn = 0
        for q in qs:
            models.clear_cache()
            t0 = time.perf_counter()
            hits = search(conn, q["company"], q["question"], mode, nature_weight=weight)
            t1 = time.perf_counter()
            owners = store.owner_of(conn, [h["id"] for h in hits])
            audited += len(hits)
            leaks += sum(owners[h["id"]] != q["company"] for h in hits)
            if q["split"] != "test":
                continue
            a = answer(q["question"], hits, settings["tau"], settings["margin"], args.generator)
            t2 = time.perf_counter()
            lat_r.append(1000 * (t1 - t0))
            lat_a.append(1000 * (t2 - t0))
            ok = correct(q, a)
            acc.append(ok)
            if q["answerable"]:
                rec.append(recall_at(q, hits))
                mrr.append(rr(q, hits))
            tp += a.refused and not q["answerable"]
            fp += a.refused and q["answerable"]
            fn += (not a.refused) and not q["answerable"]
            if name == app_config:
                kind = q["kind"].split(":")[0] if q["kind"].startswith("unanswerable") else (
                    q["kind"] if q["kind"] in ("conflict", "ambiguous_date") else "single fact")
                per_kind[kind][0] += ok
                per_kind[kind][1] += 1
        rows[name] = {"recall@5": statistics.mean(rec), "mrr": statistics.mean(mrr), "accuracy": statistics.mean(acc),
                      "refusal_precision": tp / (tp + fp) if tp + fp else 0.0, "refusal_recall": tp / (tp + fn) if tp + fn else 0.0}
        latencies[name] = (lat_r, lat_a)
        print(name, {k: round(v, 3) for k, v in rows[name].items()})

    stats = json.loads((ARTIFACTS / "index_stats.json").read_text())
    n_un = sum(not q["answerable"] for q in test)
    L = ["# Benchmark results", "",
         "_Generated by `python -m eval.run_eval`. Do not edit by hand._", "",
         f"- Corpus: {stats['documents']} invented documents, {stats['chunks']} chunks, 3 companies (FR/EN/DE)",
         f"- Questions: {len(qs)} ({len(dev)} dev for tuning, **{len(test)} test reported below**, of which {n_un} have no answer)",
         f"- Embeddings: `{EMBED_MODEL}@{EMBED_REVISION[:7]}` · Reranker: `{RERANK_MODEL}@{RERANK_REVISION[:7]}`",
         f"- Answer generator: `{args.generator}` · Hardware: {platform.processor() or platform.machine()}, CPU inference",
         f"- Tuned on dev: nature weight = {settings['nature_weight']}, support threshold tau = {settings['tau']}, conflict margin = {settings['margin']}",
         "", "## Ablation (test split)", "",
         "| Configuration | Recall@5 | MRR@8 | Answer accuracy | Refusal precision | Refusal recall | p50 latency | p95 latency |",
         "|---|---|---|---|---|---|---|---|"]
    for name, _, _ in configs:
        r = rows[name]
        la = latencies[name][1]
        L.append(f"| {name} | {pct(r['recall@5'])} | {r['mrr']:.3f} | {pct(r['accuracy'])} | "
                 f"{pct(r['refusal_precision'])} | {pct(r['refusal_recall'])} | {p(la, 0.5):.0f} ms | {p(la, 0.95):.0f} ms |")
    L += ["", "Recall@5 and MRR are computed on answerable questions (share of expected source sections in the top 5; "
          "reciprocal rank of the first one). Answer accuracy covers every test question: the right value for single facts, "
          "both values flagged for conflicts, the ambiguity flagged for ambiguous dates, a refusal for questions without an answer. "
          "Every configuration uses the same answer stage; only retrieval changes. Latency = retrieval + answer, per question.",
          "", f"## Accuracy by question kind (configuration used by the app: {app_config}; test split)", "", "| Kind | Correct | Total | Accuracy |", "|---|---|---|---|"]
    for kind in sorted(per_kind):
        c, n = per_kind[kind]
        L.append(f"| {kind} | {c} | {n} | {pct(c / n)} |")
    nr = stats["nature_classifier"]
    L += ["", "## Passage-nature classifier", "",
          f"Character n-gram TF-IDF + logistic regression, trained on {nr['train_sections']} labelled sections "
          f"(half of the documents), evaluated on the {nr['test_sections']} sections of the other half: "
          f"accuracy {pct(nr['accuracy'])}, macro-F1 {nr['macro_f1']:.3f}.",
          "", f"Dev MRR by nature weight: {settings['dev_mrr_by_weight']}",
          "", "## Tenant isolation", "",
          f"Retrieved passages audited (all configurations, all {len(qs)} questions): **{audited}**. "
          f"Passages owned by another company: **{leaks}**.", ""]
    (ROOT / "eval" / "results.md").write_text("\n".join(L), encoding="utf-8")
    # The README shows the same table, copied here between markers so it is never typed by hand.
    readme = ROOT / "README.md"
    if readme.exists():
        start, end = "<!-- ablation:start -->", "<!-- ablation:end -->"
        text = readme.read_text(encoding="utf-8")
        ablation = "\n".join(L[L.index("## Ablation (test split)") + 2: L.index("## Ablation (test split)") + 4 + len(configs)])
        if start in text and end in text:
            head, _, rest = text.partition(start)
            _, _, tail = rest.partition(end)
            readme.write_text(f"{head}{start}\n{ablation}\n{end}{tail}", encoding="utf-8")
    print(f"leaks: {leaks} / {audited}")


if __name__ == "__main__":
    main()
