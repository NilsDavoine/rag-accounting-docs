"""Answering with guardrails.

1. Anchors: identifiers, years and quarters named in the question must appear in a
   passage (or its context prefix) for that passage to count as support.
2. Support threshold: if no passage reaches cross-encoder relevance `tau`, refuse.
3. Typed answer: if the question asks for an amount, a date, a duration... and no
   supporting clause contains one, refuse rather than guess.
4. Number consistency: if clauses from *different documents* that score almost as well as
   the best one (within `margin`) give different values, report both with their sources.
   Never average, never round to one of them.
5. Date ambiguity: 04/03/2025 is 4 March for a European reader and 3 April for an American
   one; say so.

The default generator is extractive (the best clause, quoted with its source) so the
benchmark is deterministic and needs no model download. An LLM (Ollama or Claude) can write
the sentence instead; its numbers are then checked against the passages."""
from __future__ import annotations

from dataclasses import dataclass, field

from rag.models import rerank_scores
from rag.text import anchors, clauses, extract_values, requested_types, satisfies

TOP_PASSAGES = 5      # passages the answerer may read
CLAUSE_PASSAGES = 3   # supporting passages split into clauses


@dataclass
class Answer:
    text: str
    refused: bool = False
    reason: str = ""
    primary: str | None = None            # normalized main value (used for scoring)
    conflict: list[dict] = field(default_factory=list)   # [{value, doc_id, heading}] when sources disagree
    flags: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    passages: list[dict] = field(default_factory=list, repr=False)   # what a generator was (or would be) given


def _refuse(reason: str) -> Answer:
    return Answer(text="I can't answer this from the documents of your company: " + reason, refused=True, reason=reason)


def _cite(h: dict) -> dict:
    return {"doc_id": h["doc_id"], "title": h["title"], "heading": h["heading"]}


def answer(question: str, hits: list[dict], tau: float, margin: float, generator: str = "extractive") -> Answer:
    anch = anchors(question)
    cands = [h for h in hits[:TOP_PASSAGES] if satisfies(anch, h["prefixed"])]
    if not cands:
        return _refuse("no passage mentions " + ", ".join(anch["ids"] + anch["years"]) if anch["ids"] or anch["years"]
                       else "no relevant passage")
    missing = [h for h in cands if "ce" not in h]
    if missing:
        for h, s in zip(missing, rerank_scores(question, [h["prefixed"] for h in missing])):
            h["ce"] = float(s)
    supported = sorted([h for h in cands if h["ce"] >= tau], key=lambda h: -h["ce"])
    if not supported:
        return _refuse("the closest passages do not support an answer")

    types = requested_types(question)
    rows = []   # (hit, clause, typed values)
    for h in supported[:CLAUSE_PASSAGES]:
        for cl in clauses(h["text"]):
            vals = [v for v in extract_values(cl) if types is None or v["type"] in types]
            if vals or types is None:
                rows.append((h, cl, vals))
    if not rows:
        return _refuse(f"the documents do not state a {'/'.join(sorted(types))}")
    scores = rerank_scores(question, [f"{h['title']} | {h['heading']}: {cl}" for h, cl, _ in rows])
    order = sorted(range(len(rows)), key=lambda i: -scores[i])
    best_h, best_cl, best_vals = rows[order[0]]
    primary = best_vals[0] if best_vals else None

    ans = Answer(text="", primary=primary["value"] if primary else None, citations=[_cite(best_h)],
                 passages=supported[:CLAUSE_PASSAGES])

    # Number consistency across sources
    if primary:
        seen = {(primary["value"], best_h["doc_id"]): best_h}
        for i in order[1:]:
            h, cl, vals = rows[i]
            if scores[i] < scores[order[0]] - margin or h["doc_id"] == best_h["doc_id"]:
                continue
            for v in vals:
                if v["type"] == primary["type"] and v["value"] != primary["value"]:
                    seen.setdefault((v["value"], h["doc_id"]), h)
                    break
        if len({v for v, _ in seen}) > 1:
            ans.conflict = [{"value": v, "doc_id": d, "heading": h["heading"], "title": h["title"]} for (v, d), h in seen.items()]
            ans.flags.append("conflicting_sources")
            ans.citations = [_cite(h) for h in seen.values()]
        if primary.get("ambiguous"):
            ans.flags.append("ambiguous_date")

    if generator == "extractive":
        ans.text = f"« {best_cl} » — {best_h['title']}, {best_h['heading']}"
    else:
        from rag.llm import NO_ANSWER, grounded_answer
        ans.text, ungrounded = grounded_answer(question, supported[:CLAUSE_PASSAGES], generator, types)
        if ans.text.strip().upper().startswith(NO_ANSWER):   # the model found no answer in the passages
            return _refuse("the model found no answer in the passages")
        if ungrounded:   # the model wrote a number that is in no passage: fall back to the quote
            ans.flags.append("llm_number_not_in_sources")
            ans.text = f"« {best_cl} » — {best_h['title']}, {best_h['heading']}"
    if ans.conflict:
        parts = [f"{c['value']} ({c['title']}, {c['heading']})" for c in ans.conflict]
        ans.text += "\n⚠ The sources disagree: " + " vs ".join(parts) + ". Check which one applies; no value was rounded or merged."
    if "ambiguous_date" in ans.flags:
        ans.text += (f"\n⚠ '{primary['raw']}' is ambiguous: DD/MM reads {primary['value']}, "
                     "MM/DD reads the other way round. Confirm with the sender.")
    return ans
