"""Ask a question from the terminal.

  python -m app.cli --company ombrelune "Quel taux horaire avons-nous convenu avec Cabinet Vauclerc Conseil ?"
"""
import argparse

from rag import store
from rag.pipeline import ask
from rag.retrieve import MODES

COMPANIES = ["ombrelune", "quillfern", "halvardt"]


def main():
    ap = argparse.ArgumentParser(description="Ask the accounting documents of one company.")
    ap.add_argument("question")
    ap.add_argument("--company", required=True, choices=COMPANIES, help="the asking user's company (tenant)")
    ap.add_argument("--mode", default="rerank+nature", choices=MODES)
    ap.add_argument("--generator", default="extractive", choices=["extractive", "ollama", "anthropic"])
    ap.add_argument("--show", type=int, default=3, help="number of retrieved passages to print")
    args = ap.parse_args()

    ans, hits, timing = ask(store.connect(), args.company, args.question, args.mode, args.generator)
    print("\n" + ans.text)
    if not ans.refused:
        print("\nSources:")
        for c in ans.citations:
            print(f"  - {c['title']} › {c['heading']}  ({c['doc_id']})")
    if ans.flags:
        print("Flags:", ", ".join(ans.flags))
    print(f"\nRetrieved passages (top {args.show}):")
    for h in hits[: args.show]:
        print(f"  [{h['score']:.3f}] {h['doc_id']} › {h['heading']}  nature={h['nature']}")
    print(f"\n{timing['retrieval_ms']} ms retrieval, {timing['answer_ms']} ms answer")


if __name__ == "__main__":
    main()
