"""Passage nature: what kind of text a chunk is (invoice line, contract clause, tax notice,
procedure step, correspondence, party details).

At indexing time a small classifier labels every chunk. At query time a set of
multilingual keyword rules says which nature the question is after, and the ranker adds
`weight * P(wanted nature | passage)` to the cross-encoder score.

The classifier is TF-IDF on character n-grams (robust to OCR noise and to FR/EN/DE
morphology) + logistic regression. It is trained on the labelled sections of half the
documents (split by document, never by section) and evaluated on the other half."""
from __future__ import annotations

import json
import pickle
import random
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

from rag.text import fold

NATURES = ["invoice_line", "contract_clause", "tax_notice", "procedure_step", "correspondence", "party_details"]
SPLIT_SEED = 7


def _features(chunk) -> str:
    # The heading is part of the passage; the title is not, so the classifier must read the text itself.
    return f"{chunk.heading}\n{chunk.text}"


def split_docs(doc_ids: list[str]) -> tuple[set[str], set[str]]:
    ids = sorted(set(doc_ids))
    random.Random(SPLIT_SEED).shuffle(ids)
    half = len(ids) // 2
    return set(ids[:half]), set(ids[half:])


def train(chunks, labels_file: Path):
    labels = {}
    for line in labels_file.read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        labels[(x["doc_id"], x["heading"])] = x["nature"]
    labelled = [(c, labels[(c.doc_id, c.heading)]) for c in chunks if (c.doc_id, c.heading) in labels]
    train_ids, test_ids = split_docs([c.doc_id for c, _ in labelled])
    tr = [(c, y) for c, y in labelled if c.doc_id in train_ids]
    te = [(c, y) for c, y in labelled if c.doc_id in test_ids]
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1),
        LogisticRegression(max_iter=2000, C=10.0),
    )
    model.fit([_features(c) for c, _ in tr], [y for _, y in tr])
    pred = model.predict([_features(c) for c, _ in te])
    gold = [y for _, y in te]
    report = {
        "train_sections": len(tr), "test_sections": len(te),
        "accuracy": round(accuracy_score(gold, pred), 3),
        "macro_f1": round(f1_score(gold, pred, average="macro"), 3),
    }
    return model, report


def predict(model, chunks) -> list[tuple[str, dict]]:
    probas = model.predict_proba([_features(c) for c in chunks])
    out = []
    for p in probas:
        d = {cls: round(float(v), 4) for cls, v in zip(model.classes_, p)}
        out.append((max(d, key=d.get), d))
    return out


def save(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(model))


# ----------------------------------------------------------------------------- query side
_QUERY_RULES = [   # first match wins; patterns run on accent-folded lower-case text
    ("party_details", r"emise|issued|ausgestellt"),
    ("invoice_line", r"facture|invoice|rechnung|prix unitaire|unit price|einzelpreis"),
    ("tax_notice", r"\btva\b|\bvat\b|umsatzsteuer|impot|\btax\b|steuer|declaration|\breturn\b|avance|advance|vorauszahlung"),
    ("procedure_step", r"note de frais|expense|reisekosten|rapprochement|reconciliation|bankabstimmung|justificatif|receipt|beleg"),
    ("contract_clause", r"contrat|contract|agreement|vertrag|loyer|\brent\b|miete|garantie|deposit|kaution|preavis|notice|kundigung|taux horaire|hourly rate|stundensatz"),
    ("correspondence", r"reunion|meeting|termin|livraison|delivery|lieferung|offre|quoted|angeboten"),
]


def wanted_nature(question: str) -> str | None:
    q = fold(question)
    for nature, pat in _QUERY_RULES:
        if re.search(pat, q):
            return nature
    return None
