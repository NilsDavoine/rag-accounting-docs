"""Fast tests: no database, no model."""
import math

import numpy as np
import pytest

from rag import answer as answer_mod
from rag.bm25 import BM25
from rag.chunking import chunk_document, parse_document
from rag.retrieve import rrf
from rag.text import anchors, extract_values, requested_types, satisfies, tokenize


def vals(text, typ=None):
    return [v["value"] for v in extract_values(text) if typ is None or v["type"] == typ]


@pytest.mark.parametrize("text,expected", [
    ("Total TTC : 2 053,35 EUR.", ["2053.35"]),
    ("Gesamtbetrag brutto: 9.660,99 EUR", ["9660.99"]),
    ("Total due: EUR 2,053.35.", ["2053.35"]),
    ("facturées 26 EUR de l'heure", ["26.00"]),
    ("Quartal 2025 9.660,99 EUR", ["9660.99"]),   # a year right before an amount is not a thousands group
])
def test_amounts_in_three_locales(text, expected):
    assert vals(text, "amount") == expected


def test_dates_survive_accents_and_ocr_noise():
    assert vals("au plus tard le 22 août 2025", "date") == ["2025-08-22"]
    assert vals("le 1er avri1 2025", "date") == ["2025-04-01"]          # scanned: l read as 1
    assert vals("spätestens am 15.08.2025", "date") == ["2025-08-15"]
    assert vals("on 3 April 2025", "date") == ["2025-04-03"]


def test_slash_date_is_flagged_ambiguous_only_when_it_is():
    amb = [v for v in extract_values("Delivery on 04/03/2025") if v["type"] == "date"][0]
    assert amb["ambiguous"] is True
    clear = [v for v in extract_values("Delivery on 25/03/2025") if v["type"] == "date"][0]
    assert clear["ambiguous"] is False and clear["value"] == "2025-03-25"


def test_percent_and_duration_are_not_amounts():
    v = extract_values("TVA à 17 % : 298,35 EUR. Préavis de 3 mois.")
    assert [(x["type"], x["value"]) for x in v] == [("percent", "17%"), ("amount", "298.35"), ("duration", "3 month")]


def test_requested_types():
    assert requested_types("How many months of rent is the office security deposit?") == {"duration"}
    assert requested_types("Wie viel Monatsmiete zahlen wir?") == {"amount"}
    assert requested_types("Bis wann muss die Rechnung bezahlt werden?") == {"date"}
    assert requested_types("Wie hoch ist die Vertragsstrafe?") == {"amount", "percent"}
    assert requested_types("What is Pellow Office Goods's IBAN?") == {"iban"}


def test_anchors_block_wrong_period_and_wrong_invoice():
    a = anchors("How much VAT do we owe for Q4 2024?")
    assert a["years"] == ["2024"] and a["quarters"] == [4]
    assert not satisfies(a, "VAT due for the second quarter of 2025 amounts to EUR 1,000.00")
    a = anchors("Wie viel Umsatzsteuer schulden wir für das zweite Quartal 2025?")
    assert satisfies(a, "la TVA due pour le deuxième trimestre 2025 s'élève à 1 000,00 EUR")
    a = anchors("What is the total of invoice TAR-2025-820?")
    assert satisfies(a, "Facture n° TAR-2025-820") and not satisfies(a, "Facture n° TAR-2025-821")


def test_tokenizer_keeps_identifiers_and_drops_stopwords():
    t = tokenize("Quel est le montant de la facture TAR-2025-820 ?")
    assert "tar-2025-820" in t and "montant" in t and "le" not in t


def test_bm25_reference_formula():
    docs = [["vat", "due"], ["rent", "rent", "office"], ["vat", "rate"]]
    bm = BM25(docs)
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 3 / bm.avgdl))
    assert bm.score(["rent"], 1) == pytest.approx(expected)
    assert [i for i, _ in bm.top(["vat"], 5)] in ([0, 2], [2, 0])


def test_rrf_rewards_agreement():
    a = [{"id": 1}, {"id": 2}, {"id": 3}]
    b = [{"id": 3}, {"id": 1}]
    fused = rrf([a, b], k=60)
    assert [h["id"] for h in fused] == [1, 3, 2]
    assert fused[0]["score"] == pytest.approx(1 / 61 + 1 / 62)


RAW = """---
doc_id: x-inv-01
company: x
title: Invoice A-1
type: invoice
source: pdf
lang: en
date: 2025-01-02
---
## Header
Supplier: S.

## Totals
Total due: EUR 10.00.
"""


def test_section_chunking_with_context_prefix():
    chunks = chunk_document(parse_document(RAW))
    assert [c.heading for c in chunks] == ["Header", "Totals"]
    assert chunks[1].prefixed.startswith("[Invoice A-1 | x | 2025-01-02 | Totals]")


# ----------------------------------------------------------------------------- guardrails
def hit(doc, heading, text, ce):
    return {"id": hash((doc, heading)) % 10_000, "doc_id": doc, "heading": heading, "title": doc, "text": text,
            "prefixed": f"[{doc} | 2025]\n{text}", "ce": ce}


@pytest.fixture
def stub_reranker(monkeypatch):
    """Score clauses by keyword overlap so guardrail logic is tested without a model."""
    def fake(query, texts):
        q = set(query.lower().split())
        return np.array([len(q & set(t.lower().split())) / 10 for t in texts])
    monkeypatch.setattr(answer_mod, "rerank_scores", fake)


def test_conflicting_sources_are_reported_not_rounded(stub_reranker):
    hits = [hit("contract", "Fees", "The hourly rate is EUR 25.50 for services.", 0.9),
            hit("email", "Message", "As agreed the hourly rate is 26 EUR for services.", 0.8)]
    a = answer_mod.answer("What hourly rate for services?", hits, tau=0.5, margin=0.5)
    assert {c["value"] for c in a.conflict} == {"25.50", "26.00"}
    assert "25.00" not in a.text and "disagree" in a.text


def test_same_value_in_two_sources_is_not_a_conflict(stub_reranker):
    hits = [hit("invoice", "Totals", "Total due is EUR 120.00 now.", 0.9),
            hit("reminder", "Message", "Total due is 120,00 EUR now.", 0.8)]
    a = answer_mod.answer("What total is due now?", hits, tau=0.5, margin=0.5)
    assert a.primary == "120.00" and not a.conflict


def test_refuses_below_support_threshold(stub_reranker):
    a = answer_mod.answer("What is the rent?", [hit("lease", "Rent", "Rent is EUR 900.00.", 0.05)], tau=0.2, margin=0.1)
    assert a.refused


def test_refuses_when_requested_type_is_absent(stub_reranker):
    a = answer_mod.answer("What is the supplier's IBAN?", [hit("inv", "Header", "Supplier: S. Total EUR 9.00.", 0.9)],
                          tau=0.2, margin=0.1)
    assert a.refused and "iban" in a.reason


def test_refuses_when_identifier_is_absent(stub_reranker):
    a = answer_mod.answer("What is the total of invoice ABC-2025-111?",
                          [hit("inv", "Totals", "Invoice ABC-2025-222. Total EUR 9.00.", 0.99)], tau=0.2, margin=0.1)
    assert a.refused


def test_ambiguous_date_is_flagged(stub_reranker):
    a = answer_mod.answer("When is the delivery scheduled?",
                          [hit("mail", "Message", "Delivery is scheduled for 04/03/2025.", 0.9)], tau=0.2, margin=0.1)
    assert "ambiguous_date" in a.flags and "04/03/2025" in a.text
