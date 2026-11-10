"""Held-out HARD set: a second invented corpus and 80 questions, generated with another seed.

Nothing is tuned on it. The support threshold, the conflict margin, the nature weight and every
rule in rag/ were fixed on the dev split of the main bench; eval/run_eval.py only measures here.

What makes it harder than the main corpus
  * tables: invoices whose line items sit in a text table, with the currency only in the column
    header on some of them;
  * heavier scan noise: o read as 0, e as c, i as l, merged and split words, line breaks inside
    sentences, a space slipped before the decimal separator of an amount. Digits themselves are
    never changed;
  * long documents with close sections: a framework agreement with three payment terms and two
    hourly rates, a VAT statement with one section per quarter, an expense policy with domestic and
    foreign ceilings. Long sections are split into several chunks, sometimes with the fact in the
    second one;
  * wording the main bench never uses, in the documents ("trois mois" in letters, other invoice
    numbering schemes) and in the questions (colloquial, lower-case, typos);
  * unanswerable questions that are near misses: a quarter the statement does not cover, a
    service line the agreement does not mention, another company's invoice.

Outputs
  data/corpus_hard/<company>/<doc_id>.txt   documents (same front-matter format as the main corpus)
  data/nature_labels_hard.jsonl             section labels, used ONLY to report how the classifier
                                            trained on the main corpus transfers; never for training
  eval/questions_hard.jsonl                 80 questions, 16 without an answer

Run:  python -m data.generate_hard
"""
from __future__ import annotations

import datetime as dt
import json
import random
import re
from pathlib import Path

from data.generate_corpus import MONTHS, fmt_amount, fmt_date, fmt_dur

SEED = 20261110            # the main corpus uses 20251110
ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus_hard"
LABELS = ROOT / "data" / "nature_labels_hard.jsonl"
QUESTIONS = ROOT / "eval" / "questions_hard.jsonl"
N_ANSWERABLE_FACTS = 58    # + 3 conflicts + 3 ambiguous dates + 16 unanswerable = 80

COMPANIES = {
    "brumeval": {"name": "Brumeval Transports S.à r.l.", "lang": "fr"},
    "larkwhistle": {"name": "Larkwhistle Analytics Ltd", "lang": "en"},
    "feldkamm": {"name": "Feldkamm Haustechnik GmbH", "lang": "de"},
}

# (supplier, language, invoice-number format) for the two table invoices of each company
INVOICE_SUPPLIERS = {
    "brumeval": [("Garage Velmont", "fr", "GV/{y}/{n:04d}"), ("Tyrewell Parts Ltd", "en", "TWP-{y}-{n:03d}")],
    "larkwhistle": [("Hollowmere Office Supply", "en", "HOS-{y}-{n:03d}"), ("Datenwerk Reutlach GmbH", "de", "RE {y}.{n:04d}")],
    "feldkamm": [("Sanitär Großhandel Ebbing", "de", "SGE-{y}-{n:03d}"), ("Comptoir Lavergne", "fr", "CL{y}{n:04d}")],
}
FRAMEWORK_PROVIDER = {"brumeval": "Conseil Arpentin", "larkwhistle": "Penhallow Consulting", "feldkamm": "Beratung Wolkenstein"}
PARTS_SENDER = {"brumeval": "Tyrewell Parts Ltd", "larkwhistle": "Kessington Hardware Inc.", "feldkamm": "Ridgefield Pumps Inc."}

ITEMS = [  # label per language, price range
    ({"fr": "Plaquettes de frein", "en": "Brake pads", "de": "Bremsbeläge"}, 40, 90),
    ({"fr": "Pneus poids lourd", "en": "Truck tyres", "de": "Lkw-Reifen"}, 280, 520),
    ({"fr": "Filtres à huile", "en": "Oil filters", "de": "Ölfilter"}, 12, 35),
    ({"fr": "Main-d'œuvre atelier", "en": "Workshop labour", "de": "Werkstattarbeit"}, 65, 110),
    ({"fr": "Classeurs à levier", "en": "Lever-arch files", "de": "Ordner"}, 3, 8),
    ({"fr": "Écrans 27 pouces", "en": "27-inch monitors", "de": "27-Zoll-Monitore"}, 210, 390),
    ({"fr": "Licences de sauvegarde", "en": "Backup licences", "de": "Backup-Lizenzen"}, 45, 120),
    ({"fr": "Robinets thermostatiques", "en": "Thermostatic valves", "de": "Thermostatventile"}, 18, 46),
    ({"fr": "Tubes cuivre 15 mm", "en": "15 mm copper pipes", "de": "Kupferrohre 15 mm"}, 9, 24),
    ({"fr": "Pompes de circulation", "en": "Circulation pumps", "de": "Umwälzpumpen"}, 160, 340),
]
ITEM_SETS = {"brumeval": [0, 1, 2, 3], "larkwhistle": [4, 5, 6, 3], "feldkamm": [7, 8, 9, 3]}

ORD = {"fr": ["premier", "deuxième", "troisième"], "en": ["first", "second", "third"], "de": ["ersten", "zweiten", "dritten"]}

FILLER = {
    "fr": ["Les parties conviennent d'exécuter le présent article de bonne foi et dans le respect des usages de la profession.",
           "Toute modification du présent article fait l'objet d'un avenant écrit signé par les deux parties.",
           "Les stipulations du présent article s'appliquent sans préjudice des dispositions légales impératives.",
           "Aucune tolérance de l'une des parties ne vaut renonciation à se prévaloir ultérieurement du présent article.",
           "Les échanges relatifs au présent article peuvent avoir lieu par courrier électronique, sauf disposition contraire.",
           "Le prestataire tient à la disposition du client tout justificatif utile à la vérification des montants facturés.",
           "Le présent article est interprété à la lumière des autres stipulations du contrat-cadre et de ses annexes.",
           "En cas de difficulté d'application, les parties se rencontrent dans les meilleurs délais pour rechercher une solution amiable.",
           "Les titres des articles n'ont qu'une valeur indicative et ne sauraient en limiter la portée.",
           "Chaque partie désigne un interlocuteur chargé du suivi du présent article et en informe l'autre partie par écrit."],
    "en": ["The parties agree to perform this clause in good faith and in line with professional practice.",
           "Any change to this clause requires a written amendment signed by both parties.",
           "This clause applies without prejudice to any mandatory statutory provisions.",
           "No forbearance by either party shall be construed as a waiver of its rights under this clause.",
           "Communications about this clause may be exchanged by e-mail unless stated otherwise.",
           "The provider keeps available to the client any document needed to check the amounts invoiced.",
           "This clause is to be read together with the other terms of the framework agreement and its schedules.",
           "Should a difficulty arise in applying this clause, the parties shall meet promptly to seek an amicable solution.",
           "Headings are for convenience only and do not limit the scope of this clause.",
           "Each party appoints a contact person in charge of this clause and informs the other party in writing."],
    "de": ["Die Parteien verpflichten sich, diese Bestimmung nach Treu und Glauben und gemäß den Gepflogenheiten des Berufsstands zu erfüllen.",
           "Änderungen dieser Bestimmung bedürfen eines schriftlichen Nachtrags, der von beiden Parteien unterzeichnet wird.",
           "Diese Bestimmung gilt unbeschadet zwingender gesetzlicher Vorschriften.",
           "Eine Duldung durch eine Partei gilt nicht als Verzicht auf ihre Rechte aus dieser Bestimmung.",
           "Mitteilungen zu dieser Bestimmung können per E-Mail erfolgen, sofern nichts anderes vereinbart ist.",
           "Der Auftragnehmer hält alle Unterlagen bereit, die zur Prüfung der berechneten Beträge erforderlich sind.",
           "Diese Bestimmung ist im Zusammenhang mit den übrigen Regelungen des Rahmenvertrags und seiner Anlagen auszulegen.",
           "Treten bei der Anwendung dieser Bestimmung Schwierigkeiten auf, suchen die Parteien zeitnah eine einvernehmliche Lösung.",
           "Überschriften dienen nur der Orientierung und schränken den Inhalt dieser Bestimmung nicht ein.",
           "Jede Partei benennt eine Ansprechperson für diese Bestimmung und teilt sie der anderen Partei schriftlich mit."],
}


# --------------------------------------------------------------------------- formatting
def fmt_num(lang: str, x: float) -> str:
    """An amount without its currency, as it appears in a table cell."""
    return fmt_amount(lang, x).replace(" EUR", "").replace("EUR ", "")


def month_name(lang: str, m: int) -> str:
    return MONTHS[lang][m - 1]


CONFUSIONS = [("o", "0"), ("e", "c"), ("i", "l"), ("l", "1"), ("m", "rn"), ("h", "b"), ("S", "5"), ("a", "o")]


def heavy_ocr(rng: random.Random, text: str) -> str:
    """A poor scan. Letters are confused, words merged or split, lines broken mid-sentence, and a
    space can slip in before the decimal separator of an amount. Digits are never changed."""
    words = text.replace(".. ", ". ").split(" ")
    out: list[str] = []
    glue = False
    for w in words:
        if any(c.isdigit() for c in w):
            if rng.random() < 0.15:
                w = re.sub(r"([.,])(\d{2})\b", r" \1\2", w, count=1)
        elif rng.random() < 0.40:
            k = rng.random()
            if k < 0.25:
                w = w.translate(str.maketrans("éèêàùçäöüßÉ", "eeeaucaousE"))
            elif k < 0.6:
                a, b = rng.choice(CONFUSIONS)
                w = w.replace(a, b, 1)
            elif k < 0.75 and len(w) > 7:
                w = w[: len(w) // 2] + " " + w[len(w) // 2:]
            elif k < 0.9:
                if out:
                    out[-1] += w
                    glue = True
            elif len(w) > 6:
                w = w[: len(w) // 2] + "-\n" + w[len(w) // 2:]
        if glue:
            glue = False
            continue
        out.append(w)
    return "".join(w + ("\n" if rng.random() < 0.06 else " ") for w in out).strip()


def style_question(rng: random.Random, q: str) -> str:
    """Typed by a busy person: sometimes lower-case without a question mark, sometimes one typo."""
    if rng.random() < 0.25:
        q = q.rstrip(" ?").lower()
    if rng.random() < 0.15:
        words = q.split(" ")
        idx = [i for i, w in enumerate(words) if w.isalpha() and len(w) > 5]
        if idx:
            i = rng.choice(idx)
            w = words[i]
            j = rng.randrange(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            q = " ".join(words)
    return q


# --------------------------------------------------------------------------- generator
class HardGenerator:
    def __init__(self, seed: int = SEED):
        self.rng = random.Random(seed)
        self.docs: list[dict] = []
        self.labels: list[dict] = []
        self.facts: list[dict] = []
        self.extra_q: list[dict] = []
        self.invoice_numbers: dict[str, list[str]] = {c: [] for c in COMPANIES}

    def add_doc(self, company, doc_id, title, dtype, source, lang, date, sections, header=None):
        meta = {"doc_id": doc_id, "company": company, "title": title, "type": dtype,
                "source": source, "lang": lang, "date": date.isoformat()}
        parts = ["\n".join(header) + "\n"] if header else []
        for heading, text, nature in sections:
            text = text.replace(".. ", ". ")
            if source == "scan":
                text = heavy_ocr(self.rng, text)
            parts.append(text if heading == "Message" else f"## {heading}\n{text}")
            self.labels.append({"doc_id": doc_id, "heading": heading, "nature": nature})
        self.docs.append({"doc_id": doc_id, "company": company, "meta": meta, "body": "\n\n".join(parts)})

    def fact(self, company, kind, doc_id, heading, vtype, expected, **slots):
        self.facts.append({"company": company, "kind": kind, "doc_id": doc_id, "heading": heading,
                           "type": vtype, "expected": expected, "slots": slots})

    def filler(self, lang: str, n: int) -> str:
        return " ".join(self.rng.sample(FILLER[lang], n))

    def email(self, company, doc_id, sender, subject, lang, date, body):
        header = [f"From: {sender}", f"To: {COMPANIES[company]['name']}", f"Date: {fmt_date(lang, date)}",
                  f"Subject: {subject}"]
        self.add_doc(company, doc_id, subject, "email", "email", lang, date, [("Message", body, "correspondence")], header)

    # -- invoices with a line-item table
    def table_invoices(self, company):
        cname = COMPANIES[company]["name"]
        for i, (sup, lang, numfmt) in enumerate(INVOICE_SUPPLIERS[company]):
            idx = self.rng.sample(ITEM_SETS[company], 3)
            lines = []
            for k in idx:
                label, lo, hi = ITEMS[k]
                qty = self.rng.randint(2, 24)
                price = round(self.rng.uniform(lo, hi) * 4) / 4
                lines.append((label, qty, price, round(qty * price, 2)))
            net = round(sum(x[3] for x in lines), 2)
            vat = round(net * 0.17, 2)
            total = round(net + vat, 2)
            date = dt.date(2025, self.rng.randint(1, 10), self.rng.randint(1, 26))
            term = self.rng.choice([14, 21, 30])
            due = date + dt.timedelta(days=term)
            num = numfmt.format(y=2025, n=self.rng.randint(100, 999))
            self.invoice_numbers[company].append(num)
            doc_id = f"{company}-tinv-{i + 1}"
            scan = i == 1
            N = lambda x: fmt_num(lang, x)
            hdr = {"fr": ("Pos.", "Désignation", "Qté", "PU HT (EUR)", "Montant HT (EUR)"),
                   "en": ("No.", "Description", "Qty", "Unit price (EUR)", "Amount (EUR)"),
                   "de": ("Pos.", "Bezeichnung", "Menge", "Einzelpreis (EUR)", "Betrag (EUR)")}[lang]
            if scan:   # a scanned table: columns aligned with spaces, no rules
                rows = ["   ".join(hdr)] + [f"{n}   {lab[lang]}   {q}   {N(p)}   {N(a)}" for n, (lab, q, p, a) in enumerate(lines, 1)]
            else:
                rows = ["| " + " | ".join(hdr) + " |", "|" + "---|" * 5] + \
                       [f"| {n} | {lab[lang]} | {q} | {N(p)} | {N(a)} |" for n, (lab, q, p, a) in enumerate(lines, 1)]
            table = "\n".join(rows)
            if scan:   # totals as table rows: the currency is only in the label
                totals = {"fr": f"Total HT (EUR)   {N(net)}\nTVA 17 % (EUR)   {N(vat)}\nNet à payer (EUR)   {N(total)}\n"
                                f"Règlement à {term} jours, soit au plus tard le {fmt_date('fr', due)}.",
                          "en": f"Subtotal (EUR)   {N(net)}\nVAT 17% (EUR)   {N(vat)}\nAmount payable (EUR)   {N(total)}\n"
                                f"Terms: {term} days net, i.e. by {fmt_date('en', due)}.",
                          "de": f"Zwischensumme (EUR)   {N(net)}\nUSt. 17 % (EUR)   {N(vat)}\nZahlbetrag (EUR)   {N(total)}\n"
                                f"Zahlbar innerhalb von {term} Tagen, d. h. bis {fmt_date('de', due)}."}[lang]
            else:
                A = lambda x: fmt_amount(lang, x)
                totals = {"fr": f"Total HT : {A(net)}. TVA 17 % : {A(vat)}. Net à payer : {A(total)}. "
                                f"Règlement à {term} jours, soit au plus tard le {fmt_date('fr', due)}.",
                          "en": f"Subtotal: {A(net)}. VAT 17%: {A(vat)}. Amount payable: {A(total)}. "
                                f"Terms: {term} days net, i.e. by {fmt_date('en', due)}.",
                          "de": f"Zwischensumme: {A(net)}. USt. 17 %: {A(vat)}. Zahlbetrag: {A(total)}. "
                                f"Zahlbar innerhalb von {term} Tagen, d. h. bis {fmt_date('de', due)}."}[lang]
            h1, h2, h3, word = {"fr": ("Émetteur et destinataire", "Lignes de facture", "Récapitulatif", "Facture"),
                                "en": ("Billing details", "Items", "Summary", "Invoice"),
                                "de": ("Rechnungsangaben", "Positionen", "Zusammenfassung", "Rechnung")}[lang]
            head = {"fr": f"{sup}, fournisseur. Facturé à : {cname}. Facture {num} du {fmt_date('fr', date)}.",
                    "en": f"From {sup}. Bill to: {cname}. Invoice {num}, dated {fmt_date('en', date)}.",
                    "de": f"Von {sup}. Rechnungsempfänger: {cname}. Rechnung {num} vom {fmt_date('de', date)}."}[lang]
            # tables are not passed through the word-level noise (it would merge cells); headers and totals are
            sections = [(h1, head, "party_details"), (h2, table, "invoice_line"), (h3, totals, "invoice_line")]
            meta_src = "scan" if scan else "pdf"
            parts_noise = []
            for heading, text, nature in sections:
                if scan and heading != h2:
                    text = heavy_ocr(self.rng, text)
                parts_noise.append((heading, text, nature))
            self.add_doc(company, doc_id, f"{word} {num} – {sup}", "invoice", "pdf", lang, date, parts_noise)
            self.docs[-1]["meta"]["source"] = meta_src
            label, qty, price, _ = lines[self.rng.randrange(3)]
            slots = dict(num=num, sup=sup, month=date.month, item=label)
            self.fact(company, "inv_price", doc_id, h2, "amount", f"{price:.2f}", **slots)
            self.fact(company, "inv_total", doc_id, h3, "amount", f"{total:.2f}", **slots)
            self.fact(company, "inv_due", doc_id, h3, "date", due.isoformat(), **slots)

    # -- annual VAT statement: one close section per quarter, scanned
    def vat_statement(self, company):
        lang = COMPANIES[company]["lang"]
        amounts = [round(self.rng.uniform(3000, 18000), 2) for _ in range(3)]
        credit = round(self.rng.uniform(150, 900), 2)
        deadline = dt.date(2025, 11, 17)
        paid = [dt.date(2025, 5, 12), dt.date(2025, 8, 14)]
        A = lambda x: fmt_amount(lang, x)
        D = lambda x: fmt_date(lang, x)
        cname = COMPANIES[company]["name"]
        if lang == "fr":
            secs = [("Destinataire", f"Administration fiscale fictive, service TVA. Relevé adressé à {cname}.", "party_details"),
                    ("Premier trimestre 2025", f"Pour le premier trimestre 2025, la taxe exigible ressort à {A(amounts[0])}. Ce montant a été réglé le {D(paid[0])}.", "tax_notice"),
                    ("Deuxième trimestre 2025", f"Pour le deuxième trimestre 2025, la taxe exigible ressort à {A(amounts[1])}. Ce montant a été réglé le {D(paid[1])}.", "tax_notice"),
                    ("Troisième trimestre 2025", f"Pour le troisième trimestre 2025, la taxe exigible ressort à {A(amounts[2])}. Ce montant reste à verser au plus tard le {D(deadline)}.", "tax_notice"),
                    ("Régularisations", f"Un crédit de {A(credit)} issu de l'exercice 2024 a été imputé sur le premier trimestre. Aucune autre régularisation n'est intervenue.", "tax_notice"),
                    ("Voies de recours", "Une réclamation peut être introduite dans un délai de trois mois à compter de la réception du présent relevé.", "tax_notice")]
            title = "Relevé annuel de TVA 2025"
        elif lang == "en":
            secs = [("Addressee", f"Fictional Tax Administration, VAT office. Statement sent to {cname}.", "party_details"),
                    ("First quarter 2025", f"For the first quarter of 2025, the tax payable comes to {A(amounts[0])}. This amount was settled on {D(paid[0])}.", "tax_notice"),
                    ("Second quarter 2025", f"For the second quarter of 2025, the tax payable comes to {A(amounts[1])}. This amount was settled on {D(paid[1])}.", "tax_notice"),
                    ("Third quarter 2025", f"For the third quarter of 2025, the tax payable comes to {A(amounts[2])}. This amount is still to be paid by {D(deadline)}.", "tax_notice"),
                    ("Adjustments", f"A credit of {A(credit)} carried over from 2024 was set against the first quarter. No other adjustment was made.", "tax_notice"),
                    ("Right of objection", "An objection may be lodged within 3 months of receiving this statement.", "tax_notice")]
            title = "Annual VAT statement 2025"
        else:
            secs = [("Empfänger", f"Fiktive Steuerverwaltung, Umsatzsteuerstelle. Kontoauszug für {cname}.", "party_details"),
                    ("Erstes Quartal 2025", f"Für das erste Quartal 2025 ergibt sich eine Steuer von {A(amounts[0])}. Der Betrag wurde am {D(paid[0])} beglichen.", "tax_notice"),
                    ("Zweites Quartal 2025", f"Für das zweite Quartal 2025 ergibt sich eine Steuer von {A(amounts[1])}. Der Betrag wurde am {D(paid[1])} beglichen.", "tax_notice"),
                    ("Drittes Quartal 2025", f"Für das dritte Quartal 2025 ergibt sich eine Steuer von {A(amounts[2])}. Der Betrag ist bis spätestens {D(deadline)} zu entrichten.", "tax_notice"),
                    ("Berichtigungen", f"Ein Guthaben von {A(credit)} aus 2024 wurde mit dem ersten Quartal verrechnet. Weitere Berichtigungen erfolgten nicht.", "tax_notice"),
                    ("Rechtsbehelf", "Ein Einspruch ist innerhalb von 3 Monaten nach Erhalt dieses Auszugs möglich.", "tax_notice")]
            title = "Umsatzsteuer-Jahresauszug 2025"
        doc_id = f"{company}-tax-annual"
        self.add_doc(company, doc_id, title, "tax_letter", "scan", lang, dt.date(2025, 10, 20), secs)
        for q in range(3):
            self.fact(company, f"vat_q{q + 1}", doc_id, secs[q + 1][0], "amount", f"{amounts[q]:.2f}", quarter=q + 1)
        self.fact(company, "vat_q3_due", doc_id, secs[3][0], "date", deadline.isoformat())
        self.fact(company, "vat_objection", doc_id, secs[5][0], "duration", "3 month")

    # -- long framework agreement with close articles, plus an amendment that changes one rate
    def framework(self, company):
        lang = COMPANIES[company]["lang"]
        sup = FRAMEWORK_PROVIDER[company]
        cname = COMPANIES[company]["name"]
        junior = float(self.rng.choice(range(80, 110, 5)))
        senior = float(self.rng.choice(range(130, 170, 5)))
        new_senior = senior + 10
        pay = dict(zip(["consulting", "licences", "maintenance"], self.rng.sample([15, 30, 45, 60], 3)))
        conf = self.rng.choice([3, 5, 7])
        cap = float(self.rng.choice([50000, 75000, 100000, 150000]))
        notice = self.rng.choice([2, 4, 6])
        start = dt.date(2025, self.rng.randint(1, 6), 1)
        A = lambda x: fmt_amount(lang, x)
        F = lambda n: self.filler(lang, n)
        if lang == "fr":
            arts = [("Article 1 – Objet", f"Le présent contrat-cadre définit les conditions dans lesquelles {sup} réalise pour {cname} des missions de conseil, fournit des licences et assure la maintenance des outils comptables. {F(2)}"),
                    ("Article 2 – Entrée en vigueur", f"{F(7)} Le contrat-cadre entre en vigueur le {fmt_date(lang, start)} et se poursuit tant qu'il n'est pas dénoncé."),
                    ("Article 3 – Tarif des consultants juniors", f"Les interventions d'un consultant junior sont facturées {A(junior)} hors taxes par heure. {F(1)}"),
                    ("Article 4 – Tarif des consultants seniors", f"Les interventions d'un consultant senior sont facturées {A(senior)} hors taxes par heure. {F(1)}"),
                    ("Article 5 – Règlement des missions de conseil", f"Les factures relatives aux missions de conseil sont réglées à {fmt_dur(lang, pay['consulting'], 'day')} fin de mois. {F(1)}"),
                    ("Article 6 – Règlement des licences", f"Les factures relatives aux licences sont réglées à {fmt_dur(lang, pay['licences'], 'day')} fin de mois. {F(1)}"),
                    ("Article 7 – Règlement de la maintenance", f"Les factures relatives à la maintenance sont réglées à {fmt_dur(lang, pay['maintenance'], 'day')} fin de mois. {F(1)}"),
                    ("Article 8 – Confidentialité", f"{F(8)} L'obligation de confidentialité survit {fmt_dur(lang, conf, 'year')} à l'expiration du contrat-cadre, quelle qu'en soit la cause."),
                    ("Article 9 – Responsabilité", f"La responsabilité totale de {sup} au titre du contrat-cadre est plafonnée à {A(cap)}. {F(3)}"),
                    ("Article 10 – Résiliation", f"{F(2)} Chaque partie peut mettre fin au contrat-cadre en respectant un préavis de {fmt_dur(lang, notice, 'month')}."),
                    ("Article 11 – Droit applicable", F(6))]
            title, atitle = f"Contrat-cadre de services – {sup}", f"Avenant n° 1 au contrat-cadre – {sup}"
            amend = [("Article unique – Révision tarifaire", f"À compter du 1er septembre 2025, le tarif horaire d'un consultant senior est porté à {A(new_senior)} hors taxes. Les autres stipulations du contrat-cadre restent inchangées.")]
        elif lang == "en":
            arts = [("Clause 1 – Scope", f"This framework agreement sets the terms on which {sup} provides {cname} with advisory assignments, software licences and maintenance of its accounting tools. {F(2)}"),
                    ("Clause 2 – Commencement", f"{F(7)} This framework agreement commences on {fmt_date(lang, start)} and continues until terminated."),
                    ("Clause 3 – Junior consultant rate", f"Work by a junior consultant is charged at {A(junior)} per hour, excluding VAT. {F(1)}"),
                    ("Clause 4 – Senior consultant rate", f"Work by a senior consultant is charged at {A(senior)} per hour, excluding VAT. {F(1)}"),
                    ("Clause 5 – Settlement of advisory invoices", f"Invoices for advisory assignments are settled {fmt_dur(lang, pay['consulting'], 'day')} after month end. {F(1)}"),
                    ("Clause 6 – Settlement of licence invoices", f"Invoices for licences are settled {fmt_dur(lang, pay['licences'], 'day')} after month end. {F(1)}"),
                    ("Clause 7 – Settlement of maintenance invoices", f"Invoices for maintenance are settled {fmt_dur(lang, pay['maintenance'], 'day')} after month end. {F(1)}"),
                    ("Clause 8 – Confidentiality", f"{F(8)} The duty of confidentiality survives for {fmt_dur(lang, conf, 'year')} after this framework agreement ends, for whatever reason."),
                    ("Clause 9 – Liability", f"The aggregate liability of {sup} under this framework agreement is capped at {A(cap)}. {F(3)}"),
                    ("Clause 10 – Termination", f"{F(2)} Either party may end this framework agreement by giving {fmt_dur(lang, notice, 'month')}' written notice."),
                    ("Clause 11 – Governing law", F(6))]
            title, atitle = f"Framework services agreement – {sup}", f"Amendment No. 1 to the framework agreement – {sup}"
            amend = [("Sole clause – Rate review", f"With effect from 1 September 2025, the hourly rate of a senior consultant is raised to {A(new_senior)}, excluding VAT. All other terms of the framework agreement remain unchanged.")]
        else:
            arts = [("§ 1 Gegenstand", f"Dieser Rahmenvertrag regelt, wie {sup} für die {cname} Beratungsaufträge ausführt, Lizenzen bereitstellt und die Buchhaltungswerkzeuge wartet. {F(2)}"),
                    ("§ 2 Inkrafttreten", f"{F(7)} Der Rahmenvertrag tritt am {fmt_date(lang, start)} in Kraft und läuft bis zu seiner Kündigung."),
                    ("§ 3 Vergütung Junior-Berater", f"Einsätze eines Junior-Beraters werden mit {A(junior)} netto je Stunde abgerechnet. {F(1)}"),
                    ("§ 4 Vergütung Senior-Berater", f"Einsätze eines Senior-Beraters werden mit {A(senior)} netto je Stunde abgerechnet. {F(1)}"),
                    ("§ 5 Zahlung Beratungsleistungen", f"Rechnungen über Beratungsleistungen sind {fmt_dur(lang, pay['consulting'], 'day')} nach Monatsende zu begleichen. {F(1)}"),
                    ("§ 6 Zahlung Lizenzen", f"Rechnungen über Lizenzen sind {fmt_dur(lang, pay['licences'], 'day')} nach Monatsende zu begleichen. {F(1)}"),
                    ("§ 7 Zahlung Wartung", f"Rechnungen über Wartung sind {fmt_dur(lang, pay['maintenance'], 'day')} nach Monatsende zu begleichen. {F(1)}"),
                    ("§ 8 Vertraulichkeit", f"{F(8)} Die Verschwiegenheitspflicht gilt nach Beendigung des Rahmenvertrags, gleich aus welchem Grund, noch {fmt_dur(lang, conf, 'year')} fort."),
                    ("§ 9 Haftung", f"Die Gesamthaftung von {sup} aus diesem Rahmenvertrag ist auf {A(cap)} begrenzt. {F(3)}"),
                    ("§ 10 Kündigung", f"{F(2)} Jede Partei kann den Rahmenvertrag mit einer Frist von {fmt_dur(lang, notice, 'month')} beenden."),
                    ("§ 11 Anwendbares Recht", F(6))]
            title, atitle = f"Rahmendienstleistungsvertrag – {sup}", f"Nachtrag Nr. 1 zum Rahmenvertrag – {sup}"
            amend = [("Einziger Paragraph – Preisanpassung", f"Ab dem 1. September 2025 beträgt der Stundensatz eines Senior-Beraters {A(new_senior)} netto. Alle übrigen Bestimmungen des Rahmenvertrags bleiben unverändert.")]
        doc_id, amend_id = f"{company}-framework", f"{company}-framework-amendment"
        self.add_doc(company, doc_id, title, "contract", "pdf", lang, start - dt.timedelta(days=15),
                     [(h, t, "contract_clause") for h, t in arts])
        self.add_doc(company, amend_id, atitle, "contract", "pdf", lang, dt.date(2025, 8, 18),
                     [(h, t, "contract_clause") for h, t in amend])
        h = [a[0] for a in arts]
        s = dict(sup=sup)
        self.fact(company, "fw_start", doc_id, h[1], "date", start.isoformat(), **s)
        self.fact(company, "fw_junior", doc_id, h[2], "amount", f"{junior:.2f}", **s)
        self.fact(company, "fw_pay_consulting", doc_id, h[4], "duration", f"{pay['consulting']} day", **s)
        self.fact(company, "fw_pay_licences", doc_id, h[5], "duration", f"{pay['licences']} day", **s)
        self.fact(company, "fw_pay_maintenance", doc_id, h[6], "duration", f"{pay['maintenance']} day", **s)
        self.fact(company, "fw_conf", doc_id, h[7], "duration", f"{conf} year", **s)
        self.fact(company, "fw_cap", doc_id, h[8], "amount", f"{cap:.2f}", **s)
        self.fact(company, "fw_notice", doc_id, h[9], "duration", f"{notice} month", **s)
        self.extra_q.append(dict(company=company, kind="conflict", type="amount", expected=[f"{senior:.2f}", f"{new_senior:.2f}"],
                                 sources=[[doc_id, h[3]], [amend_id, amend[0][0]]], sup=sup))

    # -- travel and purchasing policy: near-identical sections
    def policy(self, company):
        lang = COMPANIES[company]["lang"]
        meal_home, meal_abroad = float(self.rng.choice([20, 25, 30])), float(self.rng.choice([35, 40, 45, 50]))
        hotel_home, hotel_abroad = float(self.rng.choice([120, 130, 140])), float(self.rng.choice([160, 180, 200]))
        km = self.rng.choice([0.28, 0.30, 0.32, 0.35])
        quote1, quote3 = float(self.rng.choice([500, 1000, 1500])), float(self.rng.choice([5000, 7500, 10000]))
        A = lambda x: fmt_amount(lang, x)
        F = lambda n: self.filler(lang, n)
        if lang == "fr":
            secs = [("1. Champ d'application", f"La présente politique s'applique à tous les salariés de {COMPANIES[company]['name']} qui engagent des frais pour le compte de la société. {F(2)}"),
                    ("2.1 Repas – Luxembourg", f"Un repas d'affaires pris au Luxembourg est remboursé dans la limite de {A(meal_home)} par personne, boissons comprises."),
                    ("2.2 Repas – étranger", f"Un repas d'affaires pris à l'étranger est remboursé dans la limite de {A(meal_abroad)} par personne, boissons comprises."),
                    ("3.1 Hébergement – Luxembourg", f"Une nuit d'hôtel au Luxembourg est prise en charge dans la limite de {A(hotel_home)}, petit-déjeuner inclus."),
                    ("3.2 Hébergement – étranger", f"Une nuit d'hôtel à l'étranger est prise en charge dans la limite de {A(hotel_abroad)}, petit-déjeuner inclus."),
                    ("4. Véhicule personnel", f"L'utilisation d'un véhicule personnel pour un déplacement professionnel donne lieu à une indemnité de {A(km)} par kilomètre. {F(1)}"),
                    ("5.1 Achats – devis unique", f"Tout achat d'un montant supérieur à {A(quote1)} hors taxes fait l'objet d'un devis écrit. {F(1)}"),
                    ("5.2 Achats – mise en concurrence", f"Tout achat d'un montant supérieur à {A(quote3)} hors taxes fait l'objet de trois devis comparables. {F(1)}")]
            title = "Politique de déplacements et d'achats"
        elif lang == "en":
            secs = [("1. Scope", f"This policy applies to every employee of {COMPANIES[company]['name']} who incurs costs on behalf of the company. {F(2)}"),
                    ("2.1 Meals – Luxembourg", f"A business meal taken in Luxembourg is reimbursed up to {A(meal_home)} per person, drinks included."),
                    ("2.2 Meals – abroad", f"A business meal taken abroad is reimbursed up to {A(meal_abroad)} per person, drinks included."),
                    ("3.1 Accommodation – Luxembourg", f"A hotel night in Luxembourg is covered up to {A(hotel_home)}, breakfast included."),
                    ("3.2 Accommodation – abroad", f"A hotel night abroad is covered up to {A(hotel_abroad)}, breakfast included."),
                    ("4. Private vehicle", f"Using a private vehicle for a business trip gives rise to an allowance of {A(km)} per kilometre. {F(1)}"),
                    ("5.1 Purchasing – single quote", f"Any purchase above {A(quote1)} excluding VAT requires a written quote. {F(1)}"),
                    ("5.2 Purchasing – competitive quotes", f"Any purchase above {A(quote3)} excluding VAT requires three comparable quotes. {F(1)}")]
            title = "Travel and purchasing policy"
        else:
            secs = [("1. Geltungsbereich", f"Diese Richtlinie gilt für alle Beschäftigten der {COMPANIES[company]['name']}, die Ausgaben für das Unternehmen tätigen. {F(2)}"),
                    ("2.1 Bewirtung – Luxemburg", f"Ein Geschäftsessen in Luxemburg wird bis zu {A(meal_home)} pro Person erstattet, Getränke inbegriffen."),
                    ("2.2 Bewirtung – Ausland", f"Ein Geschäftsessen im Ausland wird bis zu {A(meal_abroad)} pro Person erstattet, Getränke inbegriffen."),
                    ("3.1 Übernachtung – Luxemburg", f"Eine Hotelübernachtung in Luxemburg wird bis zu {A(hotel_home)} übernommen, Frühstück inbegriffen."),
                    ("3.2 Übernachtung – Ausland", f"Eine Hotelübernachtung im Ausland wird bis zu {A(hotel_abroad)} übernommen, Frühstück inbegriffen."),
                    ("4. Privatfahrzeug", f"Für Dienstfahrten mit dem Privatfahrzeug wird eine Pauschale von {A(km)} je Kilometer gezahlt. {F(1)}"),
                    ("5.1 Einkauf – Einzelangebot", f"Jeder Einkauf über {A(quote1)} netto erfordert ein schriftliches Angebot. {F(1)}"),
                    ("5.2 Einkauf – Vergleichsangebote", f"Jeder Einkauf über {A(quote3)} netto erfordert drei vergleichbare Angebote. {F(1)}")]
            title = "Reise- und Einkaufsrichtlinie"
        doc_id = f"{company}-policy"
        self.add_doc(company, doc_id, title, "procedure", "pdf", lang, dt.date(2025, 1, 6),
                     [(h, t, "procedure_step") for h, t in secs])
        h = [x[0] for x in secs]
        for kind, i, v in [("pol_meal_home", 1, meal_home), ("pol_meal_abroad", 2, meal_abroad), ("pol_hotel_home", 3, hotel_home),
                           ("pol_hotel_abroad", 4, hotel_abroad), ("pol_km", 5, km), ("pol_quote1", 6, quote1), ("pol_quote3", 7, quote3)]:
            self.fact(company, kind, doc_id, h[i], "amount", f"{v:.2f}")

    # -- e-mails: a payment plan, a meeting moved to another day, a US-style date
    def emails(self, company):
        lang = COMPANIES[company]["lang"]
        sup = FRAMEWORK_PROVIDER[company]
        balance = round(self.rng.uniform(2400, 9000), 2)
        inst = round(balance / 3, 2)
        balance = round(inst * 3, 2)
        first = dt.date(2025, 11, self.rng.randint(3, 14))
        A = lambda x: fmt_amount(lang, x)
        body = {"fr": f"Bonjour,\n\nSuite à notre échange, nous acceptons que le solde de {A(balance)} soit réglé en trois mensualités de {A(inst)}, "
                      f"la première le {fmt_date(lang, first)}.\n\nBien à vous,\n{sup}",
                "en": f"Hello,\n\nFurther to our call, we are happy for the outstanding {A(balance)} to be paid in three monthly instalments of {A(inst)}, "
                      f"the first on {fmt_date(lang, first)}.\n\nBest wishes,\n{sup}",
                "de": f"Guten Tag,\n\nwie besprochen sind wir einverstanden, dass der offene Betrag von {A(balance)} in drei Monatsraten zu je {A(inst)} "
                      f"beglichen wird, die erste am {fmt_date(lang, first)}.\n\nFreundliche Grüße\n{sup}"}[lang]
        subj = {"fr": "Échelonnement du solde", "en": "Payment plan", "de": "Ratenzahlung"}[lang]
        self.email(company, f"{company}-mail-plan", sup, subj, lang, first - dt.timedelta(days=20), body)
        self.fact(company, "mail_instalment", f"{company}-mail-plan", "Message", "amount", f"{inst:.2f}", sup=sup)

        old = dt.date(2026, 3, self.rng.randint(2, 12))
        new = old + dt.timedelta(days=7)
        body = {"fr": f"Bonjour,\n\nLa revue des comptes initialement prévue le {fmt_date(lang, old)} est décalée au {fmt_date(lang, new)}, même heure, même salle. "
                      "Merci de votre compréhension.\n\nCordialement,\nVotre expert-comptable",
                "en": f"Hi,\n\nThe accounts review originally planned for {fmt_date(lang, old)} has been pushed back to {fmt_date(lang, new)}, same time, same room. "
                      "Sorry for the change.\n\nRegards,\nYour accountant",
                "de": f"Guten Tag,\n\ndie ursprünglich für den {fmt_date(lang, old)} geplante Kontenprüfung wird auf den {fmt_date(lang, new)} verschoben, gleiche Uhrzeit, gleicher Raum. "
                      "Danke für Ihr Verständnis.\n\nMit freundlichen Grüßen\nIhr Steuerberater"}[lang]
        subj = {"fr": "Revue des comptes : nouvelle date", "en": "Accounts review: new date", "de": "Kontenprüfung: neuer Termin"}[lang]
        self.email(company, f"{company}-mail-moved", "Cabinet comptable fictif", subj, lang, dt.date(2026, 2, 20), body)
        self.fact(company, "mail_moved", f"{company}-mail-moved", "Message", "date", new.isoformat())

        sender = PARTS_SENDER[company]
        a, b = self.rng.sample(range(1, 13), 2)
        raw = f"{a:02d}/{b:02d}/2025"
        body = (f"Hello,\n\nThe replacement parts you ordered left our warehouse today. Expected arrival: {raw}. "
                f"Tracking details will follow.\n\nRegards,\n{sender}")
        self.email(company, f"{company}-mail-parts", sender, "Replacement parts dispatched", "en", dt.date(2025, 2, 25), body)
        self.extra_q.append(dict(company=company, kind="ambiguous_date", type="date", expected=[raw],
                                 sources=[[f"{company}-mail-parts", "Message"]], sender=sender))

    def build(self):
        for c in COMPANIES:
            self.table_invoices(c)
            self.vat_statement(c)
            self.framework(c)
            self.policy(c)
            self.emails(c)


# --------------------------------------------------------------------------- questions
def render_fact(f: dict, lang: str, rng: random.Random) -> str:
    s, k = f["slots"], f["kind"]
    sup = s.get("sup", "")
    if k.startswith("inv_"):
        if rng.random() < 0.5:
            ref = {"fr": f"la facture {s['num']}", "en": f"invoice {s['num']}", "de": f"Rechnung {s['num']}"}[lang]
        else:
            m = month_name(lang, s["month"])
            ref = {"fr": f"la note de {s['sup']} de {m}", "en": f"the {s['sup']} bill from {m}", "de": f"die Rechnung von {s['sup']} aus dem {m}"}[lang]
        item = s["item"][lang]
        return {
            "inv_price": {"fr": f"Combien nous a-t-on compté pour « {item} » sur {ref}, à l'unité ?", "en": f"{item} on {ref}: what did they charge apiece?",
                          "de": f"Was hat {item} auf {ref} pro Stück gekostet?"},
            "inv_total": {"fr": f"On doit combien au final pour {ref} ?", "en": f"What's the bottom line on {ref}?",
                          "de": f"Was steht unterm Strich auf {ref}?"},
            "inv_due": {"fr": f"Jusqu'à quand peut-on régler {ref} ?", "en": f"What's the last day to pay {ref}?",
                        "de": f"Bis zu welchem Tag ist {ref} zu begleichen?"},
        }[k][lang]
    if k in ("vat_q1", "vat_q2", "vat_q3"):
        o = ORD[lang][s["quarter"] - 1]
        return {"fr": f"Montant de TVA retenu par l'administration pour le {o} trimestre 2025 ?",
                "en": f"How much VAT did the tax office settle on for the {o} quarter of 2025?",
                "de": f"Welche Umsatzsteuer wurde für das {o} Quartal 2025 festgestellt?"}[lang]
    T = {
        "vat_q3_due": {"fr": "Pour quand faut-il avoir payé la TVA du troisième trimestre 2025 ?",
                       "en": "Q3 2025 VAT: when does it have to be paid by?",
                       "de": "Bis wann müssen wir die Umsatzsteuer fürs dritte Quartal 2025 überweisen?"},
        "vat_objection": {"fr": "Quel délai pour réclamer contre le relevé annuel de TVA ?",
                          "en": "If we disagree with the annual VAT statement, how long do we have to object?",
                          "de": "Wie viel Zeit bleibt uns für einen Einspruch gegen den Umsatzsteuer-Jahresauszug?"},
        "fw_start": {"fr": f"Depuis quand le contrat-cadre avec {sup} est-il en vigueur ?", "en": f"When did the framework agreement with {sup} kick in?",
                     "de": f"Seit wann gilt der Rahmenvertrag mit {sup}?"},
        "fw_junior": {"fr": f"Un consultant junior de {sup}, ça nous coûte combien de l'heure ?", "en": f"What do we pay per hour for a junior consultant from {sup}?",
                      "de": f"Was kostet uns eine Stunde Junior-Berater bei {sup}?"},
        "fw_pay_consulting": {"fr": f"En combien de jours doit-on payer les factures de conseil de {sup} ?",
                              "en": f"How many days do we get to pay {sup}'s advisory invoices?",
                              "de": f"Wie viele Tage Zahlungsziel haben wir bei Beratungsrechnungen von {sup}?"},
        "fw_pay_licences": {"fr": f"Les factures de licences de {sup}, on a combien de jours pour les payer ?",
                            "en": f"How long can we take to pay {sup} for licences?",
                            "de": f"Wie viele Tage haben wir, um Lizenzrechnungen von {sup} zu zahlen?"},
        "fw_pay_maintenance": {"fr": f"Quel délai de paiement pour la maintenance facturée par {sup} ?",
                               "en": f"Payment terms for {sup}'s maintenance invoices?",
                               "de": f"Welches Zahlungsziel gilt für Wartungsrechnungen von {sup}?"},
        "fw_conf": {"fr": f"La confidentialité avec {sup} dure combien de temps après la fin du contrat ?",
                    "en": f"How long after the agreement ends are we still bound to confidentiality with {sup}?",
                    "de": f"Wie lange nach Vertragsende gilt die Verschwiegenheit mit {sup}?"},
        "fw_cap": {"fr": f"Jusqu'à quel montant {sup} est-il responsable en cas de dommage ?", "en": f"What's the cap on {sup}'s liability?",
                   "de": f"Bis zu welcher Summe haftet {sup}?"},
        "fw_notice": {"fr": f"Si on veut arrêter avec {sup}, il faut prévenir combien de temps avant ?",
                      "en": f"How much notice do we need to give to end things with {sup}?",
                      "de": f"Mit welchem Vorlauf können wir den Rahmenvertrag mit {sup} beenden?"},
        "pol_meal_home": {"fr": "Plafond d'un repas d'affaires au Luxembourg ?", "en": "How much can I spend on a business meal in Luxembourg?",
                          "de": "Wie viel darf ein Geschäftsessen in Luxemburg kosten?"},
        "pol_meal_abroad": {"fr": "Plafond d'un repas d'affaires à l'étranger ?", "en": "How much can I spend on a business meal abroad?",
                            "de": "Wie viel darf ein Geschäftsessen im Ausland kosten?"},
        "pol_hotel_home": {"fr": "Prix maximum remboursé pour une nuit d'hôtel au Luxembourg ?", "en": "Hotel cap per night in Luxembourg?",
                           "de": "Obergrenze für eine Hotelnacht in Luxemburg?"},
        "pol_hotel_abroad": {"fr": "Prix maximum remboursé pour une nuit d'hôtel à l'étranger ?", "en": "Hotel cap per night abroad?",
                             "de": "Obergrenze für eine Hotelnacht im Ausland?"},
        "pol_km": {"fr": "Combien est remboursé le kilomètre avec sa voiture perso ?", "en": "What's the mileage rate for using my own car?",
                   "de": "Wie viel Kilometergeld gibt es mit dem eigenen Auto?"},
        "pol_quote1": {"fr": "À partir de quel montant faut-il un devis pour un achat ?", "en": "From what amount do purchases need a written quote?",
                       "de": "Ab welchem Betrag braucht ein Einkauf ein schriftliches Angebot?"},
        "pol_quote3": {"fr": "Au-delà de quelle somme doit-on demander trois devis ?", "en": "Above what spend do we need three competing quotes?",
                       "de": "Ab welcher Summe sind drei Vergleichsangebote nötig?"},
        "mail_instalment": {"fr": f"{sup} accepte un échelonnement : combien par mensualité ?", "en": f"How big is each instalment in the payment plan with {sup}?",
                            "de": f"Wie hoch ist jede Rate im Ratenplan mit {sup}?"},
        "mail_moved": {"fr": "Finalement, la revue des comptes, c'est quel jour ?", "en": "So when is the accounts review now, after the change?",
                       "de": "Auf welchen Tag wurde die Kontenprüfung verschoben?"},
    }
    return T[k][lang]


def unanswerable(rng: random.Random, gen: HardGenerator) -> list[dict]:
    out = []
    langs = ["fr", "en", "de"]
    names = list(COMPANIES)
    for i, c in enumerate(names):
        sup = FRAMEWORK_PROVIDER[c]
        other = names[(i + 1) % 3]
        num = rng.choice(gen.invoice_numbers[other])
        isup = INVOICE_SUPPLIERS[c][rng.randrange(2)][0]
        cases = [
            ("absent_period", {"fr": "Combien de TVA pour le quatrième trimestre 2025 ?", "en": "VAT assessed for Q4 2025?",
                               "de": "Wie viel Umsatzsteuer für das vierte Quartal 2025?"}),
            ("near_miss", {"fr": f"En combien de jours doit-on payer le matériel informatique acheté à {sup} ?",
                           "en": f"Payment terms for hardware bought from {sup}?", "de": f"Welches Zahlungsziel gilt für Hardware von {sup}?"}),
            ("other_company", {"fr": f"Montant total de la facture {num} ?", "en": f"What's the total on invoice {num}?",
                               "de": f"Gesamtbetrag der Rechnung {num}?"}),
            ("absent_topic", {"fr": "Quel est le budget voiture de fonction pour un cadre ?", "en": "What's the company car allowance for managers?",
                              "de": "Wie hoch ist das Dienstwagenbudget für Führungskräfte?"}),
            ("absent_value", {"fr": f"Numéro de TVA intracommunautaire de {isup} ?", "en": f"What is {isup}'s VAT registration number?",
                              "de": f"Wie lautet die USt-IdNr. von {isup}?"}),
        ]
        for reason, q in cases:
            lang = rng.choice(langs)
            out.append(dict(company=c, lang=lang, reason=reason, question=q[lang]))
    c = rng.choice(names)
    lang = rng.choice(langs)
    sup = FRAMEWORK_PROVIDER[c]
    out.append(dict(company=c, lang=lang, reason="absent_value", question={
        "fr": f"Quel est le numéro de téléphone de {sup} ?", "en": f"What's {sup}'s phone number?", "de": f"Unter welcher Nummer erreichen wir {sup}?"}[lang]))
    return out


def build_questions(gen: HardGenerator, rng: random.Random) -> list[dict]:
    langs = ["fr", "en", "de"]
    qs = []
    for f in rng.sample(gen.facts, N_ANSWERABLE_FACTS):
        lang = rng.choice(langs)
        qs.append(dict(company=f["company"], lang=lang, question=style_question(rng, render_fact(f, lang, rng)), answerable=True,
                       kind=f["kind"], type=f["type"], expected=[f["expected"]], sources=[[f["doc_id"], f["heading"]]]))
    for x in gen.extra_q:
        lang = rng.choice(langs)
        if x["kind"] == "conflict":
            q = {"fr": f"Combien facture {x['sup']} pour une heure de consultant senior ?", "en": f"Hourly price of a senior consultant at {x['sup']}?",
                 "de": f"Welcher Stundensatz gilt für Senior-Berater von {x['sup']}?"}[lang]
        else:
            q = {"fr": f"Quand arrivent les pièces de rechange de {x['sender']} ?", "en": f"When will the replacement parts from {x['sender']} arrive?",
                 "de": f"Wann kommen die Ersatzteile von {x['sender']} an?"}[lang]
        qs.append(dict(company=x["company"], lang=lang, question=style_question(rng, q), answerable=True, kind=x["kind"],
                       type=x["type"], expected=x["expected"], sources=x["sources"]))
    for u in unanswerable(rng, gen):
        qs.append(dict(company=u["company"], lang=u["lang"], question=style_question(rng, u["question"]), answerable=False,
                       kind="unanswerable:" + u["reason"], type=None, expected=[], sources=[]))
    rng.shuffle(qs)
    for i, q in enumerate(qs):
        q["id"] = f"h{i + 1:03d}"
        q["split"] = "hard"
    return qs


def main():
    gen = HardGenerator()
    gen.build()
    questions = build_questions(gen, random.Random(SEED + 1))
    for d in gen.docs:
        p = CORPUS / d["company"] / f"{d['doc_id']}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        fm = "\n".join(f"{k}: {v}" for k, v in d["meta"].items())
        p.write_text(f"---\n{fm}\n---\n{d['body']}\n", encoding="utf-8")
    LABELS.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in gen.labels), encoding="utf-8")
    keys = ["id", "split", "company", "lang", "question", "answerable", "kind", "type", "expected", "sources"]
    QUESTIONS.write_text("".join(json.dumps({k: q[k] for k in keys}, ensure_ascii=False) + "\n" for q in questions),
                         encoding="utf-8")
    n_un = sum(not q["answerable"] for q in questions)
    print(f"{len(gen.docs)} documents, {len(gen.labels)} labelled sections, "
          f"{len(questions)} questions ({n_un} without answer, {100 * n_un / len(questions):.0f} %)")


if __name__ == "__main__":
    main()
