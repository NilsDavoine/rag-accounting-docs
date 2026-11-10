"""Deterministic generator for the invented accounting corpus and the question bench.

Everything here is fictional: companies, suppliers, amounts, tax office. The same seed
always produces byte-identical files, so the benchmark is reproducible.

Outputs
  data/corpus/<company>/<doc_id>.txt   documents (front-matter + body)
  data/nature_labels.jsonl             one label per section (used to train the classifier)
  eval/questions.jsonl                 ~150 questions, ~20 % without an answer

Run:  python -m data.generate_corpus
"""
from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

SEED = 20251110
ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
LABELS = ROOT / "data" / "nature_labels.jsonl"
QUESTIONS = ROOT / "eval" / "questions.jsonl"

COMPANIES = {
    "ombrelune": {"name": "Ombrelune Logistique S.à r.l.", "lang": "fr", "short": "Ombrelune"},
    "quillfern": {"name": "Quillfern Software S.A.", "lang": "en", "short": "Quillfern"},
    "halvardt": {"name": "Halvardt Bautechnik GmbH", "lang": "de", "short": "Halvardt"},
}

# Invented suppliers: (name, language, invoice prefix, catalogue key)
SUPPLIERS = {
    "ombrelune": [("Cabinet Vauclerc Conseil", "fr", "VCC", "consulting"),
                  ("Papeterie Grisolle", "fr", "PGR", "office"),
                  ("Trans-Ardoise SA", "fr", "TAR", "transport"),
                  ("Nettoyage Brillamonde", "fr", "NBM", "cleaning"),
                  ("Kelderwick Software Ltd", "en", "KSW", "software"),
                  ("Brightmoor Supply Inc.", "en", "BMS", "office"),
                  ("Fenwald Druckerei GmbH", "de", "FDR", "printing")],
    "quillfern": [("Ashcombe Advisory Ltd", "en", "AAL", "consulting"),
                  ("Pellow Office Goods", "en", "POG", "office"),
                  ("Corriveau Hébergement", "fr", "CHB", "hosting"),
                  ("Marlstone Cleaning Co.", "en", "MCC", "cleaning"),
                  ("Dunmere Cloud Services", "en", "DCS", "hosting"),
                  ("Orrin & Tate Hardware", "en", "OTH", "hardware"),
                  ("Selbrecht IT-Service", "de", "SIT", "software")],
    "halvardt": [("Ingenieurbüro Kranzfeld", "de", "IKF", "consulting"),
                 ("Baustoffe Ottenmoor KG", "de", "BOK", "materials"),
                 ("Schwarzach Transporte", "de", "SWT", "transport"),
                 ("Gerüstbau Lindmarsch", "de", "GLM", "scaffolding"),
                 ("Atelier Morvignac", "fr", "AMV", "consulting"),
                 ("Wexley Tools Ltd", "en", "WXT", "hardware"),
                 ("Bürobedarf Quendt", "de", "BBQ", "office")],
}

CATALOGUE = {
    "consulting": [({"fr": "Prestations de conseil", "en": "Consulting services", "de": "Beratungsleistungen"}, "h", 85, 140)],
    "office": [({"fr": "Ramettes de papier A4", "en": "A4 paper reams", "de": "Kopierpapier A4"}, "u", 4, 9),
               ({"fr": "Cartouches d'encre", "en": "Ink cartridges", "de": "Tintenpatronen"}, "u", 25, 60)],
    "transport": [({"fr": "Transport de palettes", "en": "Pallet transport", "de": "Palettentransport"}, "u", 45, 90)],
    "cleaning": [({"fr": "Nettoyage des bureaux", "en": "Office cleaning", "de": "Büroreinigung"}, "h", 28, 38)],
    "software": [({"fr": "Licence logicielle annuelle", "en": "Annual software licence", "de": "Jährliche Softwarelizenz"}, "u", 900, 2400)],
    "hosting": [({"fr": "Hébergement de serveurs", "en": "Server hosting", "de": "Server-Hosting"}, "m", 350, 900)],
    "hardware": [({"fr": "Ordinateurs portables", "en": "Laptops", "de": "Laptops"}, "u", 850, 1400)],
    "printing": [({"fr": "Impression de brochures", "en": "Brochure printing", "de": "Broschürendruck"}, "u", 2, 5)],
    "materials": [({"fr": "Ciment (sacs de 25 kg)", "en": "Cement (25 kg bags)", "de": "Zement (25-kg-Säcke)"}, "u", 7, 12)],
    "scaffolding": [({"fr": "Location d'échafaudage", "en": "Scaffolding rental", "de": "Gerüstmiete"}, "w", 300, 650)],
}
UNIT = {"h": {"fr": "h", "en": "h", "de": "Std."}, "u": {"fr": "unités", "en": "units", "de": "Stück"},
        "m": {"fr": "mois", "en": "months", "de": "Monate"}, "w": {"fr": "semaines", "en": "weeks", "de": "Wochen"}}

MONTHS = {
    "fr": ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
           "octobre", "novembre", "décembre"],
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December"],
    "de": ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September",
           "Oktober", "November", "Dezember"],
}
DUR = {"day": {"fr": "jours", "en": "days", "de": "Tage"},
       "month": {"fr": "mois", "en": "months", "de": "Monate"},
       "year": {"fr": "ans", "en": "years", "de": "Jahre"},
       "week": {"fr": "semaines", "en": "weeks", "de": "Wochen"}}


# --------------------------------------------------------------------------- formatting
def fmt_amount(lang: str, x: float) -> str:
    whole, cents = f"{x:,.2f}".split(".")
    if lang == "fr":
        return f"{whole.replace(',', ' ')},{cents} EUR"
    if lang == "de":
        return f"{whole.replace(',', '.')},{cents} EUR"
    return f"EUR {whole}.{cents}"


def fmt_date(lang: str, d: dt.date) -> str:
    if lang == "fr":
        return f"{d.day} {MONTHS['fr'][d.month - 1]} {d.year}"
    if lang == "de":
        return f"{d.day:02d}.{d.month:02d}.{d.year}"
    return f"{d.day} {MONTHS['en'][d.month - 1]} {d.year}"


def fmt_dur(lang: str, n: int, unit: str) -> str:
    return f"{n} {DUR[unit][lang]}"


def fmt_pct(lang: str, p: float) -> str:
    s = f"{p:g}"
    return (s.replace(".", ",") + " %") if lang in ("fr", "de") else s + "%"


def ocr_noise(rng: random.Random, text: str) -> str:
    """Simulate a scanned page: lost accents, l/1 and rn/m confusions, broken hyphenation.
    Digits are never altered, so every fact stays recoverable."""
    out = []
    for word in text.split(" "):
        if any(c.isdigit() for c in word) or rng.random() > 0.18:
            out.append(word)
            continue
        w = word
        roll = rng.random()
        if roll < 0.35:
            w = w.translate(str.maketrans("éèêàùçäöüß", "eeeaucaous"))
        elif roll < 0.6:
            w = w.replace("l", "1", 1)
        elif roll < 0.8:
            w = w.replace("m", "rn", 1)
        elif len(w) > 7:
            w = w[: len(w) // 2] + "-\n" + w[len(w) // 2:]
        out.append(w)
    return " ".join(out)


# --------------------------------------------------------------------------- generator
class Generator:
    def __init__(self, seed: int = SEED):
        self.rng = random.Random(seed)
        self.docs: list[dict] = []      # {doc_id, company, meta, body}
        self.labels: list[dict] = []    # {doc_id, heading, nature}
        self.facts: list[dict] = []     # answerable facts (one question each, later sampled)
        self.extra_q: list[dict] = []   # conflict / ambiguity / unanswerable questions
        self.invoice_numbers: dict[str, list[str]] = {c: [] for c in COMPANIES}

    # -- helpers
    def add_doc(self, company, doc_id, title, dtype, source, lang, date, sections, header=None):
        """sections: list of (heading, text, nature). Email documents pass header lines."""
        meta = {"doc_id": doc_id, "company": company, "title": title, "type": dtype,
                "source": source, "lang": lang, "date": date.isoformat()}
        parts = []
        if header:
            parts.append("\n".join(header) + "\n")
        for heading, text, nature in sections:
            if source == "scan":
                text = ocr_noise(self.rng, text)
            if heading == "Message":
                parts.append(text)
            else:
                parts.append(f"## {heading}\n{text}")
            self.labels.append({"doc_id": doc_id, "heading": heading, "nature": nature})
        body = "\n\n".join(parts).replace(".. ", ". ")
        self.docs.append({"doc_id": doc_id, "company": company, "meta": meta, "body": body})

    def fact(self, company, kind, doc_id, heading, vtype, expected, **slots):
        self.facts.append({"company": company, "kind": kind, "doc_id": doc_id, "heading": heading,
                           "type": vtype, "expected": expected, "slots": slots})

    # -- invoices
    def invoices(self, company):
        cname = COMPANIES[company]["name"]
        for i, (sup, lang, prefix, cat) in enumerate(SUPPLIERS[company]):
            item, unit, lo, hi = self.rng.choice(CATALOGUE[cat])
            qty = self.rng.randint(2, 40) if unit in ("h", "u") else self.rng.randint(1, 12)
            price = round(self.rng.uniform(lo, hi) * 2) / 2
            net = round(qty * price, 2)
            vat = round(net * 0.17, 2)
            total = round(net + vat, 2)
            date = dt.date(2025, self.rng.randint(1, 10), self.rng.randint(1, 26))
            due = date + dt.timedelta(days=30)
            num = f"{prefix}-2025-{self.rng.randint(100, 899)}"
            self.invoice_numbers[company].append(num)
            doc_id = f"{company}-inv-{i + 1:02d}"
            source = "scan" if i % 3 == 2 else "pdf"
            L = {
                "fr": ("En-tête", "Détail", "Totaux et paiement", "Facture",
                       f"Fournisseur : {sup}. Client : {cname}. Facture n° {num}, émise le {fmt_date('fr', date)}.",
                       f"{item['fr']} : {qty} {UNIT[unit]['fr']} à {fmt_amount('fr', price)} l'unité, soit {fmt_amount('fr', net)} hors taxes.",
                       f"Montant hors taxes : {fmt_amount('fr', net)}. TVA à 17 % : {fmt_amount('fr', vat)}. Total TTC : {fmt_amount('fr', total)}. "
                       f"Paiement par virement au plus tard le {fmt_date('fr', due)}."),
                "en": ("Header", "Line items", "Totals and payment", "Invoice",
                       f"Supplier: {sup}. Customer: {cname}. Invoice no. {num}, issued on {fmt_date('en', date)}.",
                       f"{item['en']}: {qty} {UNIT[unit]['en']} at {fmt_amount('en', price)} each, i.e. {fmt_amount('en', net)} excluding VAT.",
                       f"Net amount: {fmt_amount('en', net)}. VAT at 17%: {fmt_amount('en', vat)}. Total due: {fmt_amount('en', total)}. "
                       f"Payment by bank transfer no later than {fmt_date('en', due)}."),
                "de": ("Kopfdaten", "Positionen", "Summen und Zahlung", "Rechnung",
                       f"Lieferant: {sup}. Kunde: {cname}. Rechnung Nr. {num} vom {fmt_date('de', date)}.",
                       f"{item['de']}: {qty} {UNIT[unit]['de']} zu je {fmt_amount('de', price)}, insgesamt {fmt_amount('de', net)} netto.",
                       f"Nettobetrag: {fmt_amount('de', net)}. MwSt. 17 %: {fmt_amount('de', vat)}. Gesamtbetrag brutto: {fmt_amount('de', total)}. "
                       f"Zahlung per Überweisung spätestens am {fmt_date('de', due)}."),
            }[lang]
            h1, h2, h3, word, t1, t2, t3 = L
            self.add_doc(company, doc_id, f"{word} {num} – {sup}", "invoice", source, lang, date,
                         [(h1, t1, "party_details"), (h2, t2, "invoice_line"), (h3, t3, "invoice_line")])
            slots = dict(num=num, sup=sup, item=item, month=(date.year, date.month))
            self.fact(company, "inv_total", doc_id, h3, "amount", f"{total:.2f}", **slots)
            self.fact(company, "inv_vat", doc_id, h3, "amount", f"{vat:.2f}", **slots)
            self.fact(company, "inv_due", doc_id, h3, "date", due.isoformat(), **slots)
            self.fact(company, "inv_price", doc_id, h2, "amount", f"{price:.2f}", **slots)
            self.fact(company, "inv_date", doc_id, h1, "date", date.isoformat(), **slots)
            if i == 0:   # the consulting supplier also writes a payment reminder later
                self.reminder(company, sup, lang, num, total, due)

    def reminder(self, company, sup, lang, num, total, due):
        d = due + dt.timedelta(days=12)
        doc_id = f"{company}-mail-reminder"
        body = {
            "fr": f"Bonjour,\n\nSauf erreur de notre part, la facture {num} d'un montant de {fmt_amount('fr', total)} "
                  f"arrivée à échéance le {fmt_date('fr', due)} reste impayée. Merci de procéder au règlement.\n\nCordialement,\n{sup}",
            "en": f"Hello,\n\nUnless we are mistaken, invoice {num} for {fmt_amount('en', total)}, due on {fmt_date('en', due)}, "
                  f"is still unpaid. Could you please arrange payment?\n\nKind regards,\n{sup}",
            "de": f"Guten Tag,\n\nnach unseren Unterlagen ist die Rechnung {num} über {fmt_amount('de', total)}, fällig am "
                  f"{fmt_date('de', due)}, noch offen. Wir bitten um Begleichung.\n\nMit freundlichen Grüßen\n{sup}",
        }[lang]
        subj = {"fr": f"Relance facture {num}", "en": f"Reminder: invoice {num}", "de": f"Zahlungserinnerung {num}"}[lang]
        self.email(company, doc_id, sup, subj, lang, d, body)

    def email(self, company, doc_id, sender, subject, lang, date, body):
        header = [f"From: {sender}", f"To: {COMPANIES[company]['name']}", f"Date: {fmt_date(lang, date)}",
                  f"Subject: {subject}"]
        self.add_doc(company, doc_id, subject, "email", "email", lang, date,
                     [("Message", body, "correspondence")], header=header)

    # -- service contract (+ the conflicting e-mail)
    def service_contract(self, company, conflict):
        sup, lang, _, _ = SUPPLIERS[company][0]
        rate = {"ombrelune": 25.5, "quillfern": 31.0, "halvardt": 28.0}[company]
        start = dt.date(2025, self.rng.randint(1, 4), 1)
        months = self.rng.choice([12, 24, 36])
        pay = self.rng.choice([30, 45, 60])
        notice = 3
        doc_id = f"{company}-contract-services"
        T = {
            "fr": [("Article 1 – Objet", f"Le prestataire {sup} assiste {COMPANIES[company]['name']} dans la tenue de sa comptabilité et la préparation des clôtures."),
                   ("Article 2 – Durée", f"Le contrat prend effet le {fmt_date('fr', start)} pour une durée de {fmt_dur('fr', months, 'month')}, renouvelable par écrit."),
                   ("Article 3 – Honoraires", f"Les prestations sont facturées au taux horaire de {fmt_amount('fr', rate)} hors taxes, sur la base des heures réellement effectuées."),
                   ("Article 4 – Modalités de paiement", f"Les factures sont payables à {fmt_dur('fr', pay, 'day')} date de facture, par virement."),
                   ("Article 5 – Résiliation", f"Chaque partie peut résilier le contrat moyennant un préavis de {fmt_dur('fr', notice, 'month')}, notifié par lettre recommandée.")],
            "en": [("Article 1 – Purpose", f"The provider {sup} supports {COMPANIES[company]['name']} with bookkeeping and the preparation of closings."),
                   ("Article 2 – Term", f"The agreement takes effect on {fmt_date('en', start)} for a term of {fmt_dur('en', months, 'month')}, renewable in writing."),
                   ("Article 3 – Fees", f"Services are invoiced at an hourly rate of {fmt_amount('en', rate)} excluding VAT, based on hours actually worked."),
                   ("Article 4 – Payment terms", f"Invoices are payable within {fmt_dur('en', pay, 'day')} of the invoice date, by bank transfer."),
                   ("Article 5 – Termination", f"Either party may terminate the agreement with {fmt_dur('en', notice, 'month')}' notice, given by registered letter.")],
            "de": [("Artikel 1 – Gegenstand", f"Der Auftragnehmer {sup} unterstützt die {COMPANIES[company]['name']} bei der Buchhaltung und der Vorbereitung der Abschlüsse."),
                   ("Artikel 2 – Laufzeit", f"Der Vertrag beginnt am {fmt_date('de', start)} und läuft {fmt_dur('de', months, 'month')}; eine Verlängerung bedarf der Schriftform."),
                   ("Artikel 3 – Vergütung", f"Die Leistungen werden zu einem Stundensatz von {fmt_amount('de', rate)} netto nach tatsächlich geleisteten Stunden abgerechnet."),
                   ("Artikel 4 – Zahlungsbedingungen", f"Rechnungen sind innerhalb von {fmt_dur('de', pay, 'day')} ab Rechnungsdatum per Überweisung zahlbar."),
                   ("Artikel 5 – Kündigung", f"Jede Partei kann den Vertrag mit einer Frist von {fmt_dur('de', notice, 'month')} per Einschreiben kündigen.")],
        }[lang]
        sections = [(h, t, "contract_clause") for h, t in T]
        if company == "halvardt":
            sections.append(("Artikel 6 – Vertragsstrafe", "Bei verspäteter Lieferung der Abschlussunterlagen wird eine Vertragsstrafe von 0,5 % des Monatshonorars pro angefangene Woche fällig.", "contract_clause"))
        title = {"fr": "Contrat de prestation de services", "en": "Service agreement", "de": "Dienstleistungsvertrag"}[lang]
        self.add_doc(company, doc_id, f"{title} – {sup}", "contract", "pdf", lang, start - dt.timedelta(days=20), sections)
        hs = [s[0] for s in sections]
        s = dict(sup=sup)
        self.fact(company, "svc_start", doc_id, hs[1], "date", start.isoformat(), **s)
        self.fact(company, "svc_duration", doc_id, hs[1], "duration", f"{months} month", **s)
        self.fact(company, "svc_pay", doc_id, hs[3], "duration", f"{pay} day", **s)
        if company != "ombrelune":
            self.fact(company, "svc_rate", doc_id, hs[2], "amount", f"{rate:.2f}", **s)
        if company != "quillfern":
            self.fact(company, "svc_notice", doc_id, hs[4], "duration", f"{notice} month", **s)

        # The real-world failure this repo reproduces: two sources disagree on a number.
        mail_id = f"{company}-mail-conflict"
        d = start + dt.timedelta(days=40)
        if company == "ombrelune":
            body = (f"Bonjour,\n\nComme convenu lors de notre réunion, nos prestations comptables vous seront facturées "
                    f"26 EUR de l'heure à compter de ce mois-ci. Le reste du contrat est inchangé.\n\nBien à vous,\n{sup}")
            self.email(company, mail_id, sup, "Taux horaire convenu", "fr", d, body)
            self.extra_q.append(dict(company=company, kind="conflict", type="amount",
                                     expected=["25.50", "26.00"], sources=[(doc_id, hs[2]), (mail_id, "Message")], sup=sup, field="rate"))
        elif company == "quillfern":
            body = (f"Hi,\n\nJust a quick note on our agreement: as discussed, the notice period for terminating the service "
                    f"agreement is 2 months, not more. Everything else stays as signed.\n\nBest,\n{sup}")
            self.email(company, mail_id, sup, "Notice period", "en", d, body)
            self.extra_q.append(dict(company=company, kind="conflict", type="duration",
                                     expected=["3 month", "2 month"], sources=[(doc_id, hs[4]), (mail_id, "Message")], sup=sup, field="notice"))
        else:
            body = (f"Guten Tag,\n\nzur Klarstellung: die vereinbarte Vertragsstrafe bei verspäteten Abschlussunterlagen beträgt "
                    f"0,6 % des Monatshonorars pro Woche.\n\nMit freundlichen Grüßen\n{sup}")
            self.email(company, mail_id, sup, "Vertragsstrafe", "de", d, body)
            self.extra_q.append(dict(company=company, kind="conflict", type="percent",
                                     expected=["0.5%", "0.6%"], sources=[(doc_id, hs[5]), (mail_id, "Message")], sup=sup, field="penalty"))

    # -- office lease: the "same subject in two companies" case (same invented building)
    def lease(self, company):
        lang = COMPANIES[company]["lang"]
        rent = {"ombrelune": 2450.0, "quillfern": 3180.0, "halvardt": 2875.0}[company]
        deposit = {"ombrelune": 3, "quillfern": 2, "halvardt": 3}[company]
        floor = {"ombrelune": 2, "quillfern": 4, "halvardt": 1}[company]
        start = dt.date(2024, self.rng.randint(1, 12), 1)
        doc_id = f"{company}-contract-lease"
        T = {
            "fr": [("Article 1 – Locaux", f"Le bailleur, la SCI Belvédère Fictive, loue au preneur des bureaux situés au {floor}e étage de la Résidence Belvédère."),
                   ("Article 2 – Loyer", f"Le loyer mensuel est fixé à {fmt_amount('fr', rent)} hors charges, payable d'avance le premier jour de chaque mois."),
                   ("Article 3 – Garantie", f"Le preneur verse une garantie locative égale à {fmt_dur('fr', deposit, 'month')} de loyer."),
                   ("Article 4 – Indexation", "Le loyer est indexé une fois par an sur l'indice des prix à la consommation.")],
            "en": [("Article 1 – Premises", f"The landlord, Belvedere Fictive Properties, lets to the tenant offices on floor {floor} of the Belvedere Residence."),
                   ("Article 2 – Rent", f"The monthly rent is {fmt_amount('en', rent)} excluding service charges, payable in advance on the first day of each month."),
                   ("Article 3 – Deposit", f"The tenant pays a security deposit equal to {fmt_dur('en', deposit, 'month')} of rent."),
                   ("Article 4 – Indexation", "The rent is indexed once a year on the consumer price index.")],
            "de": [("Artikel 1 – Mietgegenstand", f"Der Vermieter, Belvedere Fiktiv Immobilien, vermietet dem Mieter Büroräume im {floor}. Stock der Residenz Belvedere."),
                   ("Artikel 2 – Miete", f"Die monatliche Miete beträgt {fmt_amount('de', rent)} zuzüglich Nebenkosten und ist am ersten Tag jedes Monats im Voraus fällig."),
                   ("Artikel 3 – Kaution", f"Der Mieter leistet eine Kaution in Höhe von {fmt_dur('de', deposit, 'month')} Miete."),
                   ("Artikel 4 – Indexierung", "Die Miete wird einmal jährlich an den Verbraucherpreisindex angepasst.")],
        }[lang]
        title = {"fr": "Bail commercial – Résidence Belvédère", "en": "Office lease – Belvedere Residence",
                 "de": "Gewerbemietvertrag – Residenz Belvedere"}[lang]
        self.add_doc(company, doc_id, title, "contract", "pdf", lang, start, [(h, t, "contract_clause") for h, t in T])
        self.fact(company, "lease_rent", doc_id, T[1][0], "amount", f"{rent:.2f}")
        self.fact(company, "lease_deposit", doc_id, T[2][0], "duration", f"{deposit} month")

    # -- tax letters from an invented administration
    def tax(self, company):
        lang = COMPANIES[company]["lang"]
        vat_due = round(self.rng.uniform(4000, 20000), 2)
        adv = round(self.rng.uniform(1500, 9000), 2)
        pen = round(self.rng.uniform(50, 400), 2)
        tid = f"LU-FICT-{self.rng.randint(10000, 99999)}"
        d1, d2, d3 = dt.date(2025, 7, 18), dt.date(2025, 5, 6), dt.date(2025, 9, 9)
        dl1, dl2 = dt.date(2025, 8, 15), dt.date(2025, 6, 10)
        office = {"fr": "Administration fiscale fictive", "en": "Fictional Tax Administration",
                  "de": "Fiktive Steuerverwaltung"}[lang]
        A = lambda x: fmt_amount(lang, x)
        D = lambda x: fmt_date(lang, x)
        if lang == "fr":
            vat = [("Références", f"{office}. Contribuable : {COMPANIES[company]['name']}, matricule {tid}. Objet : décompte de TVA du deuxième trimestre 2025.", "party_details"),
                   ("Décompte", f"Après contrôle de votre déclaration, la TVA due pour le deuxième trimestre 2025 s'élève à {A(vat_due)}.", "tax_notice"),
                   ("Paiement", f"Ce montant doit être versé au plus tard le {D(dl1)}. Tout retard entraîne des intérêts de retard.", "tax_notice"),
                   ("Recours", f"Vous disposez d'un délai de {fmt_dur(lang, 3, 'month')} à compter de la notification pour introduire une réclamation.", "tax_notice")]
            advn = [("Références", f"{office}. Contribuable : {COMPANIES[company]['name']}, matricule {tid}.", "party_details"),
                    ("Avance trimestrielle", f"L'avance trimestrielle d'impôt sur le revenu des collectivités pour 2025 est fixée à {A(adv)} par trimestre.", "tax_notice"),
                    ("Échéance", f"La prochaine avance est payable au plus tard le {D(dl2)}.", "tax_notice")]
            rem = [("Références", f"{office}. Matricule {tid}. Rappel de dépôt de déclaration.", "party_details"),
                   ("Rappel", f"La déclaration d'impôt 2024 n'a pas été déposée dans les délais. Une astreinte de {A(pen)} est appliquée.", "tax_notice")]
            titles = ("Décompte de TVA T2 2025", "Avis d'avances trimestrielles 2025", "Rappel de déclaration 2024")
        elif lang == "en":
            vat = [("References", f"{office}. Taxpayer: {COMPANIES[company]['name']}, tax number {tid}. Subject: VAT assessment for the second quarter of 2025.", "party_details"),
                   ("Assessment", f"After review of your return, the VAT due for the second quarter of 2025 amounts to {A(vat_due)}.", "tax_notice"),
                   ("Payment", f"This amount must be paid no later than {D(dl1)}. Late payment triggers default interest.", "tax_notice"),
                   ("Appeal", f"You have {fmt_dur(lang, 3, 'month')} from notification to file an objection.", "tax_notice")]
            advn = [("References", f"{office}. Taxpayer: {COMPANIES[company]['name']}, tax number {tid}.", "party_details"),
                    ("Quarterly advance", f"The quarterly corporate income tax advance for 2025 is set at {A(adv)} per quarter.", "tax_notice"),
                    ("Due date", f"The next advance is payable no later than {D(dl2)}.", "tax_notice")]
            rem = [("References", f"{office}. Tax number {tid}. Filing reminder.", "party_details"),
                   ("Reminder", f"The 2024 tax return was not filed on time. A late-filing penalty of {A(pen)} is applied.", "tax_notice")]
            titles = ("VAT assessment Q2 2025", "Notice of quarterly advances 2025", "Filing reminder 2024")
        else:
            vat = [("Bezug", f"{office}. Steuerpflichtiger: {COMPANIES[company]['name']}, Steuernummer {tid}. Betreff: Umsatzsteuerbescheid für das zweite Quartal 2025.", "party_details"),
                   ("Festsetzung", f"Nach Prüfung Ihrer Erklärung beträgt die Umsatzsteuer für das zweite Quartal 2025 {A(vat_due)}.", "tax_notice"),
                   ("Zahlung", f"Der Betrag ist spätestens am {D(dl1)} zu entrichten. Bei Verspätung fallen Verzugszinsen an.", "tax_notice"),
                   ("Rechtsbehelf", f"Gegen diesen Bescheid kann innerhalb von {fmt_dur(lang, 3, 'month')} ab Bekanntgabe Einspruch eingelegt werden.", "tax_notice")]
            advn = [("Bezug", f"{office}. Steuerpflichtiger: {COMPANIES[company]['name']}, Steuernummer {tid}.", "party_details"),
                    ("Vierteljährliche Vorauszahlung", f"Die vierteljährliche Körperschaftsteuer-Vorauszahlung für 2025 beträgt {A(adv)} pro Quartal.", "tax_notice"),
                    ("Fälligkeit", f"Die nächste Vorauszahlung ist spätestens am {D(dl2)} fällig.", "tax_notice")]
            rem = [("Bezug", f"{office}. Steuernummer {tid}. Erinnerung zur Abgabe der Steuererklärung.", "party_details"),
                   ("Erinnerung", f"Die Steuererklärung 2024 wurde nicht fristgerecht abgegeben. Es wird ein Verspätungszuschlag von {A(pen)} festgesetzt.", "tax_notice")]
            titles = ("Umsatzsteuerbescheid Q2 2025", "Bescheid über Vorauszahlungen 2025", "Erinnerung Steuererklärung 2024")
        self.add_doc(company, f"{company}-tax-vat", titles[0], "tax_letter", "scan", lang, d1, vat)
        self.add_doc(company, f"{company}-tax-advance", titles[1], "tax_letter", "pdf", lang, d2, advn)
        self.add_doc(company, f"{company}-tax-reminder", titles[2], "tax_letter", "scan", lang, d3, rem)
        self.fact(company, "tax_vat_due", f"{company}-tax-vat", vat[1][0], "amount", f"{vat_due:.2f}")
        self.fact(company, "tax_vat_deadline", f"{company}-tax-vat", vat[2][0], "date", dl1.isoformat())
        self.fact(company, "tax_objection", f"{company}-tax-vat", vat[3][0], "duration", "3 month")
        self.fact(company, "tax_advance", f"{company}-tax-advance", advn[1][0], "amount", f"{adv:.2f}")
        self.fact(company, "tax_advance_due", f"{company}-tax-advance", advn[2][0], "date", dl2.isoformat())
        self.fact(company, "tax_penalty", f"{company}-tax-reminder", rem[1][0], "amount", f"{pen:.2f}")

    # -- internal procedures
    def procedures(self, company):
        lang = COMPANIES[company]["lang"]
        thr = {"ombrelune": 500, "quillfern": 750, "halvardt": 600}[company]
        sub = {"ombrelune": 30, "quillfern": 15, "halvardt": 20}[company]
        reimb = {"ombrelune": 10, "quillfern": 14, "halvardt": 7}[company]
        bank = {"ombrelune": 3, "quillfern": 2, "halvardt": 4}[company]
        A = lambda x: fmt_amount(lang, x)
        if lang == "fr":
            exp = [("Étape 1 – Saisie", f"Le salarié saisit sa note de frais dans un délai de {fmt_dur(lang, sub, 'day')} après la dépense, justificatifs joints."),
                   ("Étape 2 – Validation", f"Le responsable valide la note. Au-delà de {A(thr)}, l'accord du directeur financier est requis."),
                   ("Étape 3 – Remboursement", f"La comptabilité rembourse la note validée sous {fmt_dur(lang, reimb, 'day')}."),
                   ("Étape 4 – Archivage", f"Les justificatifs sont conservés pendant {fmt_dur(lang, 10, 'year')}.")]
            clo = [("Étape 1 – Rapprochements bancaires", f"Les rapprochements bancaires sont terminés dans les {fmt_dur(lang, bank, 'day')} ouvrés suivant la fin du mois."),
                   ("Étape 2 – Charges à payer", "Les charges à payer et produits à recevoir sont comptabilisés avant le reporting."),
                   ("Étape 3 – Revue", "Le responsable comptable revoit la balance et signe la clôture.")]
            titles = ("Procédure des notes de frais", "Procédure de clôture mensuelle")
        elif lang == "en":
            exp = [("Step 1 – Submission", f"Employees submit their expense claim within {fmt_dur(lang, sub, 'day')} of the expense, with receipts attached."),
                   ("Step 2 – Approval", f"The line manager approves the claim. Above {A(thr)}, the CFO's approval is required."),
                   ("Step 3 – Reimbursement", f"Accounting reimburses approved claims within {fmt_dur(lang, reimb, 'day')}."),
                   ("Step 4 – Archiving", f"Receipts are kept for {fmt_dur(lang, 10, 'year')}.")]
            clo = [("Step 1 – Bank reconciliations", f"Bank reconciliations are completed within {fmt_dur(lang, bank, 'day')} working days after month end."),
                   ("Step 2 – Accruals", "Accruals and deferred income are booked before reporting."),
                   ("Step 3 – Review", "The head of accounting reviews the trial balance and signs off the close.")]
            titles = ("Expense claim procedure", "Month-end closing procedure")
        else:
            exp = [("Schritt 1 – Erfassung", f"Mitarbeitende erfassen ihre Reisekostenabrechnung innerhalb von {fmt_dur(lang, sub, 'day')} nach der Ausgabe, mit Belegen."),
                   ("Schritt 2 – Freigabe", f"Die Führungskraft gibt die Abrechnung frei. Über {A(thr)} ist die Zustimmung des Finanzleiters erforderlich."),
                   ("Schritt 3 – Erstattung", f"Die Buchhaltung erstattet freigegebene Abrechnungen innerhalb von {fmt_dur(lang, reimb, 'day')}."),
                   ("Schritt 4 – Archivierung", f"Belege werden {fmt_dur(lang, 10, 'year')} aufbewahrt.")]
            clo = [("Schritt 1 – Bankabstimmung", f"Die Bankabstimmungen werden innerhalb von {fmt_dur(lang, bank, 'day')} Arbeitstagen nach Monatsende abgeschlossen."),
                   ("Schritt 2 – Abgrenzungen", "Rechnungsabgrenzungen werden vor dem Reporting gebucht."),
                   ("Schritt 3 – Prüfung", "Die Leitung Buchhaltung prüft die Saldenliste und zeichnet den Abschluss ab.")]
            titles = ("Verfahren Reisekostenabrechnung", "Verfahren Monatsabschluss")
        d = dt.date(2024, 11, 4)
        self.add_doc(company, f"{company}-proc-expenses", titles[0], "procedure", "pdf", lang, d, [(h, t, "procedure_step") for h, t in exp])
        self.add_doc(company, f"{company}-proc-closing", titles[1], "procedure", "pdf", lang, d, [(h, t, "procedure_step") for h, t in clo])
        e, c = f"{company}-proc-expenses", f"{company}-proc-closing"
        self.fact(company, "proc_threshold", e, exp[1][0], "amount", f"{thr:.2f}")
        self.fact(company, "proc_submit", e, exp[0][0], "duration", f"{sub} day")
        self.fact(company, "proc_reimb", e, exp[2][0], "duration", f"{reimb} day")
        self.fact(company, "proc_retention", e, exp[3][0], "duration", "10 year")
        self.fact(company, "proc_bank", c, clo[0][0], "duration", f"{bank} day")

    # -- other e-mails, including the ambiguous date
    def emails(self, company):
        lang = COMPANIES[company]["lang"]
        # A US-style date: 04/03/2025 means April 3 for the sender, March 4 for a European reader.
        doc_id = f"{company}-mail-delivery"
        body = ("Hi,\n\nYour order of ink cartridges has shipped. Delivery is scheduled for 04/03/2025. "
                "Please make sure someone can receive the parcel.\n\nThanks,\nBrightmoor Supply Inc.")
        self.email(company, doc_id, "Brightmoor Supply Inc.", "Shipping confirmation", "en", dt.date(2025, 3, 28), body)
        self.extra_q.append(dict(company=company, kind="ambiguous_date", type="date", expected=["04/03/2025"],
                                 sources=[(doc_id, "Message")]))
        # Meeting e-mail from the accountant
        mdate = dt.date(2025, 10, self.rng.randint(6, 24))
        doc_id = f"{company}-mail-meeting"
        body = {
            "fr": f"Bonjour,\n\nJe vous propose de faire le point sur la clôture annuelle le {fmt_date('fr', mdate)} à 10 h dans nos bureaux. "
                  "Merci d'apporter le grand livre provisoire.\n\nBien cordialement,\nVotre expert-comptable",
            "en": f"Dear all,\n\nI suggest we review the year-end closing on {fmt_date('en', mdate)} at 10 am in our offices. "
                  "Please bring the draft general ledger.\n\nBest regards,\nYour accountant",
            "de": f"Guten Tag,\n\nich schlage vor, den Jahresabschluss am {fmt_date('de', mdate)} um 10 Uhr in unserem Büro zu besprechen. "
                  "Bitte bringen Sie das vorläufige Hauptbuch mit.\n\nMit freundlichen Grüßen\nIhr Steuerberater",
        }[lang]
        subj = {"fr": "Réunion de clôture annuelle", "en": "Year-end closing meeting", "de": "Termin Jahresabschluss"}[lang]
        self.email(company, doc_id, "Cabinet comptable fictif", subj, lang, mdate - dt.timedelta(days=9), body)
        self.fact(company, "mail_meeting", doc_id, "Message", "date", mdate.isoformat())
        # A distracting e-mail: same topic as the expense procedure, no number in it.
        doc_id = f"{company}-mail-expenses"
        body = {
            "fr": "Bonjour à tous,\n\nPetit rappel : pensez à joindre vos justificatifs aux notes de frais, sinon la comptabilité ne peut pas les traiter.\n\nMerci,\nLa direction",
            "en": "Hi all,\n\nFriendly reminder: please attach receipts to your expense claims, otherwise accounting cannot process them.\n\nThanks,\nManagement",
            "de": "Hallo zusammen,\n\nkurze Erinnerung: bitte hängt eure Belege an die Reisekostenabrechnungen an, sonst kann die Buchhaltung sie nicht bearbeiten.\n\nDanke,\nGeschäftsleitung",
        }[lang]
        subj = {"fr": "Notes de frais – justificatifs", "en": "Expense claims – receipts", "de": "Reisekosten – Belege"}[lang]
        self.email(company, doc_id, "Direction", subj, lang, dt.date(2025, 2, 3), body)
        # Audit fee quote
        fee = round(self.rng.uniform(6000, 15000) / 50) * 50.0
        doc_id = f"{company}-mail-audit"
        body = {
            "fr": f"Madame, Monsieur,\n\nPour le contrôle légal des comptes annuels 2025, nos honoraires s'élèveraient à {fmt_amount('fr', fee)} hors TVA, "
                  "débours compris.\n\nNous restons à votre disposition,\nRéviseurs Associés Fictifs",
            "en": f"Dear Sir or Madam,\n\nFor the statutory audit of the 2025 annual accounts, our fee would be {fmt_amount('en', fee)} excluding VAT, "
                  "expenses included.\n\nKind regards,\nFictive Audit Partners",
            "de": f"Sehr geehrte Damen und Herren,\n\nfür die Abschlussprüfung des Jahresabschlusses 2025 beträgt unser Honorar {fmt_amount('de', fee)} netto, "
                  "Auslagen inbegriffen.\n\nMit freundlichen Grüßen\nFiktive Prüfungsgesellschaft",
        }[lang]
        subj = {"fr": "Offre d'honoraires – audit 2025", "en": "Fee proposal – 2025 audit", "de": "Honorarangebot Abschlussprüfung 2025"}[lang]
        self.email(company, doc_id, "Audit", subj, lang, dt.date(2025, 9, 22), body)
        self.fact(company, "mail_audit", doc_id, "Message", "amount", f"{fee:.2f}")

    def build(self):
        for c in COMPANIES:
            self.invoices(c)
            self.service_contract(c, conflict=True)
            self.lease(c)
            self.tax(c)
            self.procedures(c)
            self.emails(c)


# --------------------------------------------------------------------------- questions
def month_label(lang, ym):
    return f"{MONTHS[lang][ym[1] - 1]} {ym[0]}"


def render_fact(rng: random.Random, f: dict, lang: str) -> str:
    s = f["slots"]
    k = f["kind"]
    if k.startswith("inv_"):
        if rng.random() < 0.5:
            ref = {"fr": f"la facture {s['num']}", "en": f"invoice {s['num']}", "de": f"die Rechnung {s['num']}"}[lang]
        else:
            ref = {"fr": f"la facture de {s['sup']} de {month_label('fr', s['month'])}",
                   "en": f"the {month_label('en', s['month'])} invoice from {s['sup']}",
                   "de": f"die Rechnung von {s['sup']} aus dem {month_label('de', s['month'])}"}[lang]
        item = s["item"][lang]
        T = {
            "inv_total": {"fr": f"Quel est le montant total TTC de {ref} ?", "en": f"How much is the total, VAT included, on {ref}?",
                          "de": f"Wie hoch ist der Bruttobetrag für {ref}?"},
            "inv_vat": {"fr": f"Combien de TVA est facturée sur {ref} ?", "en": f"How much VAT is charged on {ref}?",
                        "de": f"Wie viel Mehrwertsteuer wird auf {ref} berechnet?"},
            "inv_due": {"fr": f"Avant quelle date faut-il payer {ref} ?", "en": f"By what date must {ref} be paid?",
                        "de": f"Bis wann muss {ref} bezahlt werden?"},
            "inv_price": {"fr": f"Quel prix unitaire est appliqué pour « {item} » sur {ref} ?", "en": f"What unit price is charged for {item} on {ref}?",
                          "de": f"Welcher Einzelpreis wird für {item} auf {ref} berechnet?"},
            "inv_date": {"fr": f"À quelle date a été émise {ref} ?", "en": f"On what date was {ref} issued?",
                         "de": f"An welchem Datum wurde {ref} ausgestellt?"},
        }
        return T[k][lang]
    sup = s.get("sup", "")
    T = {
        "svc_start": {"fr": f"À quelle date commence le contrat de services avec {sup} ?", "en": f"When does the service agreement with {sup} start?",
                      "de": f"Wann beginnt der Dienstleistungsvertrag mit {sup}?"},
        "svc_duration": {"fr": f"Pour combien de temps est conclu le contrat de services avec {sup} ?", "en": f"How long is the service agreement with {sup} concluded for?",
                         "de": f"Für welchen Zeitraum ist der Dienstleistungsvertrag mit {sup} abgeschlossen?"},
        "svc_pay": {"fr": f"Sous quel délai faut-il payer les factures de {sup} selon le contrat ?", "en": f"Within how many days must {sup}'s invoices be paid under the agreement?",
                    "de": f"Innerhalb welcher Frist sind Rechnungen von {sup} laut Vertrag zu zahlen?"},
        "svc_rate": {"fr": f"Quel est le taux horaire facturé par {sup} ?", "en": f"What hourly rate does {sup} charge?",
                     "de": f"Welchen Stundensatz berechnet {sup}?"},
        "svc_notice": {"fr": f"Quel préavis faut-il respecter pour résilier le contrat avec {sup} ?", "en": f"What notice period applies to terminate the agreement with {sup}?",
                       "de": f"Welche Kündigungsfrist gilt für den Vertrag mit {sup}?"},
        "lease_rent": {"fr": "Combien payons-nous de loyer mensuel pour nos bureaux ?", "en": "How much monthly rent do we pay for our offices?",
                       "de": "Wie viel Monatsmiete zahlen wir für unsere Büros?"},
        "lease_deposit": {"fr": "À combien de mois de loyer correspond la garantie locative des bureaux ?", "en": "How many months of rent is the office security deposit?",
                          "de": "Wie viele Monatsmieten beträgt die Kaution für die Büros?"},
        "tax_vat_due": {"fr": "Combien de TVA devons-nous pour le deuxième trimestre 2025 ?", "en": "How much VAT do we owe for Q2 2025?",
                        "de": "Wie viel Umsatzsteuer schulden wir für das zweite Quartal 2025?"},
        "tax_vat_deadline": {"fr": "Avant quelle date faut-il payer la TVA du deuxième trimestre 2025 ?", "en": "What is the payment deadline for the Q2 2025 VAT?",
                             "de": "Bis wann ist die Umsatzsteuer für das zweite Quartal 2025 zu zahlen?"},
        "tax_objection": {"fr": "De combien de temps disposons-nous pour contester le décompte de TVA ?", "en": "How long do we have to object to the VAT assessment?",
                          "de": "Wie lange haben wir Zeit, gegen den Umsatzsteuerbescheid Einspruch einzulegen?"},
        "tax_advance": {"fr": "Quel est le montant de chaque avance trimestrielle d'impôt pour 2025 ?", "en": "How much is each quarterly corporate tax advance for 2025?",
                        "de": "Wie hoch ist jede vierteljährliche Steuervorauszahlung für 2025?"},
        "tax_advance_due": {"fr": "Quand faut-il payer la prochaine avance d'impôt ?", "en": "When is the next tax advance due?",
                            "de": "Wann ist die nächste Steuervorauszahlung fällig?"},
        "tax_penalty": {"fr": "Quelle pénalité avons-nous reçue pour le retard de la déclaration 2024 ?", "en": "What penalty did we get for filing the 2024 return late?",
                        "de": "Welchen Zuschlag haben wir für die verspätete Steuererklärung 2024 bekommen?"},
        "proc_threshold": {"fr": "À partir de quel montant une note de frais doit-elle être approuvée par le directeur financier ?",
                           "en": "Above what amount does an expense claim need the CFO's sign-off?",
                           "de": "Ab welchem Betrag muss der Finanzleiter eine Reisekostenabrechnung genehmigen?"},
        "proc_submit": {"fr": "Combien de jours ont les salariés pour déposer une note de frais ?", "en": "How many days do employees have to submit an expense claim?",
                        "de": "Wie viele Tage haben Mitarbeitende, um eine Reisekostenabrechnung einzureichen?"},
        "proc_reimb": {"fr": "En combien de jours une note de frais validée est-elle remboursée ?", "en": "How quickly is an approved expense claim reimbursed?",
                       "de": "Wie schnell wird eine freigegebene Reisekostenabrechnung erstattet?"},
        "proc_retention": {"fr": "Combien de temps faut-il garder les justificatifs de frais ?", "en": "How long must expense receipts be kept?",
                           "de": "Wie lange müssen Reisekostenbelege aufbewahrt werden?"},
        "proc_bank": {"fr": "En combien de jours ouvrés les rapprochements bancaires doivent-ils être finis à la clôture ?",
                      "en": "How many working days after month end do we have to finish bank reconciliations?",
                      "de": "Innerhalb wie vieler Arbeitstage nach Monatsende muss die Bankabstimmung fertig sein?"},
        "mail_audit": {"fr": "Combien coûterait l'audit légal des comptes 2025 selon l'offre reçue ?", "en": "What fee were we quoted for the 2025 statutory audit?",
                       "de": "Welches Honorar wurde uns für die Abschlussprüfung 2025 angeboten?"},
        "mail_meeting": {"fr": "Quel jour est prévue la réunion de clôture annuelle avec l'expert-comptable ?", "en": "What day is the year-end closing meeting with the accountant?",
                         "de": "An welchem Tag ist der Termin zum Jahresabschluss mit dem Steuerberater?"},
    }
    return T[k][lang]


def unanswerable(rng: random.Random, gen: Generator) -> list[dict]:
    """~20 % of the bench: the corpus of the asking company does not contain the answer."""
    out = []
    langs = ["fr", "en", "de"]
    others = {c: [o for o in COMPANIES if o != c] for c in COMPANIES}
    for c in COMPANIES:
        sups = [s[0] for s in SUPPLIERS[c]]
        # 1. value types that never appear (IBAN, phone)
        for sup in rng.sample(sups, 2):
            lang = rng.choice(langs)
            out.append(dict(company=c, lang=lang, reason="absent_value", question={
                "fr": f"Quel est l'IBAN de {sup} ?", "en": f"What is {sup}'s IBAN?", "de": f"Wie lautet die IBAN von {sup}?"}[lang]))
        lang = rng.choice(langs)
        out.append(dict(company=c, lang=lang, reason="absent_value", question={
            "fr": "Quel est le numéro de téléphone de l'administration fiscale ?", "en": "What is the tax office's phone number?",
            "de": "Wie lautet die Telefonnummer der Steuerverwaltung?"}[lang]))
        # 2. another company's invoice: must be refused AND must not leak
        for o in others[c]:
            for num in rng.sample(gen.invoice_numbers[o], 2 if o == others[c][0] else 1):
                lang = rng.choice(langs)
                out.append(dict(company=c, lang=lang, reason="other_company", question={
                    "fr": f"Quel est le montant total de la facture {num} ?", "en": f"What is the total of invoice {num}?",
                    "de": f"Wie hoch ist der Gesamtbetrag der Rechnung {num}?"}[lang]))
        # 3. periods that are not covered
        for q in [{"fr": "Combien de TVA devons-nous pour le quatrième trimestre 2024 ?", "en": "How much VAT do we owe for Q4 2024?",
                   "de": "Wie viel Umsatzsteuer schulden wir für das vierte Quartal 2024?"},
                  {"fr": "Quelle pénalité avons-nous reçue pour la déclaration 2022 ?", "en": "What penalty did we get for the 2022 return?",
                   "de": "Welchen Verspätungszuschlag gab es für die Steuererklärung 2022?"}]:
            lang = rng.choice(langs)
            out.append(dict(company=c, lang=lang, reason="absent_period", question=q[lang]))
        # 4. topics that do not exist at all
        for q in [{"fr": "Quel est le loyer mensuel du leasing de nos véhicules ?", "en": "What is the monthly fee of our vehicle leasing contract?",
                   "de": "Wie hoch ist die monatliche Rate unseres Fahrzeugleasings?"},
                  {"fr": "Quel est le montant de notre prime d'assurance responsabilité civile ?", "en": "How much is our liability insurance premium?",
                   "de": "Wie hoch ist unsere Haftpflichtversicherungsprämie?"}]:
            lang = rng.choice(langs)
            out.append(dict(company=c, lang=lang, reason="absent_topic", question=q[lang]))
    return out


def build_questions(gen: Generator, rng: random.Random) -> list[dict]:
    qs = []
    langs = ["fr", "en", "de"]
    # Keep the invoice facts to 2 per invoice so invoices do not swamp the bench.
    by_doc: dict[str, list[dict]] = {}
    for f in gen.facts:
        by_doc.setdefault(f["doc_id"], []).append(f)
    chosen = []
    for doc_id in sorted(by_doc):
        fs = by_doc[doc_id]
        chosen += rng.sample(fs, 2) if "-inv-" in doc_id else fs
    for f in chosen:
        lang = rng.choice(langs)
        qs.append(dict(company=f["company"], lang=lang, question=render_fact(rng, f, lang), answerable=True,
                       kind=f["kind"], type=f["type"], expected=[f["expected"]],
                       sources=[[f["doc_id"], f["heading"]]]))
    for x in gen.extra_q:
        for lang in langs:
            if x["kind"] == "conflict":
                q = {"rate": {"fr": f"Quel taux horaire avons-nous convenu avec {x['sup']} ?", "en": f"What hourly rate did we agree with {x['sup']}?",
                              "de": f"Welchen Stundensatz haben wir mit {x['sup']} vereinbart?"},
                     "notice": {"fr": f"Quel est le préavis de résiliation du contrat avec {x['sup']} ?", "en": f"What is the notice period of the agreement with {x['sup']}?",
                                "de": f"Wie lang ist die Kündigungsfrist im Vertrag mit {x['sup']}?"},
                     "penalty": {"fr": f"Quelle pénalité de retard est prévue avec {x['sup']} ?", "en": f"What late-delivery penalty applies with {x['sup']}?",
                                 "de": f"Wie hoch ist die Vertragsstrafe mit {x['sup']}?"}}[x["field"]][lang]
            else:
                q = {"fr": "Quand la livraison de Brightmoor Supply est-elle prévue ?", "en": "When is the Brightmoor Supply delivery scheduled?",
                     "de": "Wann ist die Lieferung von Brightmoor Supply geplant?"}[lang]
            qs.append(dict(company=x["company"], lang=lang, question=q, answerable=True, kind=x["kind"], type=x["type"],
                           expected=x["expected"], sources=[list(s) for s in x["sources"]]))
    for u in unanswerable(rng, gen):
        qs.append(dict(company=u["company"], lang=u["lang"], question=u["question"], answerable=False,
                       kind="unanswerable:" + u["reason"], type=None, expected=[], sources=[]))
    rng.shuffle(qs)
    for i, q in enumerate(qs):
        q["id"] = f"q{i + 1:03d}"
        q["split"] = "dev" if i % 3 == 0 else "test"   # dev: tune thresholds, test: report
    return qs


def main():
    gen = Generator()
    gen.build()
    qrng = random.Random(SEED + 1)
    questions = build_questions(gen, qrng)
    for d in gen.docs:
        p = CORPUS / d["company"] / f"{d['doc_id']}.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        fm = "\n".join(f"{k}: {v}" for k, v in d["meta"].items())
        p.write_text(f"---\n{fm}\n---\n{d['body']}\n", encoding="utf-8")
    LABELS.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in gen.labels), encoding="utf-8")
    QUESTIONS.parent.mkdir(parents=True, exist_ok=True)
    keys = ["id", "split", "company", "lang", "question", "answerable", "kind", "type", "expected", "sources"]
    QUESTIONS.write_text("".join(json.dumps({k: q[k] for k in keys}, ensure_ascii=False) + "\n" for q in questions),
                         encoding="utf-8")
    n_un = sum(not q["answerable"] for q in questions)
    print(f"{len(gen.docs)} documents, {len(gen.labels)} labelled sections, "
          f"{len(questions)} questions ({n_un} without answer, {100 * n_un / len(questions):.0f} %)")


if __name__ == "__main__":
    main()
