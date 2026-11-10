"""Text utilities: BM25 tokenizer, typed value extraction (amounts, dates, durations,
percentages), anchors, and sentence/clause splitting.

Value extraction is tolerant to the OCR noise found in scans (lost accents, l→1, m→rn),
because a date written "1er avri1 2025" on a scanned letter is still a date."""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

STOPWORDS = set("""
le la les un une des de du d l et ou à a au aux en dans pour par sur avec est sont que qui quel quelle quels quelles
ce cette ces nos notre votre vos il elle on nous vous faut-il doit être été avons
the a an of to in on for by with is are was were be what which who how when do does did our we you it this that at from
der die das den dem des ein eine einer einem und oder zu im in am an auf mit für von ist sind wie was wer wann welche
welcher welchen welches wir unser unsere ihr sie es bei bis aus
""".split())


def fold(s: str) -> str:
    """Lower-case and strip accents."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


ID_RE = re.compile(r"\b[A-Z]{2,4}-(?:FICT-)?\d{3,5}(?:-\d{3})?\b")


def tokenize(text: str) -> list[str]:
    """Tokens for BM25: folded words (stop-words removed) plus whole identifiers."""
    ids = [m.group(0).lower() for m in ID_RE.finditer(text)]
    words = [w for w in re.findall(r"[a-z0-9]+", fold(text)) if w not in STOPWORDS and len(w) > 1]
    return words + ids


# ----------------------------------------------------------------------------- OCR tolerance
def ocr_pattern(word: str) -> str:
    """Regex matching a word as a scanner may have damaged it."""
    out = []
    for c in fold(word):
        if c == "l":
            out.append("[l1]")
        elif c == "m":
            out.append("(?:m|rn)")
        else:
            out.append(re.escape(c))
    return "".join(out)


MONTHS = {
    1: ["janvier", "january", "januar"], 2: ["fevrier", "february", "februar"], 3: ["mars", "march", "marz"],
    4: ["avril", "april"], 5: ["mai", "may"], 6: ["juin", "june", "juni"], 7: ["juillet", "july", "juli"],
    8: ["aout", "august"], 9: ["septembre", "september"], 10: ["octobre", "october", "oktober"],
    11: ["novembre", "november"], 12: ["decembre", "december", "dezember"],
}
_MONTH_ALT = {}
for n, names in MONTHS.items():
    for name in names:
        _MONTH_ALT[name] = n
_MONTH_RE = "|".join(sorted((ocr_pattern(m) for m in _MONTH_ALT), key=len, reverse=True))


def _month_number(token: str) -> int | None:
    t = fold(token).replace("rn", "m").replace("1", "l")
    return _MONTH_ALT.get(t)


# ----------------------------------------------------------------------------- values
NUM = r"(?<![\d.,])(?:\d{1,3}(?:[  .,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)(?![\d])"
AMOUNT_RE = re.compile(rf"(?:(?:EUR|€)\s?({NUM}))|(?:({NUM})\s?(?:EUR|€|euros?)\b)", re.I)
PCT_RE = re.compile(r"(?<![\d.,])(\d+(?:[.,]\d+)?)\s?%")
DUR_UNITS = {
    "day": r"jours?|days?|tagen?|tage|arbeitstagen?|working days",
    "month": r"mois|months?|monaten?|monate|monatsmieten?",
    "year": r"ans|annees|years?|jahren?|jahre",
    "week": r"semaines?|weeks?|wochen?",
}
DUR_RE = re.compile(r"(?<![\d.,])(\d+)\s?(" + "|".join(DUR_UNITS.values()) + r")\b", re.I)
DATE_TEXT_RE = re.compile(rf"(?<!\d)(\d{{1,2}})(?:er|\.)?\s+({_MONTH_RE})\s+(\d{{4}})", re.I)
DATE_EN_RE = re.compile(rf"({_MONTH_RE})\s+(\d{{1,2}}),\s+(\d{{4}})", re.I)
DATE_DOT_RE = re.compile(r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{4})")
DATE_SLASH_RE = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4})")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}\b")
PHONE_RE = re.compile(r"\+\d{2,3}[\d \-/]{6,}\d")


def norm_amount(raw: str) -> str | None:
    s = re.sub(r"[  ]", "", raw)
    if "." in s and "," in s:
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        s = s.replace("," if dec == "." else ".", "").replace(dec, ".")
    elif "," in s or "." in s:
        sep = "," if "," in s else "."
        head, _, tail = s.rpartition(sep)
        s = f"{head.replace(sep, '')}.{tail}" if len(tail) <= 2 else s.replace(sep, "")
    try:
        return f"{Decimal(s):.2f}"
    except InvalidOperation:
        return None


def _iso(y: int, m: int, d: int) -> str | None:
    if 1 <= m <= 12 and 1 <= d <= 31:
        return f"{y:04d}-{m:02d}-{d:02d}"
    return None


def extract_values(text: str) -> list[dict]:
    """All typed values in `text`, in reading order: {type, value, raw, start, ambiguous?}."""
    out = []
    for m in AMOUNT_RE.finditer(text):
        v = norm_amount(m.group(1) or m.group(2))
        if v:
            out.append({"type": "amount", "value": v, "raw": m.group(0), "start": m.start()})
    for m in PCT_RE.finditer(text):
        out.append({"type": "percent", "value": f"{float(m.group(1).replace(',', '.')):g}%", "raw": m.group(0), "start": m.start()})
    for m in DUR_RE.finditer(text):
        unit = next(u for u, pat in DUR_UNITS.items() if re.fullmatch(pat, fold(m.group(2))))
        out.append({"type": "duration", "value": f"{int(m.group(1))} {unit}", "raw": m.group(0), "start": m.start()})
    folded = fold(text)
    ftext = folded if len(folded) == len(text) else text  # month names are matched without accents
    for m in DATE_TEXT_RE.finditer(ftext):
        mo = _month_number(m.group(2))
        v = mo and _iso(int(m.group(3)), mo, int(m.group(1)))
        if v:
            out.append({"type": "date", "value": v, "raw": text[m.start():m.end()], "start": m.start()})
    for m in DATE_EN_RE.finditer(ftext):
        mo = _month_number(m.group(1))
        v = mo and _iso(int(m.group(3)), mo, int(m.group(2)))
        if v:
            out.append({"type": "date", "value": v, "raw": text[m.start():m.end()], "start": m.start()})
    for m in DATE_DOT_RE.finditer(text):
        v = _iso(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if v:
            out.append({"type": "date", "value": v, "raw": m.group(0), "start": m.start()})
    for m in DATE_SLASH_RE.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        v = _iso(int(m.group(3)), b, a)  # read as DD/MM, the European convention
        if v:
            out.append({"type": "date", "value": v, "raw": m.group(0), "start": m.start(),
                        "ambiguous": a <= 12 and b <= 12 and a != b})
    for m in IBAN_RE.finditer(text):
        out.append({"type": "iban", "value": m.group(0).replace(" ", ""), "raw": m.group(0), "start": m.start()})
    for m in PHONE_RE.finditer(text):
        out.append({"type": "phone", "value": re.sub(r"\D", "", m.group(0)), "raw": m.group(0), "start": m.start()})
    # a percentage or a date is not also an amount/duration: drop overlaps, keep the first type found
    out.sort(key=lambda x: (x["start"], -len(x["raw"])))
    kept, end = [], -1
    for v in out:
        if v["start"] >= end:
            kept.append(v)
            end = v["start"] + len(v["raw"])
    return kept


# ----------------------------------------------------------------------------- question analysis
_TYPE_RULES = [
    ("iban", r"\biban\b"),
    ("phone", r"telephone|phone|telefon"),
    ("percent+amount", r"penalite|penalty|vertragsstrafe"),
    ("duration", r"combien de (temps|jours|mois)|pour combien de temps|en combien de jours|delai|preavis|how long|"
                 r"how many (working )?(days|months)|how quickly|within how many|notice period|wie lange|wie viele|"
                 r"wie schnell|innerhalb (welcher|wie)|frist|zeitraum|a combien de mois"),
    ("date", r"\bquand\b|quelle date|quel jour|\bwhen\b|what date|what day|deadline|\bwann\b|an welchem (tag|datum)|bis wann|echeance"),
    ("amount", r"montant|combien|prix|cout|loyer|honoraires|taux horaire|how much|price|\bcost|\bfee|\brent\b|\brate\b|"
               r"total|amount|wie hoch|wie viel|betrag|preis|miete|honorar|stundensatz|zuschlag|pramie|premium"),
]


def requested_types(question: str) -> set[str] | None:
    q = fold(question)
    for types, pat in _TYPE_RULES:
        if re.search(pat, q):
            return set(types.split("+"))
    return None


_ORD = {1: ["premier", "first", "erste", "erstes", "ersten"], 2: ["deuxieme", "second", "zweite", "zweiten"],
        3: ["troisieme", "third", "dritte", "dritten"], 4: ["quatrieme", "fourth", "vierte", "vierten"]}
_QUARTER_RES = {n: re.compile(r"\b(?:q|t)" + str(n) + r"\b|\b(?:" + "|".join(ocr_pattern(w) for w in ws) +
                              r")\s+(?:trimestre|quarter|quartal)", re.I) for n, ws in _ORD.items()}
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def anchors(question: str) -> dict:
    """Hard constraints a supporting passage must satisfy: identifiers, years, quarters."""
    folded = fold(question)
    return {
        "ids": sorted({m.group(0).lower() for m in ID_RE.finditer(question)}),
        "years": sorted(set(YEAR_RE.findall(question))),
        "quarters": sorted(n for n, r in _QUARTER_RES.items() if r.search(folded)),
    }


def satisfies(anch: dict, text: str) -> bool:
    folded = fold(text)
    if any(i not in folded for i in anch["ids"]):
        return False
    if any(y not in text for y in anch["years"]):
        return False
    return all(_QUARTER_RES[n].search(folded) for n in anch["quarters"])


def clauses(text: str) -> list[str]:
    """Split a passage into short clauses (sentence, then ', ' / '; ') for answer selection."""
    parts = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        parts += [p.strip() for p in re.split(r",\s(?=\D)|;\s", sent) if p.strip()]
    return [p for p in parts if len(p) > 3]
