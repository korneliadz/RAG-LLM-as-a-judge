import re


# Początki treści przypisów w tekście jednolitym regulaminu PŁ.
_FOOTNOTE_START = (
    r"(?:Zdanie \w+ )?(?:[Ww] tym brzmieniu obowiązuje"
    r"|[Ww] brzmieniu ustalonym"
    r"|Dodan[ya] przez|dodane przez|Zdanie \w+ dodane"
    r"|Ze zmianą wprowadzoną"
    r"|Przez § "
    r"|Część wspólna"
    r"|Tekst jednolity wymienionej)"
)
# Przypis kończy się na jednym z tych zwrotów.
_FOOTNOTE_END = (
    r"(?:wejdzie w życie z dniem \d{1,2} \w+ \d{4} r\."
    r"|o której mowa w odnośniku \d+\."
    r"|poz\. [\d ,i]+?\.(?= ))"
)
_FOOTNOTE_RE = re.compile(
    r"(?<![\d.)])\d{1,3}\) " + _FOOTNOTE_START + r".*?" + _FOOTNOTE_END
)
_OLD_VERSION_RE = re.compile(
    r"(?<![\d.)])(\d{1,3})\) (?:Zdanie \w+ )?[Ww] tym brzmieniu obowiązuje do "
    r"wejścia w życie zmiany, o której mowa w odnośniku (\d+)\."
)


def _marker(n):
    """Znacznik przypisu doklejony do numeru ustępu/punktu, np. '6.107)' albo '23)15)'."""
    return re.compile(r"(?<=[.):;,~A-Za-ząćęłńóśźżĄĆĘŁŃÓŚŹŻ])" + n + r"\)")


def _split_glued_paragraphs(text):
    """'§ 23117)' = paragraf 23 + przypis 117 -> '§ 23~117)'."""
    footnotes = set(re.findall(r"(?<![\d.)])(\d{1,3})\) (?=[A-ZŁŚŻŹĆ])", text))
    footnotes |= set(re.findall(r"odnośniku (\d+)", text))

    def fix(m):
        digits = m.group(1)
        for k in (1, 2):
            para, fn = digits[:k], digits[k:]
            if fn and not fn.startswith("0") and fn in footnotes:
                return f"§ {para}~{fn})"
        return m.group(0)

    return re.sub(r"§\s*(\d{3,5})\)", fix, text)


def clean_regulamin(text):
    """Zostawia tylko brzmienie obowiązujące po zmianach (Uchwała 73/2026)
    i usuwa treść przypisów.

    W tekście jednolitym stare i nowe brzmienie stoją obok siebie:
        6.107) <stare brzmienie> ... 6.108) <nowe brzmienie>
    a przypis 107 mówi "W tym brzmieniu obowiązuje do wejścia w życie zmiany,
    o której mowa w odnośniku 108". Usuwamy tekst od znacznika 107 do znacznika 108.
    """
    text = _split_glued_paragraphs(text)
    pairs = _OLD_VERSION_RE.findall(text)
    footnote_spans = [m.span() for m in _FOOTNOTE_RE.finditer(text)]

    def in_footnote(pos):
        return any(a <= pos < b for a, b in footnote_spans)

    cut = []
    for old, new in pairs:
        old_m = next((m for m in _marker(old).finditer(text) if not in_footnote(m.start())), None)
        if old_m is None:
            continue
        new_m = next(
            (m for m in _marker(new).finditer(text, old_m.end()) if not in_footnote(m.start())),
            None,
        )
        if new_m is None:
            continue
        # Usuwamy od znacznika starego brzmienia do znacznika nowego
        # (numer ustępu/punktu przed znacznikiem zostaje).
        cut.append((old_m.start(), new_m.start()))

    cut += footnote_spans
    cut.sort()
    out, pos = [], 0
    for a, b in cut:
        if a < pos:
            a = pos
        if a >= b:
            continue
        out.append(text[pos:a])
        pos = b
    out.append(text[pos:])
    text = "".join(out)

    # Pozostałe znaczniki przypisów: "2.100) Jednostką" -> "2. Jednostką", "23)16) rok" -> "23) rok"
    text = re.sub(r"~\d{1,3}\)", "", text)
    text = re.sub(r"(?<=\d[.)])\d{1,3}\)", "", text)
    text = re.sub(r"(?<=[A-Za-ząćęłńóśźż:;,])\d{1,3}\)(?= )", "", text)
    # Resztki przypisów rozciętych przez łamanie stron
    text = re.sub(
        r"§ 1 pkt \d+(?: lit\. [a-z])?(?: tiret \w+)? uchwały, o której mowa w odnośniku \d+; "
        r"wejdzie w życie z dniem \d{1,2} \w+ \d{4} r\.(?: \d{1,3}(?= ))?",
        "",
        text,
    )
    # "(uchylony)" - po zmianach przepis nie istnieje
    text = re.sub(r"\b\d+[a-z]?[.)] \(uchylon[ya]\)", "", text)
    return re.sub(r"\s+", " ", text).strip()
