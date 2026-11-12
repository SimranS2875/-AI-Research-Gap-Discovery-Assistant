"""Text cleaning and quote verification helpers."""
import re
import unicodedata
from typing import List

_REF_HEADING = re.compile(
    r"^\s*(?:\d+[\.\)]?\s+|[IVX]+\.\s+)?(references|bibliography|literature cited)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_pdf_text(raw: str) -> str:
    """Normalize raw text extracted from a PDF."""
    text = raw.replace("\x00", " ")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated at line ends
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_references(text: str) -> str:
    """Drop the reference list at the end of a paper (only if it is in the last 60%)."""
    matches = [m for m in _REF_HEADING.finditer(text) if m.start() > 0.4 * len(text)]
    if not matches:
        return text
    return text[: matches[-1].start()].rstrip()


def truncate_middle(text: str, max_chars: int, head_fraction: float = 0.6) -> str:
    """Keep the start (abstract/methods) and the end (conclusion/limitations) of long texts."""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * head_fraction)
    tail = max_chars - head
    return text[:head] + "\n\n[... MIDDLE OF PAPER OMITTED FOR LENGTH ...]\n\n" + text[-tail:]


def _alnum(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"[\W_]+", "", s)


_ELLIPSIS = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]|\.\.\.|…")


class QuoteVerifier:
    """Checks that a quote really occurs in a paper (ignoring spacing, case, punctuation)."""

    def __init__(self, text: str):
        self._hay = _alnum(text)

    def verify(self, quote: str) -> bool:
        if not quote or not quote.strip():
            return False
        parts: List[str] = [_alnum(p) for p in _ELLIPSIS.split(quote)]
        parts = [p for p in parts if p]
        if not parts:
            return False
        return all(p in self._hay for p in parts)
