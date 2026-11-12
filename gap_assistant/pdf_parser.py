"""PDF -> clean text."""
from dataclasses import dataclass

from .text_utils import clean_pdf_text, strip_references

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # pragma: no cover
    import fitz  # older PyMuPDF


class PdfError(Exception):
    """Raised when a PDF cannot be turned into usable text."""


@dataclass
class ParsedPdf:
    text: str
    title: str
    n_pages: int


MIN_TEXT_CHARS = 300


def parse_pdf(data: bytes, filename: str = "") -> ParsedPdf:
    """Extract text from PDF bytes. Scanned (image-only) PDFs are rejected: OCR is not included."""
    label = filename or "PDF"
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise PdfError(f"{label}: cannot open as a PDF ({e})") from e
    try:
        if doc.needs_pass:
            raise PdfError(f"{label}: the PDF is password-protected")
        pages = [page.get_text("text") for page in doc]
        title = ((doc.metadata or {}).get("title") or "").strip()
        n_pages = len(pages)
    finally:
        doc.close()

    text = strip_references(clean_pdf_text("\n".join(pages)))
    if len(text) < MIN_TEXT_CHARS:
        raise PdfError(
            f"{label}: almost no extractable text. It may be a scanned PDF (OCR is not supported) "
            "or an unusual layout."
        )
    return ParsedPdf(text=text, title=title, n_pages=n_pages)
