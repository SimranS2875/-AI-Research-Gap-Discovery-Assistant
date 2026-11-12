"""Semantic Scholar helpers: find papers by topic, download open-access PDFs, and look up
related work for the "has this gap already been addressed?" check."""
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import requests

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,authors,venue,citationCount,openAccessPdf,url"
USER_AGENT = "research-gap-assistant/1.0 (student project)"


class SearchError(Exception):
    pass


@dataclass
class SearchHit:
    paper_id: str
    title: str
    abstract: str
    year: Optional[int]
    authors: List[str]
    venue: str
    citations: Optional[int]
    url: str
    pdf_url: str


# Semantic Scholar's key terms allow 1 request per second across all endpoints. Every request
# to the S2 API goes through `_wait_for_slot`, so retries and parallel callers stay under it.
S2_MIN_INTERVAL_S = 1.1
_s2_lock = threading.Lock()
_s2_last = 0.0


def _wait_for_slot() -> None:
    global _s2_last
    with _s2_lock:
        wait = _s2_last + S2_MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _s2_last = time.monotonic()


def _headers() -> dict:
    headers = {"User-Agent": USER_AGENT}
    key = os.getenv("S2_API_KEY")  # optional: raises the rate limit
    if key:
        headers["x-api-key"] = key
    return headers


def clean_query(q: str) -> str:
    """Semantic Scholar's keyword search dislikes punctuation/operators."""
    q = re.sub(r"[^\w\s-]", " ", q or "")
    return re.sub(r"\s+", " ", q).strip()


def search_papers(query: str, limit: int = 10, year_from: Optional[int] = None,
                  session=None, retries: int = 3) -> List[SearchHit]:
    query = clean_query(query)
    if not query:
        raise SearchError("Empty search query.")
    http = session or requests
    params = {"query": query, "limit": max(1, min(limit, 50)), "fields": FIELDS}
    if year_from:
        params["year"] = f"{year_from}-"
    delay = 2.0
    for attempt in range(retries + 1):
        _wait_for_slot()
        try:
            resp = http.get(S2_SEARCH_URL, params=params, headers=_headers(), timeout=30)
        except requests.RequestException as e:
            raise SearchError(f"Could not reach Semantic Scholar: {e}") from e
        if resp.status_code == 429:
            if attempt == retries:
                raise SearchError("Semantic Scholar is rate-limiting requests. Wait a minute and retry "
                                  "(a free S2_API_KEY removes most limits).")
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code != 200:
            raise SearchError(f"Semantic Scholar returned HTTP {resp.status_code}.")
        break
    try:
        data = resp.json().get("data") or []
    except ValueError as e:
        raise SearchError("Semantic Scholar returned an unreadable response.") from e

    hits = []
    for d in data:
        title = (d.get("title") or "").strip()
        if not title:
            continue
        hits.append(SearchHit(
            paper_id=d.get("paperId") or "",
            title=title,
            abstract=(d.get("abstract") or "").strip(),
            year=d.get("year"),
            authors=[a.get("name", "") for a in (d.get("authors") or []) if a.get("name")],
            venue=(d.get("venue") or "").strip(),
            citations=d.get("citationCount"),
            url=d.get("url") or "",
            pdf_url=((d.get("openAccessPdf") or {}).get("url") or ""),
        ))
    return hits


def download_pdf(url: str, max_mb: int = 40, session=None) -> bytes:
    """Download a PDF (open-access link). Raises SearchError if it is not a real PDF."""
    http = session or requests
    try:
        resp = http.get(url, headers={"User-Agent": USER_AGENT}, timeout=45, stream=True)
        if resp.status_code != 200:
            raise SearchError(f"HTTP {resp.status_code} while downloading the PDF.")
        chunks, size = [], 0
        for chunk in resp.iter_content(chunk_size=65536):
            size += len(chunk)
            if size > max_mb * 1024 * 1024:
                raise SearchError(f"PDF larger than {max_mb} MB; skipped.")
            chunks.append(chunk)
    except requests.RequestException as e:
        raise SearchError(f"Download failed: {e}") from e
    data = b"".join(chunks)
    if not data.startswith(b"%PDF"):
        raise SearchError("The link did not return a PDF file (publisher page or paywall).")
    return data
