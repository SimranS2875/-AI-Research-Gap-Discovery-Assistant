"""Text embeddings: offline TF-IDF (default) or Gemini embeddings, with automatic fallback."""
from typing import List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .llm import AuthError, LLMClient, QuotaExceededError


def tfidf_embed(texts: Sequence[str]) -> np.ndarray:
    """L2-normalised TF-IDF vectors (dense). Works offline."""
    texts = [t if t and t.strip() else "." for t in texts]
    try:
        mat = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts)
        return mat.toarray().astype(np.float32)
    except ValueError:  # empty vocabulary (e.g. only stop words)
        return np.eye(len(texts), dtype=np.float32)


class Embedder:
    """Embeds texts with the chosen backend. If Gemini embeddings fail, falls back to TF-IDF
    and records a warning."""

    def __init__(self, backend: str = "tfidf", client: Optional[LLMClient] = None):
        self.requested = backend if backend in ("tfidf", "gemini") else "tfidf"
        self.client = client
        self.last_backend = "tfidf"
        self.warnings: List[str] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self.requested == "gemini" and self.client is not None:
            try:
                vecs = self.client.embed(texts)
                self.last_backend = "gemini"
                return vecs
            except (AuthError, QuotaExceededError) as e:
                self._warn(f"Gemini embeddings unavailable ({e}); used offline TF-IDF instead.")
            except Exception as e:  # any provider problem should not break the analysis
                self._warn(f"Gemini embeddings failed ({str(e)[:150]}); used offline TF-IDF instead.")
        self.last_backend = "tfidf"
        return tfidf_embed(texts)

    def _warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)
