"""Central configuration, read from environment variables (and an optional .env file)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

# Bump when the extraction schema/prompt changes, so old cache entries are ignored.
SCHEMA_VERSION = "1"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass
class Settings:
    api_key: str = ""
    model: str = ""  # "" = auto-detect a current Flash model
    gap_model: str = ""  # "" = same as `model`
    embedding_backend: str = "tfidf"  # "tfidf" (offline, free) or "gemini"
    embedding_model: str = ""  # "" = auto-detect
    max_chars_per_paper: int = 120_000
    max_gaps: int = 8
    novelty_check: bool = True
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / ".cache")


def load_settings() -> Settings:
    """Build Settings from environment variables."""
    return Settings(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "",
        model=os.getenv("GEMINI_MODEL", ""),
        gap_model=os.getenv("GEMINI_GAP_MODEL", ""),
        embedding_backend=(os.getenv("EMBEDDING_BACKEND", "tfidf") or "tfidf").lower(),
        embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", ""),
        max_chars_per_paper=_int_env("MAX_CHARS_PER_PAPER", 120_000),
        max_gaps=_int_env("MAX_GAPS", 8),
    )
