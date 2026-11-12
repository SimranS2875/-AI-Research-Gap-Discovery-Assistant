"""End-to-end pipeline: papers in -> structured records -> comparison -> ranked research gaps."""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .compare import ComparisonResult, build_comparison
from .config import Settings
from .embeddings import Embedder
from .extract import extract_paper
from .gaps import check_novelty, generate_gaps
from .llm import AuthError, LLMClient, QuotaExceededError
from .pdf_parser import PdfError, parse_pdf
from .schemas import Gap, Paper

ProgressFn = Callable[[float, str], None]


@dataclass
class PaperInput:
    """One paper to analyse: its text plus where it came from."""
    name: str  # file name or title
    text: str
    source_type: str = "full_text"  # "full_text" or "abstract"
    url: str = ""
    title_hint: str = ""


@dataclass
class AnalysisResult:
    topic: str = ""
    papers: List[Paper] = field(default_factory=list)
    texts: Dict[str, str] = field(default_factory=dict)  # paper_id -> text used for quote checks
    comparison: ComparisonResult = field(default_factory=ComparisonResult)
    gaps: List[Gap] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    model: str = ""
    gap_model: str = ""

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "model": self.model,
            "gap_model": self.gap_model,
            "papers": [p.model_dump() for p in self.papers],
            "comparison": self.comparison.to_dict(),
            "gaps": [g.model_dump() for g in self.gaps],
            "warnings": self.warnings,
        }


def load_pdf_input(name: str, data: bytes) -> PaperInput:
    """Parse PDF bytes into a PaperInput. Raises PdfError with a readable message."""
    parsed = parse_pdf(data, name)
    return PaperInput(name=name, text=parsed.text, title_hint=parsed.title)


def load_pdf_paths(paths: List[Path]) -> Tuple[List[PaperInput], List[str]]:
    inputs, errors = [], []
    for path in paths:
        try:
            inputs.append(load_pdf_input(Path(path).name, Path(path).read_bytes()))
        except (PdfError, OSError) as e:
            errors.append(str(e))
    return inputs, errors


def _noop(_frac: float, _msg: str) -> None:
    pass


def run_analysis(client: LLMClient, inputs: List[PaperInput], topic: str, settings: Settings,
                 progress: Optional[ProgressFn] = None) -> AnalysisResult:
    progress = progress or _noop
    result = AnalysisResult(topic=topic.strip())

    # Resolve the model first so a bad key / unknown model fails immediately.
    result.model = settings.model or client.default_model()
    result.gap_model = settings.gap_model or result.model

    # ---- 1. extraction (one LLM call per paper, cached on disk)
    unique, seen = [], set()
    for inp in inputs:
        digest = hashlib.sha256(inp.text.encode("utf-8", errors="ignore")).hexdigest()
        if digest in seen:
            result.warnings.append(f"Skipped '{inp.name}': identical text to an earlier paper.")
            continue
        seen.add(digest)
        unique.append(inp)

    for i, inp in enumerate(unique, 1):
        progress(0.6 * (i - 1) / max(len(unique), 1), f"Reading paper {i}/{len(unique)}: {inp.name}")
        paper_id = f"P{len(result.papers) + 1}"
        try:
            paper = extract_paper(
                client, inp.text, source=inp.name, paper_id=paper_id, model=result.model,
                source_type=inp.source_type, url=inp.url, fallback_title=inp.title_hint or inp.name,
                max_chars=settings.max_chars_per_paper, cache_dir=settings.cache_dir)
        except AuthError:
            raise
        except QuotaExceededError as e:
            result.warnings.append(f"Stopped after {len(result.papers)} paper(s): {e}")
            break
        except Exception as e:
            result.warnings.append(f"Could not analyse '{inp.name}': {str(e)[:300]}")
            continue
        result.papers.append(paper)
        result.texts[paper_id] = inp.text

    if len(result.papers) < 2:
        result.warnings.append("At least 2 papers are needed to compare them and find gaps.")
        progress(1.0, "Done")
        return result

    # ---- 2. comparison
    progress(0.62, "Comparing papers")
    embedder = Embedder(settings.embedding_backend, client)
    result.comparison = build_comparison(result.papers, embedder)
    result.warnings.extend(embedder.warnings)

    # ---- 3. gap generation
    progress(0.72, "Reasoning about research gaps")
    try:
        result.gaps, gap_warnings = generate_gaps(
            client, result.papers, result.texts, result.comparison, topic=result.topic,
            max_gaps=settings.max_gaps, model=result.gap_model)
        result.warnings.extend(gap_warnings)
    except AuthError:
        raise
    except Exception as e:
        result.warnings.append(f"Gap generation failed: {str(e)[:300]}. Your extracted papers and "
                               "comparison are still available; run again (extractions are cached).")
        progress(1.0, "Done")
        return result

    # ---- 4. novelty check (optional)
    if settings.novelty_check and result.gaps:
        progress(0.86, "Checking whether the gaps are already addressed")
        result.warnings.extend(check_novelty(client, result.gaps, topic=result.topic,
                                             model=result.gap_model))
    progress(1.0, "Done")
    return result
