"""Command-line interface.

Example:
    python cli.py papers/ --topic "retinal vessel segmentation" --out report.md
    python cli.py a.pdf b.pdf c.pdf --no-novelty-check
"""
import argparse
import json
import sys
from pathlib import Path

from gap_assistant.config import load_settings
from gap_assistant.llm import GeminiClient, LLMError
from gap_assistant.pipeline import load_pdf_paths, run_analysis
from gap_assistant.report import to_markdown


def collect_pdfs(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(p.glob("*.pdf")) + sorted(p.glob("*.PDF"))
        elif p.is_file():
            files.append(p)
        else:
            print(f"Warning: {p} not found", file=sys.stderr)
    seen, unique = set(), []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            unique.append(f)
    return unique


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find research gaps across a set of papers.")
    ap.add_argument("inputs", nargs="+", help="PDF files and/or folders containing PDFs")
    ap.add_argument("--topic", default="", help="research topic (optional)")
    ap.add_argument("--out", default="research_gap_report.md", help="Markdown report path")
    ap.add_argument("--json", dest="json_out", default="", help="also save full results as JSON")
    ap.add_argument("--model", default="", help="Gemini model (default: auto-detect / GEMINI_MODEL)")
    ap.add_argument("--gap-model", default="", help="model for gap reasoning (default: same as --model)")
    ap.add_argument("--embeddings", choices=["tfidf", "gemini"], default="", help="similarity method")
    ap.add_argument("--max-gaps", type=int, default=0)
    ap.add_argument("--no-novelty-check", action="store_true", help="skip the Semantic Scholar check")
    args = ap.parse_args(argv)

    settings = load_settings()
    settings.model = args.model or settings.model
    settings.gap_model = args.gap_model or settings.gap_model
    settings.embedding_backend = args.embeddings or settings.embedding_backend
    settings.max_gaps = args.max_gaps or settings.max_gaps
    settings.novelty_check = not args.no_novelty_check

    pdfs = collect_pdfs(args.inputs)
    inputs, errors = load_pdf_paths(pdfs)
    for e in errors:
        print(f"Warning: {e}", file=sys.stderr)
    if len(inputs) < 2:
        print("Need at least 2 readable PDFs.", file=sys.stderr)
        return 2

    try:
        client = GeminiClient(settings.api_key, model=settings.model, embedding_model=settings.embedding_model)
        result = run_analysis(client, inputs, args.topic, settings,
                              progress=lambda f, m: print(f"[{int(f * 100):3d}%] {m}", flush=True))
    except LLMError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    Path(args.out).write_text(to_markdown(result), encoding="utf-8")
    print(f"Report written to {args.out}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
        print(f"Data written to {args.json_out}")
    for w in result.warnings:
        print(f"Note: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
