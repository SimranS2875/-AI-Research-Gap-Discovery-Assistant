# AI Research Gap Discovery Assistant

Upload research papers (PDF) or find them by topic. The assistant reads each paper, compares them, and proposes
**ranked research gaps backed by quotes from the papers** — so you can verify every claim instead of trusting the AI.

## How it works

1. **Read** – PDF text is extracted (PyMuPDF). Gemini turns each paper into structured data: problem, methods, datasets,
   metrics, key results, limitations, future work, assumptions. Every item carries a quote copied from the paper.
2. **Verify** – the code checks each quote really appears in the paper text (✅ / ⚠️). Invented quotes are flagged.
3. **Compare** – method × dataset matrix, popular combinations nobody tried, limitations / future work repeated by
   several papers (clustering), and a 2-D map of how similar the papers are.
4. **Find gaps** – Gemini proposes gaps from that evidence. The code drops gaps citing papers that don't exist,
   re-checks quotes, and ranks by novelty, feasibility and how many papers support each gap.
5. **Check novelty** – each gap is searched on Semantic Scholar and judged "open / partly / already addressed".
6. **Ask & export** – chat about the papers; download a Markdown report or JSON.

## Setup (Windows)

1. Install Python 3.10+ (python.org). Get a free Gemini API key at https://aistudio.google.com.
2. Double-click `setup.bat` (creates a virtual environment, installs packages, creates `.env`).
3. Open `.env` in Notepad and paste your key after `GEMINI_API_KEY=`.
4. Double-click `run_app.bat`. The app opens in your browser.

Manual alternative: `python -m venv .venv`, activate it, `pip install -r requirements.txt`, copy `.env.example` to `.env`,
then `python -m streamlit run app.py`.

## Command line

```
python cli.py papers_folder --topic "your topic" --out report.md --json data.json
python cli.py a.pdf b.pdf --no-novelty-check --max-gaps 5
```

## Tips and limits

- Use **3–15 papers on one narrow topic**. Gaps are relative to the papers you give it; the novelty check is a rough screen.
- **Free-tier quota is small** and changes. Each paper costs one request, plus one for gaps, one for the novelty
  judgement and one per chat question. Results are cached in `.cache/`, so re-running never repeats finished papers.
  If you hit the daily limit, switch the model in the sidebar to a Flash-Lite model or wait for the reset.
- **Rate limiting:** requests to Gemini are spaced 13 s apart by default (stays under the free tier's 5 requests/minute). Set `GEMINI_MIN_INTERVAL_S=0` in `.env` if you have a paid key.
- Model names change often. Leave **Model** blank to auto-detect, or set `GEMINI_MODEL` in `.env`.
- Scanned PDFs (images only) are not supported — there is no OCR.
- Abstract-only papers (from search, when no open-access PDF exists) give weaker results.
- Treat every gap as a lead to verify, not a conclusion.
- Your PDFs' text is sent to Google's Gemini API; don't use confidential documents.

## Project layout

```
app.py                  Streamlit interface
cli.py                  command-line interface
gap_assistant/
  pdf_parser.py         PDF -> text
  extract.py            per-paper structured extraction (+ cache, quote check)
  compare.py            matrix, untried combinations, recurring limitations, paper map
  gaps.py               gap generation, validation, ranking, novelty check
  search.py             Semantic Scholar search / PDF download
  llm.py                Gemini wrapper (retries, quota handling, model auto-detect)
  pipeline.py, report.py, schemas.py, embeddings.py, text_utils.py, config.py
tests/                  25+ tests using a fake LLM (no key or network needed)
```

Run tests: `pip install -r requirements-dev.txt` then `python -m pytest`.

## Ideas to extend (for your report)

OCR for scanned PDFs, citation-graph expansion, trend analysis by year, evaluation against gaps listed in published
survey papers, comparison with a plain-LLM baseline, support for other LLM providers (implement `LLMClient`).
