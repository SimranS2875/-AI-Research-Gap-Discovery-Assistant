import json
from types import SimpleNamespace

import numpy as np
import pytest

from gap_assistant import compare, gaps as gaps_mod, llm, report, search
from gap_assistant.config import Settings
from gap_assistant.embeddings import Embedder
from gap_assistant.extract import extract_paper
from gap_assistant.pdf_parser import PdfError, parse_pdf
from gap_assistant.pipeline import PaperInput, load_pdf_paths, run_analysis
from gap_assistant.schemas import Claim, Paper
from gap_assistant.text_utils import QuoteVerifier, clean_pdf_text, strip_references, truncate_middle
from tests.conftest import PAPERS, make_pdf


# ----------------------------------------------------------------------------- text utils
def test_quote_verifier_is_lenient_about_format_but_strict_about_content():
    v = QuoteVerifier("Our U-Net improves Dice by two\npoints over the previous baseline.\nAnother line.")
    assert v.verify("Our U-Net improves Dice by two points over the previous baseline.")
    assert v.verify("our unet improves dice ... previous baseline")  # ellipsis + case + punctuation
    assert not v.verify("Our U-Net improves Dice by ten points over the previous baseline.")
    assert not v.verify("")
    assert not v.verify("...")


def test_strip_references_and_truncate():
    body = "Intro text. " * 200
    text = body + "\nReferences\n[1] A. Author. Title.\n"
    assert strip_references(text).endswith("Intro text.")
    assert strip_references("References\n" + body) .startswith("References")  # heading at the very start is kept
    t = truncate_middle("a" * 1000 + "b" * 1000, 500)
    assert t.startswith("a") and t.endswith("b") and "OMITTED" in t
    assert clean_pdf_text("segmen-\ntation  models\n\n\n\nx") == "segmentation models\n\nx"


# ----------------------------------------------------------------------------- pdf
def test_parse_pdf_reads_text_and_drops_references(pdf_dir):
    parsed = parse_pdf((pdf_dir / "paper_a.pdf").read_bytes(), "paper_a.pdf")
    assert "U-Net improves Dice" in parsed.text.replace("\n", " ")
    assert "Some Reference" not in parsed.text
    assert parsed.title.startswith("UNetVessel")


def test_parse_pdf_errors(tmp_path):
    with pytest.raises(PdfError):
        parse_pdf(b"not a pdf", "x.pdf")
    empty = tmp_path / "empty.pdf"
    make_pdf(empty, "T", [])
    import pymupdf
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(empty))
    with pytest.raises(PdfError, match="scanned"):
        parse_pdf(empty.read_bytes(), "empty.pdf")


# ----------------------------------------------------------------------------- llm helpers
def test_choose_models():
    names = ["models/gemini-2.5-pro", "models/gemini-3.8-flash", "models/gemini-3.5-flash-lite",
             "models/gemini-3-flash-preview", "models/gemini-3.6-flash", "models/gemini-2.5-flash-image",
             "models/gemini-embedding-001", "models/gemini-embedding-2", "models/text-embedding-004"]
    assert llm.choose_generation_model(names) == "gemini-3.8-flash"
    assert llm.choose_generation_model(names + ["gemini-flash-latest"]) == "gemini-flash-latest"
    assert llm.choose_generation_model(["gemini-3-flash-preview"]) == "gemini-3-flash-preview"
    assert llm.choose_generation_model(["gemini-2.5-pro"]) is None
    assert llm.choose_embedding_model(names) == "gemini-embedding-2"


def test_parse_json_model_tolerates_fences():
    from gap_assistant.schemas import LLMNoveltyList
    out = llm._parse_json_model('```json\n{"items": []}\n```', LLMNoveltyList)
    assert out.items == []
    out = llm._parse_json_model('Sure! {"items": []} Hope that helps', LLMNoveltyList)
    assert out.items == []
    with pytest.raises(ValueError):
        llm._parse_json_model("nonsense", LLMNoveltyList)


class _Err(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def _client_for_retry(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    c = llm.GeminiClient.__new__(llm.GeminiClient)  # skip real SDK init
    c.max_retries = 2
    c.min_interval_s = 0.0
    return c


def test_retry_logic(monkeypatch):
    c = _client_for_retry(monkeypatch)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Err(503, "UNAVAILABLE model overloaded")
        return "ok"

    assert c._retry(flaky) == "ok" and calls["n"] == 3

    with pytest.raises(llm.QuotaExceededError):
        c._retry(lambda: (_ for _ in ()).throw(_Err(429, "RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel")))
    with pytest.raises(llm.AuthError):
        c._retry(lambda: (_ for _ in ()).throw(_Err(400, "API key not valid. Please pass a valid API key.")))
    with pytest.raises(llm.LLMError, match="not found"):
        c._retry(lambda: (_ for _ in ()).throw(_Err(404, "models/foo is not found")))
    with pytest.raises(llm.LLMError, match="gave up"):
        c._retry(lambda: (_ for _ in ()).throw(_Err(429, "RESOURCE_EXHAUSTED per minute. Please retry in 2s")))
    assert llm._suggested_delay("Please retry in 12.5s.") == 12.5


def test_generate_json_falls_back_to_prompt_schema(monkeypatch):
    from gap_assistant.schemas import LLMNoveltyList
    c = _client_for_retry(monkeypatch)
    seen = []

    def fake_generate(model, prompt, system, schema=None, json_mode=False):
        seen.append(schema)
        if schema is not None:
            raise llm.LLMError("schema rejected", 400)
        assert "JSON Schema" in prompt
        return '{"items": []}'

    c._generate = fake_generate
    c._model = "m"
    assert c.generate_json("hi", LLMNoveltyList).items == []
    assert seen[0] is LLMNoveltyList and seen[1] is None


# ----------------------------------------------------------------------------- extraction
def test_extract_paper_verifies_quotes_and_caches(pdf_dir, fake_llm, tmp_path):
    parsed = parse_pdf((pdf_dir / "paper_a.pdf").read_bytes())
    kwargs = dict(source="paper_a.pdf", paper_id="P1", cache_dir=tmp_path / "c")
    p = extract_paper(fake_llm, parsed.text, **kwargs)
    assert p.title.startswith("UNetVessel") and p.limitations[0].verified and p.future_work[0].verified
    assert fake_llm.calls["extract"] == 1
    p2 = extract_paper(fake_llm, parsed.text, **kwargs)  # served from cache
    assert fake_llm.calls["extract"] == 1 and p2 == p
    # a text that does not contain the quote -> flagged unverified
    p3 = extract_paper(fake_llm, "Totally different text. " * 50 + " source=paper_a.pdf", **kwargs)
    assert not p3.limitations[0].verified


# ----------------------------------------------------------------------------- comparison
def _papers():
    from tests.conftest import EXTRACTIONS
    papers = []
    for i, (name, ex) in enumerate(EXTRACTIONS.items(), 1):
        papers.append(Paper(
            paper_id=f"P{i}", source=name, title=ex.title, year=ex.year, domain=ex.domain,
            research_problem=ex.research_problem, methods=ex.methods, datasets=ex.datasets,
            metrics=ex.metrics, assumptions=ex.assumptions,
            limitations=[Claim(statement=c.statement, quote=c.quote, verified=True) for c in ex.limitations],
            future_work=[Claim(statement=c.statement, quote=c.quote, verified=True) for c in ex.future_work]))
    return papers


def test_matrix_and_hints():
    papers = _papers()
    m = compare.cooccurrence_matrix(papers)
    assert m.loc["U-Net", "DRIVE"] == 2 and m.loc["SegFormer", "CHASE_DB1"] == 2
    hints = compare.coverage_hints(papers)
    pairs = {(h["method"], h["dataset"]) for h in hints}
    assert ("Swin-UNet", "STARE") in pairs and ("U-Net", "DRIVE") not in pairs


def test_recurring_statements_and_map():
    papers = _papers()
    emb = Embedder("tfidf")
    lims = compare.recurring_statements(papers, "limitations", emb)
    assert lims, "adult-only limitation is shared by several papers"
    assert any({"P1", "P2"} <= set(t["paper_ids"]) or {"P2", "P4"} <= set(t["paper_ids"]) for t in lims)
    df = compare.paper_map(papers, emb)
    assert len(df) == 4 and {"x", "y", "cluster"} <= set(df.columns)
    assert compare.recurring_statements(papers[:1], "limitations", emb) == []
    assert compare.paper_map(papers[:1], emb).empty
    two = compare.paper_map(papers[:2], emb)
    assert len(two) == 2 and not two[["x", "y"]].isna().any().any()


def test_build_comparison_serialisable():
    comp = compare.build_comparison(_papers(), Embedder("tfidf"))
    json.dumps(comp.to_dict(), default=str)


# ----------------------------------------------------------------------------- gaps
def test_validate_gaps_grounds_and_ranks(fake_llm):
    papers = _papers()
    texts = {f"P{i}": " ".join(PAPERS[n][1]) for i, n in enumerate(PAPERS, 1)}
    comp = compare.build_comparison(papers, Embedder("tfidf"))
    gaps, warnings = gaps_mod.generate_gaps(fake_llm, papers, texts, comp, topic="retinal vessels", max_gaps=8)
    assert [g.title for g in gaps] == ["Pediatric retinal images are unstudied", "Missing significance testing"]
    assert [g.rank for g in gaps] == [1, 2]
    g1, g2 = gaps
    assert [e.paper_id for e in g1.evidence] == ["P1", "P2"] and all(e.verified for e in g1.evidence)  # "p2" normalised, P9 dropped
    assert [e.verified for e in g2.evidence] == [True, False]  # invented quote flagged
    assert g1.support == 2 and 1.0 <= g1.score <= 5.0
    assert any("P9" in w for w in warnings) and any("Hallucinated gap" in w for w in warnings)
    assert "retinal vessels" in fake_llm.last_gap_prompt and "<digest>" in fake_llm.last_gap_prompt


def test_compute_score_bounds():
    assert gaps_mod.compute_score(5, 5, 4) == 5.0
    assert gaps_mod.compute_score(1, 1, 0) == 1.0
    assert gaps_mod._clamp(9) == 5 and gaps_mod._clamp(-3) == 1 and gaps_mod._clamp("x") == 3


# ----------------------------------------------------------------------------- search + novelty
class _Resp:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code, self._payload, self._content = status, payload or {}, content

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=0):
        yield self._content


class _Session:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def get(self, url, **kw):
        self.requests.append((url, kw))
        return self.responses.pop(0)


S2_PAYLOAD = {"data": [
    {"paperId": "a1", "title": "Pediatric Vessel Segmentation", "abstract": "We segment vessels in children.",
     "year": 2023, "authors": [{"name": "X"}], "venue": "V", "citationCount": 3, "url": "http://s2/a1",
     "openAccessPdf": {"url": "http://pdf/a1.pdf"}},
    {"paperId": "a2", "title": "", "abstract": None},  # no title: skipped
    {"paperId": "a3", "title": "No Abstract Paper", "abstract": None, "year": None, "authors": None,
     "openAccessPdf": None},
]}


def test_search_papers_parses_and_retries(monkeypatch):
    monkeypatch.setattr(search.time, "sleep", lambda s: None)
    s = _Session([_Resp(429), _Resp(200, S2_PAYLOAD)])
    hits = search.search_papers("retinal: vessel (segmentation)!", limit=5, year_from=2020, session=s)
    assert [h.title for h in hits] == ["Pediatric Vessel Segmentation", "No Abstract Paper"]
    assert hits[0].pdf_url == "http://pdf/a1.pdf" and hits[1].pdf_url == "" and hits[1].authors == []
    assert s.requests[-1][1]["params"]["query"] == "retinal vessel segmentation"
    assert s.requests[-1][1]["params"]["year"] == "2020-"
    with pytest.raises(search.SearchError):
        search.search_papers("x", session=_Session([_Resp(500)]))
    with pytest.raises(search.SearchError):
        search.search_papers("!!!")


def test_download_pdf_validates():
    assert search.download_pdf("http://x", session=_Session([_Resp(200, content=b"%PDF-1.5 data")])).startswith(b"%PDF")
    with pytest.raises(search.SearchError, match="not return a PDF"):
        search.download_pdf("http://x", session=_Session([_Resp(200, content=b"<html>")]))
    with pytest.raises(search.SearchError, match="HTTP 403"):
        search.download_pdf("http://x", session=_Session([_Resp(403)]))


def test_check_novelty(fake_llm, monkeypatch):
    monkeypatch.setattr(gaps_mod.time, "sleep", lambda s: None)
    papers = _papers()
    texts = {f"P{i}": " ".join(PAPERS[n][1]) for i, n in enumerate(PAPERS, 1)}
    comp = compare.build_comparison(papers, Embedder("tfidf"))
    gaps, _ = gaps_mod.generate_gaps(fake_llm, papers, texts, comp)
    s = _Session([_Resp(200, S2_PAYLOAD), _Resp(200, {"data": []})])
    warnings = gaps_mod.check_novelty(fake_llm, gaps, session=s)
    assert not warnings
    assert gaps[0].novelty_check.status == "partly_addressed" and gaps[0].novelty_check.related
    assert gaps[1].novelty_check.status == "likely_open" and "weak evidence" in gaps[1].novelty_check.reason
    assert fake_llm.calls["novelty"] == 1
    # search failure -> gap stays "unchecked", no exception
    s = _Session([_Resp(500), _Resp(500)])
    warnings = gaps_mod.check_novelty(fake_llm, gaps, session=s)
    assert len(warnings) == 2 and all(g.novelty_check.status == "unchecked" for g in gaps)


# ----------------------------------------------------------------------------- pipeline
def test_pipeline_end_to_end(pdf_dir, fake_llm, settings):
    inputs, errors = load_pdf_paths(sorted(pdf_dir.glob("*.pdf")))
    assert not errors and len(inputs) == 4
    inputs.append(PaperInput(name="dup.pdf", text=inputs[0].text))  # duplicate text is skipped
    steps = []
    res = run_analysis(fake_llm, inputs, "retinal vessel segmentation", settings, progress=lambda f, m: steps.append(f))
    assert [p.paper_id for p in res.papers] == ["P1", "P2", "P3", "P4"]
    assert all(l.verified for p in res.papers for l in p.limitations)
    assert len(res.gaps) == 2 and res.gaps[0].rank == 1
    assert any("dup.pdf" in w for w in res.warnings)
    assert steps[-1] == 1.0 and steps == sorted(steps)
    md = report.to_markdown(res)
    assert "Pediatric retinal images are unstudied" in md and "QUOTE NOT FOUND" in md
    json.dumps(res.to_dict(), default=str)
    # second run: all extractions come from the cache
    before = fake_llm.calls["extract"]
    run_analysis(fake_llm, inputs, "x", settings)
    assert fake_llm.calls["extract"] == before


def test_pipeline_needs_two_papers(pdf_dir, fake_llm, settings):
    inputs, _ = load_pdf_paths([pdf_dir / "paper_a.pdf"])
    res = run_analysis(fake_llm, inputs, "", settings)
    assert len(res.papers) == 1 and not res.gaps and any("At least 2" in w for w in res.warnings)


def test_pipeline_survives_quota_error(pdf_dir, fake_llm, settings):
    inputs, _ = load_pdf_paths(sorted(pdf_dir.glob("*.pdf")))
    real = fake_llm.generate_json
    n = {"i": 0}

    def limited(prompt, schema, **kw):
        from gap_assistant.schemas import LLMExtraction
        if schema is LLMExtraction:
            n["i"] += 1
            if n["i"] > 1:
                raise llm.QuotaExceededError("daily quota")
        return real(prompt, schema, **kw)

    fake_llm.generate_json = limited
    res = run_analysis(fake_llm, inputs, "", settings)
    assert len(res.papers) == 1 and any("daily quota" in w for w in res.warnings)


def test_report_chat(pdf_dir, fake_llm, settings):
    inputs, _ = load_pdf_paths(sorted(pdf_dir.glob("*.pdf")))
    res = run_analysis(fake_llm, inputs, "", settings)
    assert report.answer_question(fake_llm, "Which papers use DRIVE?", res, history=[]) == "Answer from the fake model."


def test_gemini_client_throttles_requests(monkeypatch):
    from gap_assistant import llm as llm_mod

    now = [100.0]
    sleeps = []
    monkeypatch.setattr(llm_mod.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)))
    client = llm_mod.GeminiClient("fake-key", model="m", min_interval_s=10)
    assert client._retry(lambda: 1) == 1  # first call: last_call=0 long ago, no wait
    now[0] += 3
    assert client._retry(lambda: 2) == 2  # only 3 s later -> must wait ~7 s
    assert len(sleeps) == 1 and abs(sleeps[0] - 7) < 1e-6
    off = llm_mod.GeminiClient("fake-key", model="m", min_interval_s=0)
    off._retry(lambda: 1); off._retry(lambda: 1)
    assert len(sleeps) == 1


def _embed_client(n_returned_fn):
    from types import SimpleNamespace
    from google.genai import types

    c = llm.GeminiClient.__new__(llm.GeminiClient)
    c._types, c._embedding_model, c.max_retries, c.min_interval_s = types, "emb", 0, 0.0
    seen = {}

    def embed_content(model, contents, config):
        seen["contents"] = contents
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0, 0.5])
                                           for _ in range(n_returned_fn(len(contents)))])

    c.client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    return c, seen


def test_embed_sends_one_content_per_text():
    c, seen = _embed_client(lambda n: n)
    out = c.embed(["a", "b", "c"])
    assert out.shape == (3, 3)
    assert len(seen["contents"]) == 3 and all(hasattr(x, "parts") for x in seen["contents"])


def test_embed_vector_count_mismatch_is_error_and_embedder_falls_back():
    from gap_assistant.embeddings import Embedder

    c, _ = _embed_client(lambda n: 1)  # model merges everything into one vector
    with pytest.raises(llm.LLMError):
        c.embed(["a", "b"])
    emb = Embedder("gemini", c)
    vecs = emb.embed(["alpha beta", "gamma delta"])
    assert vecs.shape[0] == 2 and emb.last_backend == "tfidf" and emb.warnings


def test_semantic_scholar_one_request_per_second(monkeypatch):
    from types import SimpleNamespace

    now, sleeps, sent = [1000.0], [], []
    monkeypatch.setattr(search.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(search.time, "sleep", lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)))
    monkeypatch.setattr(search, "_s2_last", 0.0)
    monkeypatch.setenv("S2_API_KEY", "k")

    def get(url, params, headers, timeout):
        sent.append((now[0], headers))
        return SimpleNamespace(status_code=200, json=lambda: {"data": []})

    session = SimpleNamespace(get=get)
    for _ in range(3):
        search.search_papers("graph neural networks", session=session)
    gaps_between = [b[0] - a[0] for a, b in zip(sent, sent[1:])]
    assert all(g >= 1.0 for g in gaps_between)  # never more than 1 request per second
    assert all(h.get("x-api-key") == "k" for _, h in sent)  # key sent in the x-api-key header
