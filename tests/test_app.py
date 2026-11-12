"""Drives the real Streamlit script headlessly (no browser, fake LLM, no network)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from gap_assistant import config, llm as llm_mod
from gap_assistant.pipeline import load_pdf_paths
from gap_assistant.search import SearchHit
from tests.conftest import FakeLLM

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _button(at, label):
    return next(b for b in at.button if b.label == label)


@pytest.fixture
def app(tmp_path, pdf_dir, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)  # cache goes to a temp dir
    # Don't let the developer's real .env settings leak into the tests.
    for var in ("GEMINI_MODEL", "GEMINI_GAP_MODEL", "GEMINI_EMBEDDING_MODEL", "EMBEDDING_BACKEND",
                "MAX_GAPS", "MAX_CHARS_PER_PAPER", "S2_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("S2_API_KEY", "test")  # novelty check defaults to on only when a key is set
    fake = FakeLLM()
    monkeypatch.setattr(llm_mod, "GeminiClient", lambda *a, **k: fake)
    import gap_assistant.gaps as gaps_mod
    monkeypatch.setattr(gaps_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(gaps_mod, "search_papers", lambda q, limit=5, session=None: [
        SearchHit("x", "Pediatric Vessel Segmentation", "We segment vessels in children.", 2023, ["A"], "V", 3,
                  "http://s2/x", "")])
    inputs, _ = load_pdf_paths(sorted(pdf_dir.glob("*.pdf")))
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["extra_inputs"] = inputs
    at.fake = fake
    return at


def test_fresh_app_has_no_errors_and_button_state():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert _button(at, "Analyze papers").disabled  # needs >= 2 papers


def test_requires_api_key(app):
    app.run()
    app.sidebar.text_input[0].set_value("")  # no key
    app.run()
    _button(app, "Analyze papers").click().run()
    assert any("API key" in e.value for e in app.error)
    assert app.session_state["result"] is None


def test_full_flow_and_ask(app):
    app.run()
    app.sidebar.text_input[0].set_value("fake-key")
    app.run()
    _button(app, "Analyze papers").click().run()
    assert not app.exception, app.exception
    result = app.session_state["result"]
    assert result is not None and len(result.papers) == 4 and len(result.gaps) == 2
    assert result.gaps[0].novelty_check.status == "partly_addressed"
    labels = [t.label for t in app.tabs]
    assert labels == ["Research gaps", "Papers", "Comparison", "Ask", "Export"]
    shown = " ".join(m.value for m in app.markdown)
    assert "1. Pediatric retinal images are unstudied" in [s.value for s in app.subheader]  # gap title rendered
    assert "quote verified" in shown and "quote not found" in shown  # verification marks rendered
    captions = " ".join(c.value for c in app.caption)
    assert "4 papers analysed · 2 gaps found" in captions
    assert "Score " in captions and "novelty " in captions and "feasibility " in captions
    assert not app.metric  # compact caption instead of big metric tiles
    # literature check sits in a collapsed expander, not inline text
    assert not any(m.value.startswith("**Literature check:**") for m in app.markdown)
    assert any(e.label.startswith("Literature check:") for e in app.expander)

    # Ask tab
    q = next(t for t in app.text_input if t.label == "Your question")
    q.set_value("Which papers use DRIVE?")
    _button(app, "Ask").click().run()
    assert not app.exception
    assert app.session_state["chat"][-1]["content"] == "Answer from the fake model."


def test_novelty_check_off_hides_literature_section(app, monkeypatch):
    monkeypatch.delenv("S2_API_KEY")
    app.run()
    assert not app.sidebar.checkbox[0].value  # off by default without an S2 key
    app.sidebar.text_input[0].set_value("fake-key")
    app.run()
    _button(app, "Analyze papers").click().run()
    assert not app.exception
    assert app.session_state["result"].gaps
    assert not any(e.label.startswith("Literature check") for e in app.expander)


def test_quota_error_is_shown_not_crashed(app, monkeypatch):
    def boom(*a, **k):
        raise llm_mod.AuthError("Gemini rejected the request (API key invalid)")
    monkeypatch.setattr(app.fake, "default_model", boom)
    app.run()
    app.sidebar.text_input[0].set_value("bad-key")
    app.run()
    _button(app, "Analyze papers").click().run()
    assert not app.exception
    assert any("rejected" in e.value for e in app.error)
