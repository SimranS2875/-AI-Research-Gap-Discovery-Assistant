"""Streamlit app: AI Research Gap Discovery Assistant.

Run with:  streamlit run app.py
"""
import json
import os
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

from gap_assistant import llm as llm_mod
from gap_assistant.config import Settings, load_settings
from gap_assistant.llm import AuthError, LLMError
from gap_assistant.pdf_parser import PdfError
from gap_assistant.pipeline import AnalysisResult, PaperInput, load_pdf_input, run_analysis
from gap_assistant.report import NOVELTY_LABELS, TYPE_LABELS, answer_question, to_markdown
from gap_assistant.search import SearchError, download_pdf, search_papers

st.set_page_config(page_title="AI Research Gap Discovery Assistant", page_icon="🔎", layout="wide")

NOVELTY_ICONS = {"likely_open": "🟢", "partly_addressed": "🟡", "likely_addressed": "🔴", "unchecked": "⚪"}

for key, default in (("extra_inputs", []), ("hits", []), ("result", None), ("chat", [])):
    if key not in st.session_state:
        st.session_state[key] = default


def esc(text: str) -> str:
    """Make text safe to show with st.markdown (no accidental LaTeX from '$')."""
    return (text or "").replace("$", "\\$").replace("\n", " ")


# ----------------------------------------------------------------------------- sidebar
env = load_settings()
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API key", value=env.api_key, type="password",
                            help="Get a free key at aistudio.google.com. It is only used in this session. "
                                 "You can also put it in the .env file.")
    max_gaps = st.slider("Max gaps to report", 3, 15, min(max(env.max_gaps, 3), 15))
    novelty_check = st.checkbox(
        "Check gaps against Semantic Scholar", value=bool(os.getenv("S2_API_KEY")),
        help="Searches the wider literature to see if each gap is already addressed. "
             "On by default only when S2_API_KEY is set in .env (without a key it is usually rate-limited).")
    with st.expander("Advanced"):
        model = st.text_input("Model (blank = auto-detect)", value=env.model,
                              help="Model names change often. Blank picks the newest stable Flash model your key can use.")
        gap_model = st.text_input("Gap-reasoning model (optional)", value=env.gap_model,
                                  help="A stronger model for the final gap analysis. Blank = same as above.")
        backend = st.selectbox("Similarity method", ["tfidf", "gemini"],
                               index=0 if env.embedding_backend != "gemini" else 1,
                               format_func=lambda b: {"tfidf": "Offline TF-IDF (no extra API calls)",
                                                      "gemini": "Gemini embeddings (uses quota)"}[b])
        max_chars = st.number_input("Max characters read per paper", 20_000, 400_000, env.max_chars_per_paper,
                                    step=10_000, help="Longer papers keep their beginning and end.")

    def get_client():
        """Create (or reuse) the Gemini client for the current settings."""
        sig = (api_key, model, env.embedding_model)
        if st.session_state.get("_client_sig") != sig:
            st.session_state["_client"] = llm_mod.GeminiClient(
                api_key, model=model, embedding_model=env.embedding_model)
            st.session_state["_client_sig"] = sig
        return st.session_state["_client"]

    if st.button("Test API key"):
        try:
            client = get_client()
            st.success(f"Key works. Will use: {client.default_model()}")
        except LLMError as e:
            st.error(str(e))

settings = Settings(
    api_key=api_key, model=model.strip(), gap_model=gap_model.strip(), embedding_backend=backend,
    embedding_model=env.embedding_model, max_chars_per_paper=int(max_chars), max_gaps=int(max_gaps),
    novelty_check=novelty_check, cache_dir=env.cache_dir)

# ----------------------------------------------------------------------------- header + input
st.title("🔎 AI Research Gap Discovery Assistant")
st.caption("Upload 3–15 papers on one topic. You get ranked research gaps, each backed by quotes you can check.")

topic = st.text_input("Research topic (optional, helps focus the analysis)",
                      placeholder="e.g. retinal vessel segmentation with deep learning")
uploads = st.file_uploader("Upload papers (PDF)", type=["pdf"], accept_multiple_files=True)

with st.expander("Or find papers by topic (Semantic Scholar)"):
    c1, c2, c3 = st.columns([3, 1, 1])
    query = c1.text_input("Search query", value=topic, key="s2_query")
    year_from = c2.number_input("From year (0 = any)", 0, 2100, 0)
    limit = c3.slider("Results", 5, 25, 10)
    if st.button("Search"):
        for stale in [k for k in st.session_state if str(k).startswith("hit_")]:
            del st.session_state[stale]  # forget ticks from a previous search
        try:
            st.session_state["hits"] = search_papers(query, limit=limit, year_from=int(year_from) or None)
            if not st.session_state["hits"]:
                st.info("No results. Try different keywords.")
        except SearchError as e:
            st.error(str(e))
    hits = st.session_state["hits"]
    if hits:
        st.caption("Tick papers to add. Open-access PDFs are downloaded when possible; otherwise only the "
                   "abstract is analysed (less reliable).")
        for i, h in enumerate(hits):
            label = f"{h.title} ({h.year or 'n.d.'}) — {'📄 open-access PDF' if h.pdf_url else '📝 abstract only'}"
            st.checkbox(label, key=f"hit_{i}")
            if h.abstract:
                st.caption(esc(h.abstract[:300] + ("…" if len(h.abstract) > 300 else "")))
        if st.button("Add selected papers"):
            existing = {p.name for p in st.session_state["extra_inputs"]}
            with st.spinner("Fetching papers…"):
                for i, h in enumerate(hits):
                    if not st.session_state.get(f"hit_{i}") or h.title in existing:
                        continue
                    added = None
                    if h.pdf_url:
                        try:
                            added = load_pdf_input(h.title, download_pdf(h.pdf_url))
                            added.url = h.url
                        except (SearchError, PdfError) as e:
                            st.warning(f"'{h.title}': PDF unavailable ({str(e)[:120]}). Falling back to abstract.")
                    if added is None and len(h.abstract) >= 100:
                        added = PaperInput(name=h.title, text=f"{h.title}\n\n{h.abstract}",
                                           source_type="abstract", url=h.url, title_hint=h.title)
                    if added is None:
                        st.warning(f"'{h.title}': no PDF and no abstract available; skipped.")
                        continue
                    st.session_state["extra_inputs"].append(added)
                    existing.add(h.title)

extra: List[PaperInput] = st.session_state["extra_inputs"]
if extra:
    cols = st.columns([6, 1])
    cols[0].info(f"{len(extra)} paper(s) added from search: " + "; ".join(
        f"{p.name[:60]} ({'abstract' if p.source_type == 'abstract' else 'PDF'})" for p in extra))
    if cols[1].button("Clear"):
        st.session_state["extra_inputs"] = []
        st.rerun()

total = len(uploads or []) + len(extra)
st.caption(f"{total} paper(s) ready.")

if st.button("Analyze papers", type="primary", disabled=total < 2):
    if not api_key:
        st.error("Enter your Gemini API key in the sidebar (or set GEMINI_API_KEY in the .env file).")
    else:
        inputs: List[PaperInput] = list(extra)
        for f in uploads or []:
            try:
                inputs.append(load_pdf_input(f.name, f.getvalue()))
            except PdfError as e:
                st.warning(str(e))
        bar = st.progress(0.0, text="Starting…")
        try:
            result = run_analysis(get_client(), inputs, topic, settings,
                                  progress=lambda frac, msg: bar.progress(min(max(frac, 0.0), 1.0), text=msg))
            st.session_state["result"] = result
            st.session_state["chat"] = []
        except AuthError as e:
            st.error(str(e))
        except LLMError as e:
            st.error(f"Analysis failed: {e}")
        finally:
            bar.empty()

# ----------------------------------------------------------------------------- results
result: AnalysisResult = st.session_state["result"]
if result is None:
    st.stop()

st.divider()
st.caption(f"{len(result.papers)} papers analysed · {len(result.gaps)} gaps found · model: {result.model}")
if result.warnings:
    with st.expander(f"Notes and warnings ({len(result.warnings)})", expanded=not result.gaps):
        for w in result.warnings:
            st.warning(w)

tab_gaps, tab_papers, tab_compare, tab_ask, tab_export = st.tabs(
    ["Research gaps", "Papers", "Comparison", "Ask", "Export"])

with tab_gaps:
    if not result.gaps:
        st.info("No gaps were generated. See the notes above; with fewer than 2 readable papers the comparison is skipped.")
    st.caption("Gaps are relative to the papers you provided; verify before committing to one. "
               "Score = 40% novelty + 30% feasibility + 30% papers backing it.")
    for g in result.gaps:
        with st.container(border=True):
            st.subheader(f"{g.rank}. {esc(g.title)}")
            st.caption(f"**Score {g.score}/5** · {TYPE_LABELS.get(g.gap_type, g.gap_type)} · "
                       f"novelty {g.novelty}/5 · feasibility {g.feasibility}/5 · "
                       f"{g.support} paper{'s' if g.support != 1 else ''} backing it")
            st.markdown(esc(g.description))
            if g.suggested_direction:
                st.markdown(f"**Suggested direction:** {esc(g.suggested_direction)}")
            with st.expander("Details and evidence"):
                if g.why_it_matters:
                    st.markdown(f"**Why it matters:** {esc(g.why_it_matters)}")
                for e in g.evidence:
                    st.markdown(f"**{e.paper_id}** · {esc(e.paper_title)}")
                    if e.quote:
                        st.markdown(f"> {esc(e.quote)}  \n{'✅ quote verified' if e.verified else '⚠️ quote not found in paper text'}")
            nc = g.novelty_check
            if nc.status != "unchecked" or nc.reason:
                with st.expander(f"Literature check: {NOVELTY_ICONS.get(nc.status, '⚪')} "
                                 f"{NOVELTY_LABELS.get(nc.status, nc.status)}"):
                    if nc.reason:
                        st.markdown(esc(nc.reason))
                    for r in nc.related:
                        title = f"[{esc(r.title)}]({r.url})" if r.url else esc(r.title)
                        st.markdown(f"- {title} ({r.year or 'n.d.'})")

with tab_papers:
    rows = [{"ID": p.paper_id, "Title": p.title, "Year": p.year, "Domain": p.domain,
             "Methods": ", ".join(p.methods), "Datasets": ", ".join(p.datasets),
             "Metrics": ", ".join(p.metrics),
             "Source": "full text" if p.source_type == "full_text" else "abstract only"} for p in result.papers]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    for p in result.papers:
        with st.expander(f"{p.paper_id} · {p.title}"):
            st.markdown(f"**Problem:** {esc(p.research_problem)}")
            if p.assumptions:
                st.markdown("**Assumptions / scope:** " + esc("; ".join(p.assumptions)))
            for heading, claims in (("Key results", p.key_results), ("Limitations", p.limitations),
                                    ("Future work", p.future_work)):
                if claims:
                    st.markdown(f"**{heading}**")
                    for cl in claims:
                        mark = "✅" if cl.verified else ("⚠️" if cl.quote else "")
                        st.markdown(f"- {esc(cl.statement)} {mark}")
                        if cl.quote:
                            st.caption(f"“{esc(cl.quote)}”")
            if p.source_type == "abstract":
                st.info("Only the abstract was available, so limitations and assumptions may be missing.")

with tab_compare:
    comp = result.comparison
    st.subheader("Methods × datasets")
    st.caption("Number of papers using each pair. Empty cells between popular methods and datasets are candidate gaps.")
    if comp.matrix.empty or comp.matrix.values.sum() == 0:
        st.info("Not enough method/dataset information to build the matrix.")
    else:
        fig = px.imshow(comp.matrix, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                        labels={"x": "Dataset", "y": "Method", "color": "Papers"})
        st.plotly_chart(fig)
    if comp.coverage_hints:
        st.subheader("Popular methods and datasets never used together")
        st.dataframe(pd.DataFrame([{"Method": h["method"], "Dataset": h["dataset"],
                                    "Method used in": ", ".join(h["method_papers"]),
                                    "Dataset used in": ", ".join(h["dataset_papers"])} for h in comp.coverage_hints]),
                     hide_index=True)
    for title, themes in (("Limitations raised by several papers", comp.recurring_limitations),
                          ("Future work suggested by several papers", comp.recurring_future_work)):
        st.subheader(title)
        if not themes:
            st.caption("None found (statements were too different, or too few papers).")
        for t in themes:
            with st.expander(f"{esc(t['theme'])} — {', '.join(t['paper_ids'])}"):
                for it in t["items"]:
                    st.markdown(f"- **{it['paper_id']}**: {esc(it['statement'])}")
    st.subheader("Paper map")
    if comp.paper_map.empty:
        st.caption("Needs at least 2 papers.")
    else:
        fig = px.scatter(comp.paper_map, x="x", y="y", color="cluster", text="paper_id", hover_name="title")
        fig.update_traces(textposition="top center", marker=dict(size=14))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(fig)
        st.caption("Papers close together address similar problems and methods.")

with tab_ask:
    st.caption("Ask questions about the analysed papers, e.g. “Which papers use the DRIVE dataset?” or "
               "“Which gap would be easiest to start with?”")
    for msg in st.session_state["chat"]:
        with st.chat_message(msg["role"]):
            st.markdown(esc(msg["content"]))
    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input("Your question")
        asked = st.form_submit_button("Ask")
    if asked and question.strip():
        try:
            answer = answer_question(get_client(), question.strip(), result, history=st.session_state["chat"])
            st.session_state["chat"] += [{"role": "user", "content": question.strip()},
                                         {"role": "assistant", "content": answer}]
            st.rerun()
        except LLMError as e:
            st.error(str(e))

with tab_export:
    st.download_button("Download report (Markdown)", data=to_markdown(result),
                       file_name="research_gap_report.md", mime="text/markdown")
    st.download_button("Download all data (JSON)", data=json.dumps(result.to_dict(), indent=2, default=str),
                       file_name="research_gap_analysis.json", mime="application/json")
