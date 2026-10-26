"""DocChat - chat with your own documents: PDFs, Word, slides, sheets, HTML, images and websites. Answers cite their sources."""
from __future__ import annotations

import hashlib
import html
import json
import os
import platform
import re
import time

import pandas as pd
import streamlit as st

import index as ix
import llm
import rag
import readers

# ── Edit these before your presentation ──
PROJECT = {"title": "DocChat: Document Question Answering with Cited Sources", "student": "Your Name",
           "reg_no": "Your Reg. No.", "supervisor": "Supervisor Name",
           "department": "Department of Computer Science, University of Agriculture, Faisalabad"}

st.set_page_config(page_title="DocChat", page_icon="💬", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&display=swap');
h1,h2,h3,h4,.hero h1{font-family:'Bricolage Grotesque',sans-serif;letter-spacing:-.01em}
.block-container{padding-top:1.4rem;max-width:1100px}
.hero{position:relative;overflow:hidden;background:#12202B;color:#EAF2F5;border-radius:6px;padding:26px 32px 20px;margin-bottom:16px}
.hero h1{color:#fff;font-size:2.1rem;font-weight:800;margin:0 0 6px;padding:0}
.hero p{margin:0;max-width:700px;color:#B9CBD3}
.hero::after{content:"";position:absolute;top:0;bottom:0;left:-6px;width:3px;background:#38D0DC;box-shadow:0 0 22px 6px rgba(56,208,220,.45);animation:scan 2.2s ease-out 1 forwards}
@keyframes scan{to{left:101%}}
@media (prefers-reduced-motion:reduce){.hero::after{display:none}}
.flow{display:flex;flex-wrap:wrap;margin-top:16px}
.flow div{flex:1 1 150px;border-top:2px solid #38D0DC;padding:9px 14px 0 0;font-size:.86rem;color:#9FB6C0}
.flow b{display:block;color:#fff;font-size:.98rem}
.src{background:rgba(127,127,127,.08);border:1px solid rgba(127,127,127,.3);border-left:4px solid #0E7C86;border-radius:4px;padding:10px 14px;margin-bottom:8px}
.src .tag{font-weight:600;color:#0E7C86;font-size:.84rem}
.src p{margin:5px 0 6px;font-size:.9rem;opacity:.92;white-space:pre-wrap}
.bar{height:3px;background:rgba(127,127,127,.25);border-radius:9px;overflow:hidden}.bar i{display:block;height:100%;background:#0E7C86}
.chip{display:inline-block;padding:2px 11px;margin:0 6px 6px 0;border:1px solid rgba(127,127,127,.4);border-radius:99px;font-size:.82rem}
mark{background:rgba(14,124,134,.28);color:inherit;padding:0 2px;border-radius:2px}
div[data-testid="stMetric"]{background:rgba(127,127,127,.08);border:1px solid rgba(127,127,127,.3);border-radius:4px;padding:10px 14px}
.stButton>button[kind="primary"]{font-weight:600;border-radius:4px;background:#0E7C86;border-color:#0E7C86;color:#fff}
</style>""", unsafe_allow_html=True)

PAGES = ["Chat", "Knowledge base", "Insights", "Evaluation", "About"]
if "index" not in st.session_state:
    st.session_state.index = ix.Index()
for k, v in {"messages": [], "meta": {}, "errors": {}, "pending": None}.items():
    st.session_state.setdefault(k, v)


def esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def secret(name: str) -> str:
    """Read a key from Streamlit secrets (Cloud or .streamlit/secrets.toml) or an environment variable."""
    try:
        v = st.secrets.get(name, "")
    except Exception:
        v = ""
    return str(v or os.environ.get(name, ""))


def idx() -> ix.Index:
    return st.session_state.index


# ───────────────────────── knowledge base ─────────────────────────
@st.cache_data(show_spinner=False, max_entries=20)
def read_cached(name: str, raw: bytes, lang: str):
    return readers.read_file(name, raw, lang)


def ingest(doc: readers.Document, src: str):
    if doc.name in idx().docs():
        idx().remove(doc.name)
    n = idx().add(doc.name, doc.pages)
    st.session_state.meta[doc.name] = {"kind": doc.kind, "chars": doc.chars, "chunks": n, "notes": doc.notes,
                                       "src": src, "preview": doc.text[:2500]}


def drop(name: str):
    idx().remove(name)
    st.session_state.meta.pop(name, None)


def sync_uploads(files, lang: str):
    names = {f.name for f in files}
    for d, m in list(st.session_state.meta.items()):
        if m["src"] == "file" and d not in names:
            drop(d)
    for f in files:
        if f.name in st.session_state.meta:
            continue
        try:
            with st.spinner(f"Reading {f.name}…"):
                ingest(read_cached(f.name, f.getvalue(), lang), "file")
            st.session_state.errors.pop(f.name, None)
        except Exception as e:
            st.session_state.errors[f.name] = str(e)
    for gone in [n for n in st.session_state.errors if n not in names and n not in st.session_state.meta]:
        st.session_state.errors.pop(gone, None)


def load_demo():
    ingest(readers.Document(rag.DEMO_NAME, "demo", [("", rag.DEMO_TEXT)]), "demo")


@st.cache_resource(show_spinner="Loading the semantic search model (first time only)…")
def load_embedder(name: str):
    return ix.Embedder(name)


# ───────────────────────── sidebar ─────────────────────────
def sidebar() -> dict:
    ollama = llm.ollama_models()
    with st.sidebar:
        st.markdown("### Knowledge base")
        files = st.file_uploader("Add files", type=readers.SUPPORTED, accept_multiple_files=True,
                                 help="PDF, Word, PowerPoint, Excel, CSV, HTML, text, Markdown and images (read with OCR).")
        ocr_lang = "eng+urd" if st.selectbox("Language of scanned pages and images", ["English", "English + Urdu"]) != "English" else "eng"
        sync_uploads(files or [], ocr_lang)
        for n, msg in st.session_state.errors.items():
            st.error(f"{n}: {msg}")
        with st.expander("Add a website"):
            url = st.text_input("Web address", placeholder="https://example.com/help")
            pages = st.slider("Pages to read from this site", 1, 15, 1, help="More than 1 follows links on the same website.")
            if st.button("Add website", disabled=not url.strip()):
                try:
                    with st.spinner("Reading the website…"):
                        ingest(readers.read_url(url, ocr_lang, pages), "url")
                except Exception as e:
                    st.error(str(e))
        if not idx().chunks:
            st.button("Load the demo document", on_click=load_demo)
        docs = idx().docs()
        for d, n in docs.items():
            c1, c2 = st.columns([4, 1])
            c1.caption(f"**{d}** · {n} passages")
            if st.session_state.meta.get(d, {}).get("src") != "file" and c2.button("✕", key=f"rm_{d}", help="Remove"):
                drop(d)
                st.rerun()
        only = set(st.multiselect("Search in", list(docs), default=list(docs))) if len(docs) > 1 else set(docs)

        st.markdown("### Answers")
        qopts = ["Keyword (fastest)"] + (list(ix.EMBED_MODELS) if ix.embeddings_available() else [])
        quality = st.selectbox("Search quality", qopts, help="Semantic search finds passages by meaning, not only by exact words.")
        top_k = st.slider("Passages used per answer", 3, 12, 6)
        style = st.selectbox("Answer style", list(rag.STYLES))
        reply_lang = st.selectbox("Reply language", rag.REPLY_LANGS)
        custom = st.text_area("Extra instructions (optional)", placeholder="e.g. Answer like a friendly teaching assistant.", height=70)

        with st.expander("AI model", expanded=False):
            default = 1 if secret("GEMINI_API_KEY") else 2 if secret("LLM_API_KEY") else 3 if ollama else 0
            provider = st.selectbox("Answer engine", llm.PROVIDERS, index=default)
            key = model = base = ""
            if provider.startswith("Gemini"):
                key = st.text_input("Free Gemini API key", value=secret("GEMINI_API_KEY"), type="password",
                                    help="Free at aistudio.google.com/apikey")
                model = st.text_input("Model", "auto", help="'auto' picks the newest free Flash model your key can use.")
                if key:
                    st.success("Gemini key loaded")
            elif provider.startswith("OpenAI"):
                preset = st.selectbox("Provider", list(llm.PRESETS))
                b0, m0 = llm.PRESETS[preset]
                base = st.text_input("Address", b0, key=f"base_{preset}")
                model = st.text_input("Model", m0, key=f"model_{preset}")
                key = st.text_input("API key", value=secret("LLM_API_KEY"), type="password")
            elif provider.startswith("Ollama"):
                base = st.text_input("Ollama address", "http://localhost:11434")
                model = st.selectbox("Model", ollama) if ollama else st.text_input("Model", "llama3.2")
            else:
                st.caption("Shows the best-matching sentences with citations. No key and no internet needed.")
            if provider != llm.PROVIDERS[0]:
                st.caption("Your questions and the matching passages are sent to this service.")
        st.caption(f"**{PROJECT['title']}**  \n{PROJECT['student']} ({PROJECT['reg_no']})  \nSupervisor: {PROJECT['supervisor']}  \n{PROJECT['department']}")

    embedder = None
    if not quality.startswith("Keyword"):
        try:
            embedder = load_embedder(ix.EMBED_MODELS[quality])
        except Exception as e:
            st.sidebar.warning(f"Semantic search is unavailable ({type(e).__name__}); using keyword search.")
    return dict(llm=llm.LLM(provider, model, key, base), top_k=top_k, style=style, reply_lang=reply_lang, custom=custom,
                only=only or None, embedder=embedder, quality=quality)


def hero():
    st.markdown("""<div class="hero"><h1>Chat with your documents.</h1>
<p>Add PDFs, Word files, slides, spreadsheets, web pages, images or whole websites, then ask questions in plain language.
Every answer shows exactly which passage it came from.</p>
<div class="flow"><div><b>Add</b>Files, scans, URLs</div><div><b>Index</b>Passages, keywords + meaning</div>
<div><b>Retrieve</b>Best passages for your question</div><div><b>Answer</b>Streamed, cited, checkable</div></div></div>""", unsafe_allow_html=True)


# ───────────────────────── chat ─────────────────────────
def hit_dict(n: int, h: ix.Hit) -> dict:
    return {"n": n, "doc": h.chunk.doc, "label": h.chunk.label, "text": h.chunk.text, "score": h.score, "coverage": h.coverage}


def render_sources(m: dict):
    cited = rag.cited_numbers(m["content"])
    show = [h for h in m["hits"] if h["n"] in cited] or m["hits"]
    with st.expander(f"Sources ({len(show)})"):
        for h in show:
            where = f"{h['doc']}, {h['label']}" if h["label"] else h["doc"]
            st.markdown(f"""<div class="src"><span class="tag">[{h['n']}] {esc(where)}</span>
<p>{ix.highlight(h['text'], m.get('query', ''))}</p><div class="bar"><i style="width:{int(h['score'] * 100)}%"></i></div></div>""",
                        unsafe_allow_html=True)


def render_message(i: int, m: dict):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("hits"):
            render_sources(m)
            c1, c2 = st.columns([1, 4])
            c1.feedback("thumbs", key=f"fb_{i}")
            c2.caption(f"{m['engine']} · found passages in {m['retrieval']:.2f}s · answered in {m['latency']:.1f}s")


def answer_turn(question: str, S: dict, task: str | None = None):
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question, "task": bool(task)})
    with st.chat_message("user"):
        st.markdown(question)
    t0 = time.time()
    if task:
        hits = rag.summary_hits(idx(), S["only"])
    else:
        hits = idx().search(rag.retrieval_query(question, history), S["top_k"], S["embedder"], S["only"])
    t_ret = time.time() - t0
    cfg = S["llm"]
    engine = f"{cfg.provider.split(' (')[0]} {cfg.model}".strip() if cfg.enabled else "Extractive"
    with st.chat_message("assistant"):
        if not hits:
            text = rag.NOT_FOUND
            st.markdown(text)
        elif task in rag.NEEDS_AI and not cfg.enabled:
            text = "This action needs an AI model. Choose one under **AI model** in the sidebar."
            hits = []
            st.markdown(text)
        else:
            fallback = lambda: rag.extractive_answer("" if task else question, hits, 8 if task else 4)
            if cfg.enabled:
                msgs = rag.build_messages(rag.QUICK[task] if task else question, hits, history)
                try:
                    text = st.write_stream(llm.stream_chat(cfg, rag.system_prompt(S["style"], S["custom"], S["reply_lang"]), msgs))
                except llm.LLMError as e:
                    st.error(f"The AI model could not answer: {e}")
                    text, engine = fallback(), "Extractive (fallback)"
                    st.markdown(text)
            else:
                text = fallback()
                st.markdown(text)
    st.session_state.messages.append({"role": "assistant", "content": text, "query": "" if task else question,
                                      "hits": [hit_dict(i, h) for i, h in enumerate(hits, 1)], "engine": engine,
                                      "retrieval": t_ret, "latency": time.time() - t0, "task": bool(task)})
    st.rerun()


def chat_markdown() -> str:
    out = ["# DocChat conversation\n"]
    for m in st.session_state.messages:
        out.append(f"**{'You' if m['role'] == 'user' else 'DocChat'}:** {m['content']}\n")
        for h in m.get("hits", []):
            if h["n"] in rag.cited_numbers(m["content"]):
                out.append(f"> [{h['n']}] {h['doc']} {h['label']}: {h['text'][:200]}…\n")
    return "\n".join(out)


def page_chat(S: dict):
    if not idx().chunks:
        st.info("Add a file or a website in the sidebar to start. No file at hand? Load the demo document.")
        st.button("Load the demo document", key="demo_main", type="primary", on_click=load_demo)
    else:
        cols = st.columns(len(rag.QUICK) + 1)
        for col, name in zip(cols, rag.QUICK):
            if col.button(name, key=f"q_{name}", help=rag.QUICK[name]):
                st.session_state.pending = (name, f"{name} my documents" if name == "Summarize" else name)
        if st.session_state.messages:
            cols[-1].download_button("Save chat", chat_markdown(), "docchat-conversation.md", "text/markdown")
    if rag.DEMO_NAME in idx().docs() and not st.session_state.messages:
        st.caption("Try: " + "  ·  ".join(f"*{q}*" for q in rag.DEMO_QUESTIONS))
    for i, m in enumerate(st.session_state.messages):
        render_message(i, m)
    pending, st.session_state.pending = st.session_state.pending, None
    q = st.chat_input("Ask a question about your documents", disabled=not idx().chunks)
    if pending:
        answer_turn(pending[1], S, task=pending[0])
    elif q:
        answer_turn(q.strip(), S)


# ───────────────────────── other pages ─────────────────────────
def page_kb(S: dict):
    docs = idx().docs()
    if not docs:
        st.info("Your knowledge base is empty. Add files or a website in the sidebar.")
        return
    m = st.columns(3)
    m[0].metric("Documents", len(docs))
    m[1].metric("Passages", sum(docs.values()))
    m[2].metric("Characters read", f"{sum(v['chars'] for v in st.session_state.meta.values()):,}")
    for d, n in docs.items():
        meta = st.session_state.meta.get(d, {})
        with st.expander(f"{d} · {meta.get('kind', '')} · {n} passages"):
            for note in meta.get("notes", []):
                st.caption(note)
            st.text(meta.get("preview", "")[:1500])
    st.markdown("##### See what the AI would read")
    probe = st.text_input("Type a question to inspect retrieval")
    if probe:
        for i, h in enumerate(idx().search(probe, S["top_k"], S["embedder"], S["only"]), 1):
            st.markdown(f"""<div class="src"><span class="tag">[{i}] {esc(h.chunk.doc)} {esc(h.chunk.label)} · match {int(h.score * 100)}% · query words found {int(h.coverage * 100)}%</span>
<p>{ix.highlight(h.chunk.text, probe)}</p></div>""", unsafe_allow_html=True)


def page_insights():
    msgs = st.session_state.messages
    ans = [m for m in msgs if m["role"] == "assistant" and not m.get("task")]
    if not ans:
        st.info("Insights appear after you ask a few questions.")
        return
    votes = [st.session_state.get(f"fb_{i}") for i, m in enumerate(msgs) if m["role"] == "assistant"]
    votes = [v for v in votes if v is not None]
    nf = sum(1 for m in ans if rag.NOT_FOUND in m["content"])
    c = st.columns(5)
    c[0].metric("Questions", len(ans))
    c[1].metric("Avg answer time", f"{sum(m['latency'] for m in ans) / len(ans):.1f}s")
    c[2].metric("Avg retrieval time", f"{sum(m['retrieval'] for m in ans) / len(ans):.2f}s")
    c[3].metric("Not found in docs", f"{nf}/{len(ans)}")
    c[4].metric("Helpful rate", f"{100 * sum(votes) / len(votes):.0f}%" if votes else "no votes yet")
    used = pd.Series([h["doc"] for m in ans for h in m["hits"] if h["n"] in rag.cited_numbers(m["content"])]).value_counts()
    if len(used):
        st.markdown("##### Most cited documents")
        st.bar_chart(used)
    st.markdown("##### Questions asked")
    st.dataframe(pd.DataFrame([{"Question": m["query"] or "(quick action)", "Seconds": round(m["latency"], 1), "Engine": m["engine"]} for m in ans]), hide_index=True)


def page_eval(S: dict):
    st.markdown("#### Measure retrieval quality")
    st.write("Give questions and a phrase that the right passage must contain. DocChat reports the **hit rate** "
             "(was the right passage in the top results?) and **MRR** (how high it ranked).")
    cases = st.text_area("One test per line:  question || expected phrase", height=150,
                         value=rag.DEMO_CASES if rag.DEMO_NAME in idx().docs() else "")
    k = st.slider("Top results to check (k)", 1, 10, S["top_k"])
    if st.button("Run evaluation", disabled=not (cases.strip() and idx().chunks)):
        parsed = rag.parse_cases(cases)
        runs = {"Keyword": rag.retrieval_eval(idx(), parsed, k)}
        if S["embedder"] is not None:
            runs[f"Hybrid ({S['quality'].split(' ')[0]})"] = rag.retrieval_eval(idx(), parsed, k, S["embedder"])
        st.session_state.eval = runs
    for name, df in st.session_state.get("eval", {}).items():
        a, b = st.columns(2)
        a.metric(f"{name}: hit rate", f"{100 * df.Hit.mean():.0f}%")
        b.metric(f"{name}: MRR", f"{df['Reciprocal rank'].mean():.2f}")
        st.dataframe(df, hide_index=True)


def page_about():
    st.graphviz_chart("""digraph {rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#12202B", fontcolor="white", color="#12202B", fontname="Helvetica"];
      edge [color="#0E7C86"];
      Files -> Reader; Website -> Reader; Reader -> Chunker -> Index;
      Index -> Keyword; Index -> Semantic; Keyword -> Fusion; Semantic -> Fusion;
      Question -> Fusion -> Prompt -> LLM -> Answer; Fusion -> Sources; Sources -> Answer;}""")
    st.dataframe(pd.DataFrame([
        ("Reading", "PDF (with OCR for scanned pages), Word, PowerPoint, Excel, CSV, HTML, text, images, websites", "PyMuPDF, python-docx, python-pptx, pandas, BeautifulSoup, Tesseract"),
        ("Chunking", "About 700-character passages, split at headings and sentences, with overlap", "Python"),
        ("Retrieval", "BM25 keyword search plus optional semantic embeddings, merged by Reciprocal Rank Fusion", "NumPy, fastembed (ONNX)"),
        ("Follow-ups", "Short or pronoun questions reuse the previous question for retrieval", "Python"),
        ("Answering", "Streaming LLM answer that must cite passages, or a no-AI extractive answer", "Gemini, Groq/OpenRouter, Ollama"),
        ("Safety", "Context is treated as data (prompt-injection guard); websites on private networks are blocked", "Python"),
    ], columns=["Stage", "Technique", "Tools"]), hide_index=True)
    st.latex(r"RRF(d)=\sum_{r\in\{keyword,\,semantic\}}\frac{1}{60+rank_r(d)}")
    st.markdown("""**Limits.** Answers are only as good as the text that could be read: poor scans and handwriting reduce quality.
Free AI services have rate limits. Large documents take longer to index. The knowledge base lives in this browser session and is not stored on the server.""")
    st.dataframe(pd.DataFrame([("Python", platform.python_version()), ("Streamlit", st.__version__),
                               ("Semantic search", "available" if ix.embeddings_available() else "not installed"),
                               ("Documents / passages", f"{len(idx().docs())} / {len(idx().chunks)}")],
                              columns=["Component", "Status"]), hide_index=True)


# ───────────────────────── main ─────────────────────────
hero()
S = sidebar()
page = st.segmented_control("Page", PAGES, default="Chat", label_visibility="collapsed") or "Chat"
if page == "Chat":
    page_chat(S)
elif page == "Knowledge base":
    page_kb(S)
elif page == "Insights":
    page_insights()
elif page == "Evaluation":
    page_eval(S)
else:
    page_about()
