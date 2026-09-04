"""
MF Facts-Only RAG Chatbot (Streamlit + all-MiniLM-L6-v2 + Chroma + Groq).

Run:
  pip install -r requirements.txt
  python ingest.py            # once (and whenever sources.txt changes)
  streamlit run rag_app.py

Keys / sources, where to put them:
  1. Source links -> sources.txt (one URL per line), then re-run `python ingest.py`.
  2. Groq key     -> EITHER .env file (GROQ_API_KEY=...) in this folder,
                     OR Streamlit Secrets (.streamlit/secrets.toml),
                     OR paste it in the sidebar at runtime (not stored).
"""
import os
import re
from datetime import date
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

BASE = Path(__file__).parent
CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE / "chroma_db"))
COLLECTION = "mf_faq"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
AS_OF = "2026-09-04"
EDU_TITLE, EDU_URL = "AMFI - Mutual Funds Sahi Hai (investor education)", "https://www.mutualfundssahihai.com/en"

load_dotenv(BASE / ".env")

ADVICE_RE = re.compile(
    r"(should i|shall i|\bbuy\b|\bsell\b|\bhold\b.*\?|best fund|which fund|which is better|"
    r"which.*(better|more).*fund|recommend|suggest a fund|portfolio|how much will|will i earn|"
    r"predict|forecast|top fund|rate this fund|is it good to|is it a good time|compare.*fund|which should i)",
    re.I,
)
RETURNS_RE = re.compile(r"(\breturns?\b|\bcagr\b|\bxirr\b|annualis[ez]ed)", re.I)
PII_RE = re.compile(
    r"([A-Z]{5}[0-9]{4}[A-Z])|(\b\d{4}\s?\d{4}\s?\d{4}\b)|(\botp\b)|"
    r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})|(\b[6-9]\d{9}\b)|"
    r"(aadhaar|pan card|account number|folio number\s*\d|password)",
    re.I,
)
FACT_HINT_RE = re.compile(r"expense|exit load|lock|minimum sip|min sip|benchmark|riskometer|statement|factsheet|ter", re.I)

SYSTEM_PROMPT = (
    "You are a facts-only mutual-fund FAQ assistant. Use ONLY the retrieved context below. "
    "Rules: answer in at most 3 short sentences; include exactly one Source link taken from the "
    "context's source URLs; quote every number EXACTLY as written in the chunk you cite, never "
    "combine or mix figures across chunks; if chunks conflict on the asked metric, ALWAYS prefer "
    "the chunk starting with \"FACTS\", it holds this scheme's own numbers; never report AMC-level / "
    "fund-house-total figures as the scheme's figure; never give buy/sell/hold advice or recommend a fund; never compute, "
    "compare or predict returns, if asked, reply you don't compute/compare returns and point to "
    "the official factsheet; if the context lacks the fact, say so and point to the Groww MF official site "
    "(https://www.growwmf.in/). "
    "Never ask for or repeat PAN, Aadhaar, account numbers, OTPs, emails or phone numbers. Use only plain hyphens (-) and commas; never use em dashes."
)


@st.cache_resource(show_spinner=False)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION)


def guardrails(q: str):
    """Return (kind, title, url) for hard refusals, else None."""
    if PII_RE.search(q):
        return ("pii", EDU_TITLE, EDU_URL)
    if RETURNS_RE.search(q) and "statement" not in q.lower():
        return ("returns", "Groww MF - Official downloads (factsheets)",
                "https://www.growwmf.in/")
    if ADVICE_RE.search(q) and not FACT_HINT_RE.search(q):
        return ("advice", EDU_TITLE, EDU_URL)
    return None


FUND_STOPWORDS = {"groww", "fund", "funds", "mutual", "direct", "growth"}


def _fund_index():
    """[(page_url, [slug tokens])] built from sources.txt fund URLs."""
    try:
        urls = [
            line.strip()
            for line in (BASE / "sources.txt").read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("http") and "groww.in/mutual-funds" in line
        ]
    except Exception:
        return []
    index = []
    for u in urls:
        slug = u.rstrip("/").split("/mutual-funds/")[-1]
        toks = [t for t in slug.split("-") if t not in FUND_STOPWORDS]
        index.append((u, toks))
    return index


FUND_INDEX = _fund_index()
_TOKEN_DF = {}
for _, toks in FUND_INDEX:
    for t in set(toks):
        _TOKEN_DF[t] = _TOKEN_DF.get(t, 0) + 1


def match_fund_url(query: str):
    """If the query names a fund, return its page URL.

    Match when >=2 slug tokens appear as whole words, or 1 token that is
    unique across all fund slugs (e.g. elss, gilt, overnight).
    """
    ql = query.lower()
    best, best_key = None, (0, 0)
    for url, toks in FUND_INDEX:
        hits = [t for t in toks if re.search(r"\b" + re.escape(t) + r"\b", ql)]
        uniq = sum(1 for t in hits if _TOKEN_DF.get(t, 99) == 1)
        key = (len(hits), uniq)
        if key > best_key:
            best_key, best = key, url
    if best and (best_key[0] >= 2 or best_key[1] >= 1):
        return best
    return None


def retrieve(query: str, k: int = 6):
    try:
        col = get_collection()
        if col.count() == 0:
            # Cloud hosts don't deploy chroma_db/: build the index on first use.
            with st.spinner("First run: building the knowledge index from 40 official pages (one-time, a few minutes)..."):
                from ingest import build_index, load_urls
                build_index(col, get_embedder(), load_urls(), log=lambda *a, **k: None)
        model = get_embedder()
        vec = model.encode([query]).tolist()
        # If the query names a fund, scope ALL retrieval to that fund's page.
        # Single-vector similarity lets a distinctive word like "AUM" outweigh
        # the fund name, so unscoped search grabs the first AUM chunk from ANY
        # fund. Scoping makes cross-fund mix-ups impossible. A fact-checker
        # must never answer fund A's question with fund B's number.
        fund_url = match_fund_url(query)
        if fund_url:
            res = col.query(query_embeddings=vec, n_results=k,
                            where={"source_url": fund_url})
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            hits = [{"text": d, "source_url": (m or {}).get("source_url", "")} for d, m in zip(docs, metas)]
        else:
            res = col.query(query_embeddings=vec, n_results=min(k, col.count()))
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            hits = [{"text": d, "source_url": (m or {}).get("source_url", "")} for d, m in zip(docs, metas)]
        # Pin the named fund's fact card first: structured numbers (expense,
        # exit load, min SIP, AUM) must always be in context for metric questions.
        if fund_url:
            try:
                got = col.get(where={"$and": [{"source_url": fund_url}, {"kind": "fact_card"}]}, limit=1)
                if got.get("documents"):
                    card = {"text": got["documents"][0], "source_url": fund_url}
                    hits = [card] + [h for h in hits if h["text"] != card["text"]][: max(0, k - 1)]
            except Exception:
                pass
        return hits
    except Exception as e:
        st.warning(f"Retrieval unavailable ({e}). Run `python ingest.py` first.")
        return []


def get_groq_key() -> str:
    k = (os.getenv("GROQ_API_KEY") or "").strip()
    if k:
        return k
    try:
        return str(st.secrets.get("GROQ_API_KEY", "") or "").strip()
    except Exception:
        return ""


def get_mistral_key() -> str:
    k = (os.getenv("MISTRAL_API_KEY") or "").strip()
    if k:
        return k
    try:
        return str(st.secrets.get("MISTRAL_API_KEY", "") or "").strip()
    except Exception:
        return ""


def _build_messages(query: str, hits):
    context = "\n\n".join(f"[S{i+1}] ({h['source_url']})\n{h['text'][:1500]}" for i, h in enumerate(hits))
    urls = [h["source_url"] for h in hits if h["source_url"]]
    user = (
        f"Question: {query}\n\nRetrieved context:\n{context}\n\n"
        f"Answer in <=3 sentences with exactly one line at the end: 'Source: <one of {urls}>'. "
        f"Today is {date.today().isoformat()}."
    )
    return urls, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def groq_answer(query: str, hits):
    from groq import Groq

    api_key = get_groq_key()
    if not api_key:
        return None, "missing_key"
    urls, messages = _build_messages(query, hits)
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=DEFAULT_GROQ_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=700,
    )
    content = (resp.choices[0].message.content or "").strip()
    return content, urls[0] if urls else ""


def mistral_answer(query: str, hits):
    from mistralai import Mistral

    api_key = get_mistral_key()
    if not api_key:
        return None, "missing_key"
    urls, messages = _build_messages(query, hits)
    client = Mistral(api_key=api_key)
    resp = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=700,
    )
    content = (resp.choices[0].message.content or "").strip()
    return content, urls[0] if urls else ""


def llm_answer(query: str, hits):
    """Groq primary, Mistral automatic failover.

    If Groq errors (rate limit, outage) or returns empty, retry once on
    Mistral. Returns (text|None, first_url, provider|reason) where provider is
    "groq", "mistral", "missing_key" (neither key set) or "provider_failed".
    """
    groq_key, mistral_key = get_groq_key(), get_mistral_key()
    if not groq_key and not mistral_key:
        return None, "", "missing_key"
    errors = []
    if groq_key:
        try:
            text, first_url = groq_answer(query, hits)
            if text:
                return text, first_url, "groq"
            errors.append("groq: empty reply")
        except Exception as e:
            errors.append(f"groq: {e}")
    if mistral_key:
        try:
            text, first_url = mistral_answer(query, hits)
            if text:
                return text, first_url, "mistral"
            errors.append("mistral: empty reply")
        except Exception as e:
            errors.append(f"mistral: {e}")
    st.warning("LLM providers failed (%s); using offline extractive answer."
               % "; ".join(errors)[:300])
    return None, "", "provider_failed"


def extractive_fallback(query: str, hits):
    """No-API-key offline answer: best chunk trimmed to <=3 sentences + citation."""
    if not hits:
        return ("I can only answer factual questions from the ingested official pages. "
                "Try asking about expense ratio, exit load, minimum SIP, ELSS lock-in, "
                "riskometer/benchmark, or statement downloads.",
                "Groww MF - Official site (scope)", "https://www.growwmf.in/")
    top = next((h for h in hits if h["text"].startswith("FACTS")), hits[0])
    sents = re.split(r"(?<=[.!?])\s+", top["text"].replace("Rs.", "Rs").strip())
    text = " ".join(sents[:3])[:900]
    return text, "Retrieved official page", top["source_url"]


# ---------------- UI (Groww-style theme) ----------------
st.set_page_config(page_title="Groww MF FAQ Assistant", page_icon="🌱")

st.markdown("""
<style>
  .stApp { background: #FFFFFF; color: #1E1E1E; }
  header[data-testid="stHeader"] { background: #FFFFFF; border-bottom: 1px solid #E9EBEE; }
  footer { display: none !important; }
  button[data-testid="stSidebarCollapseButton"],
  div[data-testid="stSidebarCollapsedControl"],
  div[data-testid="collapsedControl"] { display: none !important; }
  .block-container { max-width: 880px; padding-top: 1.2rem; padding-bottom: 24px; }
  a { color: #00B386 !important; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .groww-nav { display: flex; align-items: center; gap: 10px; padding: 2px 0 12px;
    border-bottom: 1px solid #E9EBEE; margin-bottom: 14px; }
  .groww-dummy-logo { display: inline-flex; align-items: center; justify-content: center;
    width: 34px; height: 34px; border-radius: 9px; background: #00B386; color: #fff;
    font-size: 22px; font-weight: 800; line-height: 1; }
  .groww-logo { font-size: 26px; font-weight: 800; color: #00B386; letter-spacing: -0.5px; }
  .groww-div { color: #C9CED6; font-size: 20px; }
  .groww-sub { font-size: 15px; color: #44475B; font-weight: 600; }
  .groww-hero { background: linear-gradient(135deg, #E8F9F2 0%, #F4FBF8 60%, #FFFFFF 100%);
    border: 1px solid #CDEFE6; border-radius: 16px; padding: 22px 24px; margin: 6px 0 12px; }
  .groww-hero h1 { font-size: 30px !important; margin: 0 0 6px !important; color: #1E1E1E !important;
    letter-spacing: -0.5px; line-height: 1.2; }
  .groww-hero p { font-size: 15px; color: #44475B; margin: 4px 0; }
  .groww-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
  .groww-pill { background: #fff; border: 1px solid #D5DBE3; border-radius: 999px;
    font-size: 12.5px; color: #1E1E1E; padding: 5px 12px; font-weight: 600; }
  .groww-pill b { color: #00B386; }
  .groww-badge { display: inline-block; background: #00B386; color: #fff; font-size: 12px; font-weight: 700;
    border-radius: 999px; padding: 4px 12px; margin-top: 10px; }
  .groww-label { font-size: 13px; font-weight: 700; color: #44475B; margin: 14px 0 8px; }
  .groww-steps { display: flex; gap: 10px; margin: 12px 0 4px; }
  .groww-step { flex: 1; background: #F4FBF8; border: 1px solid #CDEFE6; border-radius: 14px;
    padding: 12px 14px; font-size: 13px; color: #44475B; }
  .groww-step b { display: block; color: #1E1E1E; font-size: 13.5px; margin-bottom: 2px; }
  .groww-step span { color: #00B386; font-weight: 800; margin-right: 4px; }
  .stButton > button[data-testid="stBaseButton-primary"],
  .stButton > button[data-testid="baseButton-primary"] {
    background: #00B386 !important; color: #fff !important;
    border: 1px solid #00B386 !important; border-radius: 999px !important; font-weight: 700 !important; }
  .stButton > button[data-testid="stBaseButton-primary"]:hover,
  .stButton > button[data-testid="baseButton-primary"]:hover { background: #049E7A !important; }
  .stButton > button[data-testid="stBaseButton-secondary"],
  .stButton > button[data-testid="baseButton-secondary"] {
    background: #fff !important; color: #1E1E1E !important; border: 1px solid #D5DBE3 !important;
    border-radius: 14px !important; font-weight: 600 !important; padding: 12px 14px !important;
    min-height: 76px; white-space: normal; line-height: 1.4; box-shadow: 0 1px 2px rgba(16,24,40,0.05); }
  .stButton > button[data-testid="stBaseButton-secondary"]:hover,
  .stButton > button[data-testid="baseButton-secondary"]:hover {
    border-color: #00B386 !important; background: #F4FBF8 !important; color: #1E1E1E !important; }
  [data-testid="stChatMessage"] { border: 1px solid #E9EBEE; border-radius: 14px; background: #F7FDFB; }
  section[data-testid="stSidebar"] { background: #F8FAF9; border-right: 1px solid #E9EBEE; }
  section[data-testid="stSidebar"] h2 { font-size: 16px !important; color: #1E1E1E !important; }
  section[data-testid="stSidebar"] .stButton > button { width: 100%; }
  .groww-side-logo { display: flex; align-items: center; gap: 8px; margin: 2px 0 2px; }
  .groww-side-note { font-size: 12px; color: #44475B; margin: 10px 0 0; }
  div[data-testid="stElementContainer"]:last-child div[data-testid="stCaptionContainer"] { margin-bottom: 24px; }
  @media (max-width: 640px) {
    .groww-hero h1 { font-size: 24px !important; }
    .groww-steps { flex-direction: column; }
    .block-container { padding-bottom: 40px; }
  }
  section[data-testid="stSidebar"] .stButton > button { width: 100%; }
  .groww-side-logo { display: flex; align-items: center; gap: 8px; margin: 2px 0 2px; }
  .groww-side-note { font-size: 12px; color: #44475B; margin: 10px 0 0; }
</style>
<div class="groww-nav"><span class="groww-dummy-logo">g</span><span class="groww-logo">groww</span><span class="groww-div">|</span>
<span class="groww-sub">Mutual Fund FAQ Assistant</span></div>
<div class="groww-hero">
  <h1>Any question for Groww Mutual Fund AMC?</h1>
  <p>Welcome! Ask in plain words, expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, AUM.</p>
  <p>Every answer gives <b>one official source link</b>. Facts only, no advice, no return predictions, no personal data.</p>
  <div class="groww-pills">
    <span class="groww-pill"><b>36</b> Groww schemes</span>
    <span class="groww-pill"><b>40</b> official sources</span>
    <span class="groww-pill"><b>1390</b> indexed facts</span>
  </div>
  <span class="groww-badge">Facts-only · No investment advice</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="groww-side-logo"><span class="groww-dummy-logo">g</span>'
                '<span class="groww-logo">groww</span></div>', unsafe_allow_html=True)
    st.caption("Mutual Fund FAQ Assistant")
    if st.button("＋ New chat", type="primary"):
        st.session_state.messages = []
        st.rerun()
    query = st.chat_input("Ask about any Groww fund…")
    st.markdown('<div class="groww-side-note">Facts-only · No investment advice. '
                'Never enter PAN, OTP, or account numbers.</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

prefill = st.session_state.pop("prefill", "")
if prefill:
    query = prefill

chat_main, ex_side = st.columns([3, 1])
with ex_side:
    st.markdown("**Try an example**")
    if st.button("Expense ratio · Large Cap?"):
        st.session_state["prefill"] = "What is the expense ratio of Groww Large Cap Fund?"
        st.rerun()
    st.caption("Fund fact-sheet data")
    if st.button("Exit load · ELSS?"):
        st.session_state["prefill"] = "What is the exit load of Groww ELSS Tax Saver Fund?"
        st.rerun()
    st.caption("Fund rules & lock-in")
    if st.button("What is Riskometer?"):
        st.session_state["prefill"] = "What is a Riskometer as per SEBI?"
        st.rerun()
    st.caption("SEBI explainer · official source")
with chat_main:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Auto-scroll only when a conversation exists: on first load (no messages)
    # the hero must stay visible instead of jumping to the bottom.
    if st.session_state.messages:
        try:
            components.html(
                """<script>
                try {
                  const doc = window.parent.document;
                  const main = doc.querySelector('section[data-testid="stMain"]') || doc.querySelector('.main');
                  if (main) { main.scrollTop = main.scrollHeight; }
                } catch (e) {}
                </script>""",
                height=0,
            )
        except Exception:
            pass

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            g = guardrails(query)
            if g and g[0] == "pii":
                out = ("I can't accept PAN, Aadhaar, account/folio numbers, OTPs, emails, or phone numbers. "
                       f"Please re-ask without personal details.\n\nSource: [{g[1]}]({g[2]})\n\nLast updated from sources: {AS_OF}")
            elif g and g[0] == "returns":
                out = ("I don't compute or compare returns. Please check the official monthly factsheet for NAV history "
                       f"and benchmark figures.\n\nSource: [{g[1]}]({g[2]})\n\nLast updated from sources: {AS_OF}")
            elif g and g[0] == "advice":
                out = ("I can't advise on buying, selling, or choosing funds, I share verified facts only. "
                       f"Learn basics here.\n\nSource: [{g[1]}]({g[2]})\n\nLast updated from sources: {AS_OF}")
            else:
                with st.spinner("Fetching verified answer with citation..."):
                    hits = retrieve(query, k=6)
                    text, first_url, provider = llm_answer(query, hits)
                if not text:  # no keys, provider errors, or empty replies -> offline fallback
                    if provider == "missing_key":
                        st.caption("No GROQ_API_KEY or MISTRAL_API_KEY found, showing offline extractive answer from Chroma.")
                    text, title, url = extractive_fallback(query, hits)
                    out = f"{text}\n\nSource: [{title}]({url})\n\nLast updated from sources: {AS_OF}"
                else:
                    # Ensure single citation + as-of line even if model drifts.
                    if "http" not in text and first_url:
                        text += f"\n\nSource: {first_url}"
                    if provider == "mistral":
                        text += "\n\n_Answered by backup provider (Mistral) after Groq was unavailable._"
                    out = text + f"\n\nLast updated from sources: {AS_OF}"
            st.markdown(out)
            st.session_state.messages.append({"role": "assistant", "content": out})

st.divider()
st.caption("Facts-only helper for learning, not investment advice. TER/NAV change daily, so please verify on the "
           "cited official page. Mutual Fund investments are subject to market risks, read all scheme documents carefully.")
