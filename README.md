# Groww Mutual Fund Facts-Only RAG Chatbot 🌱

A Groww-styled FAQ chatbot that answers **factual** questions about **36 Groww Mutual Fund schemes** — expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, AUM — grounded **only** in official pages. Every answer carries **one source citation**. No advice, no return predictions, no personal data.

**Stack:** Streamlit UI · `all-MiniLM-L6-v2` embeddings · Chroma vector DB · Groq (`openai/gpt-oss-20b`)

## UI preview

- Groww-green theme (`#00B386`), dummy Groww logo, hero: *"Any question for Groww Mutual Fund AMC?"*
- Fixed left sidebar: ＋New chat, chat bar, 3 corpus-mapped example cards, facts-only note (sidebar cannot be collapsed)
- Right rail with example questions, auto-scroll to each new answer, loading animation instead of raw loader text

## Corpus (40 sources → 1,390 chunks)

- **36 Groww fund pages** (`groww.in/mutual-funds/...`) — see `sources.txt`
- **4 official pages:** Groww AMC (`growwmf.in`), AMFI TER disclosures, AMFI investor education, SEBI riskometer guide

## Quickstart

```bash
pip install -r requirements.txt

# 1. API key → copy .env.example to .env and set GROQ_API_KEY
# 2. Build the index (re-run whenever sources.txt changes)
python ingest.py --reset

# 3. Run
streamlit run rag_app.py   # → http://localhost:8501
```

Without a key the app still answers in offline extractive mode (top Chroma chunk + citation).

## How it works

**Ingest (`ingest.py`)**
1. Fetches each URL in `sources.txt` (full page text, no truncation)
2. Extracts the hidden fund-data JSON on Groww pages into a dense **FACT CARD** chunk per scheme (expense ratio, exit load, min investment/SIP, benchmark, AUM, manager) — this is how JS-rendered numbers become retrievable
3. Chunks everything else into ~700-char sliding windows with 150-char overlap, each prefixed with a `[title | url]` context header
4. Embeds with `all-MiniLM-L6-v2` and stores in Chroma (`./chroma_db`, collection `mf_faq`)

**Query (`rag_app.py`)**
1. Guardrails first: PII (PAN/Aadhaar/OTP/email/phone) → refuse; returns/CAGR questions → factsheet redirect; buy/sell advice → education-link refusal
2. If the query names a fund, retrieval is **scoped to that fund's page only** (a generic word like "AUM" can't outrank the fund name and pull another fund's number), and the fund's FACT CARD is pinned at rank 1
3. Groq answers from that context only: ≤3 sentences, numbers quoted verbatim, exactly one `Source:` link, FACTS chunk preferred on conflicts, AMC-total figures never reported as scheme figures, `Last updated from sources` stamped on every answer

## Repo layout

| File | Purpose |
|---|---|
| `rag_app.py` | Streamlit app: Groww UI + guardrails + retrieval + Groq |
| `ingest.py` | Fetch → fact cards → overlap chunks → embed → Chroma |
| `sources.txt` | 40 source URLs (one per line, `#` = comment) |
| `requirements.txt` | Python deps |
| `.env.example` | Template for `GROQ_API_KEY` (copy to `.env`, never commit it) |
| `.streamlit/config.toml` | Groww-green Streamlit theme |

## Limits

- Figures (NAV/TER/AUM) are a snapshot from ingest time — verify live values on the cited page
- Chroma (`chroma_db/`) is git-ignored; rebuild with `python ingest.py --reset` after editing sources
- First question is slow (~10–30s: embedding model loads into RAM once), then fast

## Disclaimer

Facts-only educational assistant. No investment advice or recommendation. Mutual Fund investments are subject to market risks, read all scheme related documents carefully.
