# MF Facts-Only RAG Chatbot — where to put links & keys

Stack: **Streamlit + sentence-transformers `all-MiniLM-L6-v2` + Chroma + Groq** (`llama-3.1-8b-instant` default).

## 1. Where to paste your source links
File: **`mf-faq-assistant-rag/sources.txt`**
- One URL per line. Lines starting with `#` are ignored.
- 22 official URLs are pre-filled — **replace them with the links you provide**.
- After every edit, rebuild the vector DB:
  ```
  cd mf-faq-assistant-rag
  python ingest.py
  ```
- To wipe and rebuild: `python ingest.py --reset`
- This creates/updates `./chroma_db/` (the Chroma persistent store, collection `mf_faq`).

## 2. Where to put the Groq API key (pick ONE)
| Option | Where | How |
|---|---|---|
| A. `.env` file (recommended local) | `mf-faq-assistant-rag/.env` | Copy `.env.example` → `.env`, set `GROQ_API_KEY=gsk_...` |
| B. Streamlit Secrets (for Streamlit Cloud) | `mf-faq-assistant-rag/.streamlit/secrets.toml` | `GROQ_API_KEY = "gsk_..."` |
| C. Sidebar (no file, session only) | App sidebar → “GROQ_API_KEY” box | Paste at runtime; not written to disk |

Without a key the app still works in **offline extractive mode** (top Chroma chunk + citation), but LLM phrasing needs the key.

## 3. Run it
```
cd mf-faq-assistant-rag
pip install -r requirements.txt
python ingest.py
streamlit run rag_app.py
```
UI includes: welcome line, 3 example questions, “Facts-only. No investment advice.” note, and **＋ New chat** button. Every answer shows one source link + `Last updated from sources: 2026-09-04`.

## 4. How it meets the brief
- **RAG:** `ingest.py` chunks pages (~700 chars), embeds with `all-MiniLM-L6-v2` (384-dim), stores in Chroma; `rag_app.py` retrieves top-4 and grounds Groq on that context only.
- **Facts-only:** system prompt forces ≤3 sentences, exactly one Source link, refuses buy/sell advice, refuses to compute/compare returns, refuses PII.
- **No PII stored:** chat lives only in `st.session_state`; Chroma stores only public page text + source URLs.
