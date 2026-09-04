"""
Ingest step of the RAG chatbot.
Reads URLs from sources.txt, fetches text, chunks, embeds with
all-MiniLM-L6-v2, stores in Chroma (./chroma_db).

Usage:
  pip install -r requirements.txt
  python ingest.py            # uses sources.txt + ./chroma_db
  python ingest.py --reset    # delete ./chroma_db first and rebuild
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = Path(__file__).parent
SOURCES = BASE / "sources.txt"
CHROMA_DIR = BASE / "chroma_db"
COLLECTION = "mf_faq"
EMBED_MODEL = "all-MiniLM-L6-v2"

UA = {"User-Agent": "Mozilla/5.0 (MF-FAQ-RAG ingestion; educational prototype)"}


def load_urls():
    urls = []
    for line in SOURCES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("http"):
            urls.append(line)
    # de-dupe, keep order
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _jfield(blob: str, name: str):
    """Extract a simple JSON string/number field value from a raw blob."""
    m = re.search(r'"%s"\s*:\s*("(?:[^"\\]|\\.)*"|[\d.\-]+|true|false|null)' % re.escape(name), blob)
    if not m:
        return None
    v = m.group(1)
    if v.startswith('"'):
        try:
            return json.loads(v)
        except Exception:
            return v.strip('"')
    if v in ("true", "false", "null"):
        return None
    try:
        return float(v) if "." in v else int(v)
    except Exception:
        return v


def extract_groww_facts(html: str, url: str):
    """Pull the embedded fund-facts JSON (numbers the JS app hydrates from)
    and render one dense FACT CARD chunk. Returns None if not a fund page."""
    if "groww.in/mutual-funds" not in url or '"scheme_name"' not in html:
        return None
    g = lambda n: _jfield(html, n)  # noqa: E731
    name = g("scheme_name")
    if not name:
        return None

    def rs(v):
        if v is None:
            return None
        try:
            return f"Rs {float(v):,.0f}".replace(",", ",")
        except Exception:
            return f"Rs {v}"

    parts = [f"FACTS — {name}"]
    house, cat, sub = g("fund_house"), g("category"), g("sub_category")
    scope = " | ".join(x for x in [house, f"{cat} - {sub}" if cat or sub else None] if x)
    if scope:
        parts.append(f"({scope})")
    facts = [
        ("Benchmark", g("benchmark") or g("benchmark_name")),
        ("Expense ratio", f"{g('expense_ratio')}%" if g("expense_ratio") is not None else None),
        ("Exit load", g("exit_load")),
        ("Minimum investment", rs(g("min_investment_amount"))),
        ("Additional investment", rs(g("mini_additional_investment"))),
        ("Minimum SIP", rs(g("min_sip_investment"))),
        ("Minimum withdrawal", rs(g("min_withdrawal"))),
        ("Fund AUM (this scheme only)", f"Rs {float(g('aum')):,.2f} crore" if g("aum") is not None else None),
        ("Fund manager", g("fund_manager")),
        ("Launch date", g("launch_date")),
    ]
    for label, val in facts:
        if val:
            parts.append(f"{label}: {str(val).rstrip('.')}.")
    parts.append("Any AMC-level total AUM mentioned elsewhere on the page is NOT this scheme's AUM.")
    return " ".join(parts)


def fetch_page(url: str, timeout: int = 25):
    """Return (title, text, fact_card|None)."""
    # PDFs: store the URL itself as a retrievable record (no PDF parsing dep).
    if url.lower().endswith(".pdf") or ".pdf?" in url.lower():
        return url, f"Official PDF document. See full facts, portfolio and riskometer in the PDF: {url}", None
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)  # full page text — no truncation
    title = soup.title.get_text(strip=True) if soup.title else url
    fact_card = extract_groww_facts(html, url)
    return title, f"{title}\n\n{text}", fact_card


def chunk_text(text: str, size: int = 700, overlap: int = 150, header: str = ""):
    """Sentence-packed sliding windows.

    Every chunk carries (a) a context header [page title | url] so it is
    self-describing for retrieval, and (b) a trailing character overlap from
    the previous chunk so facts split across boundaries stay answerable.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    windows, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= size:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                windows.append(cur)
            cur = s
            while len(cur) > size:  # single huge sentence: hard cut with overlap
                windows.append(cur[:size])
                cur = cur[size - overlap:]
    if cur:
        windows.append(cur)
    out = []
    for w in windows:
        if out:  # overlap: carry tail of previous chunk for shared context
            w = (out[-1][-overlap:] + " " + w).strip()
            if len(w) > size + overlap + 50:
                w = w[-(size + overlap):]
        if header:
            w = f"{header} {w}"
        if len(w.strip()) > 80:
            out.append(w.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete ./chroma_db before rebuild")
    args = ap.parse_args()

    urls = load_urls()
    if not urls:
        print(f"No URLs found in {SOURCES}. Paste one URL per line, then re-run.")
        sys.exit(1)
    print(f"Loaded {len(urls)} URLs from sources.txt")

    if args.reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("Cleared ./chroma_db")

    # Lazy imports so --help works without heavy deps.
    from sentence_transformers import SentenceTransformer
    import chromadb

    print(f"Loading embedding model {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    ok, failed = 0, []
    for i, url in enumerate(urls):
        try:
            title, text, fact_card = fetch_page(url)
        except Exception as e:
            print(f"  [skip] {url} -> {e}")
            failed.append(url)
            continue
        header = f"[{title} | {url}]"
        page_chunks = []
        if fact_card:  # structured numbers first: always retrievable as one unit
            page_chunks.append(f"{header} {fact_card}")
        page_chunks += chunk_text(text, header=header)
        for j, ch in enumerate(page_chunks):
            ids.append(f"doc{i}-c{j}")
            docs.append(ch)
            metas.append({"source_url": url, "doc_id": i, "chunk": j,
                          "kind": "fact_card" if j == 0 and fact_card else "text"})
        ok += 1
        print(f"  [{ok}/{len(urls)}] {url} -> {len(page_chunks)} chunks"
              f"{' (incl. fact card)' if fact_card else ''}")

    if not docs:
        print("Nothing to index. Check sources.txt / network.")
        sys.exit(1)

    print(f"Embedding {len(docs)} chunks with {EMBED_MODEL} ...")
    vectors = model.encode(docs, batch_size=32, show_progress_bar=False).tolist()

    # Replace existing docs with same ids (idempotent re-ingest).
    try:
        col.delete(ids=ids)
    except Exception:
        pass
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
    print(f"Done. Indexed {len(docs)} chunks from {ok} pages into {CHROMA_DIR} (collection '{COLLECTION}').")
    if failed:
        print(f"Skipped {len(failed)} URL(s). Re-run later or replace them in sources.txt:")
        for u in failed:
            print(f"  - {u}")


if __name__ == "__main__":
    main()
