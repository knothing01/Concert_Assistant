import sys
import shlex

import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# ——— CONFIGURATION ———
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUM_MODEL   = "sshleifer/distilbart-cnn-12-6"
QA_MODEL    = "distilbert-base-cased-distilled-squad"  # extractive QA
EMBED_DIM   = 384
KEYWORDS    = ("tour", "concert", "venue")

# ——— GLOBAL STATE ———
_index = faiss.IndexFlatL2(EMBED_DIM)
_meta  = []  # each entry: {"filename", "summary", "full_text"}

# Load models once at startup
_embedder   = SentenceTransformer(EMBED_MODEL)
_summarizer = pipeline("summarization", model=SUM_MODEL, device=-1)
_qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=-1)

# ——— HELPERS ———

def extract_text(path: str) -> str:
    """Extract raw text from PDF or TXT."""
    if path.lower().endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    elif path.lower().endswith(".txt"):
        return open(path, encoding="utf-8").read()
    else:
        raise ValueError("Unsupported file type (use .pdf or .txt)")

def is_relevant(text: str) -> bool:
    """Simple keyword check."""
    lo = text.lower()
    return any(k in lo for k in KEYWORDS)

def summarize(text: str) -> str:
    """Produce a 1–2 sentence summary."""
    snippet = text[:10000]  # avoid OOM on very long docs
    out = _summarizer(snippet, max_length=60, min_length=20, do_sample=False)
    return out[0]["summary_text"].strip()

def ingest(path: str):
    """Ingest a new document into our in-memory RAG."""
    try:
        txt = extract_text(path)
    except Exception as e:
        print(f"[!] Failed to read “{path}”: {e}")
        return

    if not is_relevant(txt):
        print("Error: Only concert tour documents accepted.")
        return

    print("→ Generating summary…")
    summary = summarize(txt)
    print(f"  Summary → {summary!r}")

    print("→ Embedding summary…")
    vec = _embedder.encode([summary])  # shape (1, 384)

    _index.add(vec)
    _meta.append({
        "filename": path,
        "summary": summary,
        "full_text": txt
    })
    print("✅ Document stored in RAG.")

def query(question: str):
    """Embed the question, retrieve top‑k, then run QA over their contexts."""
    if _index.ntotal == 0:
        print("No tour info found. (Nothing has been ingested yet.)")
        return

    print("→ Embedding question…")
    qvec = _embedder.encode([question])

    # retrieve top 3
    D, I = _index.search(qvec, k=3)

    # build a combined context from the top hits
    ctxs = []
    for idx in I[0]:
        if idx < len(_meta):
            # use the full text as context for QA
            ctxs.append(_meta[idx]["full_text"][:2000])  # truncate to first 2000 chars
    context = "\n\n".join(ctxs)

    if not context:
        print("No tour info found.")
        return

    print("→ Running extractive QA…")
    ans = _qa_pipeline(question=question, context=context, topk=1)
    # ans can be a list or dict depending on transformer version
    if isinstance(ans, list):
        ans = ans[0]

    print("\nAnswer:", ans["answer"])

def help_text():
    print("""
Available commands:
  ingest <path>      — ingest a PDF or TXT concert‑tour doc
  query <question>   — ask about your ingested tours
  help               — show this message
  exit               — quit
""".strip())

# ——— MAIN LOOP ———

def main():
    print("🎵 Concert Tour Assistant (in‑memory RAG) 🎵")
    help_text()

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not line:
            continue

        parts = shlex.split(line)
        cmd = parts[0].lower()
        arg = " ".join(parts[1:])

        if cmd in ("exit", "quit"):
            print("Goodbye!")
            break
        elif cmd == "help":
            help_text()
        elif cmd == "ingest":
            if not arg:
                print("Usage: ingest path/to/file.pdf")
            else:
                ingest(arg)
        elif cmd == "query":
            if not arg:
                print("Usage: query Your question here")
            else:
                query(arg)
        else:
            print(f"Unknown command: {cmd!r}. Type `help`.")

if __name__ == "__main__":
    main()
