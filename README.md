# Concert Tour Assistant

`concert_assistant_runtime.py` is a simple, in‑memory Retrieval‑Augmented Generation (RAG) tool for managing and querying concert‑tour documents.

---

## Features

- **Document Ingestion**  
  - Supports PDF and TXT files.  
  - Automatically checks for relevance using keywords (`tour`, `concert`, `venue`).  
  - Generates a concise 2–3 sentence summary of each document.  
  - Stores summaries and full texts as embeddings for fast retrieval.

- **Question Answering**  
  - Embeds user queries and retrieves top‑k relevant documents.  
  - Runs extractive QA on combined contexts to provide accurate answers.

- **Interactive CLI**  
  - Ingest new documents, query ingested tours, view help, or exit—all from a simple prompt.

---

## Prerequisites

- **Python**: Version 3.7 or newer
- **Pip packages**:
  ```bash
  pip install pdfplumber sentence-transformers faiss-cpu transformers
  ```

> **Note**: On some systems, replace `faiss-cpu` with `faiss` if you have GPU support.

---

## Installation

1. **Clone or copy** this repository to your local machine.
2. *(Optional but recommended)* Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Or install manually as shown in Prerequisites.
4. Ensure `concert_assistant_runtime.py` is executable:
   ```bash
   chmod +x concert_assistant_runtime.py
   ```

---

## Usage

1. **Start the assistant**:
   ```bash
   ./concert_assistant_runtime.py
   ```
   or
   ```bash
   python concert_assistant_runtime.py
   ```

2. **Available commands** (at the `> ` prompt):
   ```txt
   ingest <path/to/file.pdf>   # Ingest a new concert-tour PDF or TXT
   query <your question>       # Ask about ingested tours
   help                        # Show help text
   exit | quit                 # Quit the assistant
   ```

3. **Example session**:
   ```txt
   🎵 Concert Tour Assistant (in‑memory RAG) 🎵
   Available commands:
     ingest <path>      — ingest a PDF or TXT concert‑tour doc
     query <question>   — ask about your ingested tours
     help               — show this message
     exit               — quit

   > ingest tours/europe_2025.pdf
   → Generating summary…
     Summary → 'Artist X will tour 10 cities across Europe between June and August 2025.'
   → Embedding summary…
   ✅ Document stored in RAG.

   > query Where will Artist X perform in July?
   → Embedding question…
   → Running extractive QA…

   Answer: "Artist X performs in Paris, Berlin, and Madrid in July."
   ```

---

## Configuration

These constants are defined at the top of `concert_assistant_runtime.py` and can be adjusted if needed:

| Variable     | Description                                   | Default                                         |
|--------------|-----------------------------------------------|-------------------------------------------------|
| `EMBED_MODEL`| Sentence‑Transformers model for embeddings    | `"sentence-transformers/all-MiniLM-L6-v2"`    |
| `SUM_MODEL`  | Hugging Face model for summarization pipeline | `"sshleifer/distilbart-cnn-12-6"`             |
| `QA_MODEL`   | Hugging Face model for QA pipeline            | `"distilbert-base-cased-distilled-squad"`     |
| `EMBED_DIM`  | Dimension of embedding vectors                | `384`                                           |
| `KEYWORDS`   | Keywords for relevance filtering              | `("tour", "concert", "venue")`           |

---

## Troubleshooting

- **`Error: Only concert tour documents accepted.`**  
  Make sure your document contains one of the keywords: `tour`, `concert`, or `venue`.

- **Out‑of‑Memory (OOM) errors** during summarization:  
  The script truncates long texts to the first 10,000 characters by default. You can adjust this limit in the `summarize()` function.

- **Model loading issues**:  
  Ensure you have compatible versions of `transformers`, `sentence-transformers`, and `faiss` installed.

