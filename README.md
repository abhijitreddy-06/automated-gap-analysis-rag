# NexusAI (RAG) — Compliance Engine

Short description

- Small RAG-based compliance engine that ingests policy PDFs, builds a Chroma vector DB, and answers policy questions using LLMs.

Prerequisites

- Python 3.10+ (pyproject specifies >=3.10,<3.14)
- Git
- (Optional) Poetry if you prefer managing the project via Poetry
- Set up required API credentials in a `.env` file (for Google GenAI keys, etc.)

Quick start (pip)

1. Create a virtualenv and activate it

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Add `.env` with required values (example)

```
GOOGLE_API_KEY=your_key_here
```

4. Run the Streamlit app

```bash
streamlit run app.py
```

Notes

- Upload a single PDF via the UI (Policy Upload tab) — the app saves to `uploads/` and runs the analysis pipeline.
- The project currently uses a local Chroma DB at `nexus_chroma_db_local/` — included in `.gitignore` to avoid committing data.
- The provided `requirements.txt` pins top-level deps. To export a full pip freeze from Poetry run:

```bash
poetry export -f requirements.txt --output full-requirements.txt --without-hashes
```

Suggested next steps

- Add tests for core pipelines (`document_classifier`, `ingestion_service`).
- Add GitHub Actions workflow for linting and tests.
- Add input validation and more robust error handling around file I/O and external API calls.
