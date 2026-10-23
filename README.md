# local-search

An experiment in private semantic search over a personal photo library and documents, running entirely on your own machine. You type "kids asleep on a couch" and get the matching photos, or ask a question and get the passages from your PDFs and Word files that answer it. Nothing is uploaded anywhere. I used it to explore multimodal embeddings, vector search, and retrieval chunking on consumer hardware. See [About this experiment](#about-this-experiment).

- **Photos** are embedded with [CLIP](https://openai.com/research/clip) (`clip-ViT-B-32` via sentence-transformers) on the CPU. Text queries land in the same vector space as the images.
- **Documents** (`.pdf`, `.docx`, `.txt`, `.md`) are split into overlapping, sentence-aligned chunks and embedded with `nomic-embed-text` through a local [Ollama](https://ollama.com).
- Vectors live in [Qdrant](https://qdrant.tech). There's a CLI and a Streamlit UI.

## Design notes

- **Incremental and idempotent.** Point IDs are UUIDv5s of the file path (and chunk index), so re-running an index never duplicates anything. Photos that are already indexed are skipped, and deleted photos are pruned. Documents store their mtime: unchanged files are skipped, and a changed file has its old chunks replaced.
- **Bounded memory on large libraries.** Images are decoded on a thread pool, rotated by EXIF, and shrunk to 448 px straight away, since CLIP only sees 224×224. A queue of size 1 between the decoder and the model keeps at most one decoded batch waiting. Memory stays flat no matter how many photos there are, and decoding overlaps with embedding.
- **Chunking keeps context.** Chunks are about 250 words of whole sentences, and each one starts with about 100 words from the end of the previous chunk. A fact that straddles a boundary therefore appears whole in at least one chunk. A single very long sentence becomes its own chunk instead of stalling the loop.
- **Fault tolerant.** A corrupt image or an unreadable document is logged and skipped. It doesn't abort the run.

## Setup

```bash
docker compose up -d                     # Qdrant on 127.0.0.1:6333
ollama pull nomic-embed-text             # only needed for documents
python -m venv .venv && . .venv/bin/activate
pip install -e '.[ui,dev]'
```

## Use

```bash
localsearch index images ~/Pictures
localsearch search images "a red car in the snow" -n 8

localsearch index docs ~/Documents/notes
localsearch search docs "what did the lecture say about the flood narrative"

streamlit run app.py                     # web UI with Photos and Documents tabs
```

Configuration is through environment variables: `QDRANT_URL`, `OLLAMA_URL`, and `EMBED_MODEL`.

## Tests

```bash
pytest
```

The tests cover the chunker's edge cases. They also run both indexers against an in-memory fake Qdrant to check that re-indexing adds nothing, that changed documents replace their old chunks, that deleted photos are pruned, and that the model only ever receives thumbnails.

## About this experiment

The goal was to test whether useful semantic search over a real family photo archive and a set of course readings could run locally, privately, and CPU-only, on open models and self-hosted services. AI coding agents wrote the code, and the concept and constraints were mine. It began as two separate scripts written with AI assistants in June 2026: CLIP photo search and document retrieval. Before publishing, I had Claude Code merge them into one package with a shared Qdrant client, a CLI, and a UI. It also replaced random point IDs with deterministic ones, so re-indexing no longer duplicates every document chunk, and added the test suite.

## Layout

```
localsearch/
  qdrant.py      thin Qdrant REST client (ensure, scroll, upsert, delete, filter-delete, search)
  chunking.py    sentence-aligned sliding-window chunker
  images.py      CLIP photo indexer and text-to-image search
  documents.py   text extraction, chunking, Ollama embeddings, and document search
  cli.py         `localsearch` command
app.py           Streamlit UI
```
