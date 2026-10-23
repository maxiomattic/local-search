"""Document indexing and semantic search with a local Ollama embedding model."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import requests

from .chunking import chunk_text
from .qdrant import Collection, point_id

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.environ.get("EMBED_MODEL", "nomic-embed-text")
DIM = 768
COLLECTION = "documents"
EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

Progress = Callable[[int, int], None]


def collection() -> Collection:
    return Collection(COLLECTION, DIM)


def embed(texts: list[str]) -> list[list[float]]:
    res = requests.post(f"{OLLAMA_URL}/api/embed", json={"model": MODEL_NAME, "input": texts}, timeout=300)
    res.raise_for_status()
    return res.json()["embeddings"]


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix == ".docx":
        import docx

        return "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    raise ValueError(f"unsupported file type: {path}")


def find_documents(root: Path) -> Iterator[Path]:
    for dirpath, _, files in os.walk(root):
        for name in files:
            if Path(name).suffix.lower() in EXTENSIONS:
                yield Path(dirpath, name).resolve()


def index(root: Path, force: bool = False, progress: Progress | None = None) -> int:
    """Chunk and embed every new or changed document under `root`. Returns chunks written.

    Each chunk stores its file's mtime. A file whose mtime matches what's stored is
    skipped; a changed file has its old chunks deleted before the new ones are written.
    """
    col = collection()
    col.ensure()

    indexed = {} if force else {pl["path"]: pl.get("mtime") for pl in col.payloads(["path", "mtime"]) if "path" in pl}
    files = list(find_documents(root))
    written = 0
    for n, path in enumerate(files, 1):
        mtime = path.stat().st_mtime
        if indexed.get(str(path)) != mtime:
            try:
                chunks = [c for c in chunk_text(extract_text(path)) if c.strip()]
            except Exception as e:
                print(f"skipping {path}: {e}")
                chunks = []
            col.delete_where("path", str(path))
            if chunks:
                col.upsert(
                    [
                        {
                            "id": point_id(f"file://{path}#{i}"),
                            "vector": vec,
                            "payload": {
                                "path": str(path),
                                "filename": path.name,
                                "mtime": mtime,
                                "chunk": i + 1,
                                "chunks": len(chunks),
                                "text": chunk,
                            },
                        }
                        for i, (chunk, vec) in enumerate(zip(chunks, embed(chunks)))
                    ]
                )
                written += len(chunks)
        if progress:
            progress(n, len(files))
    return written


def search(text: str, limit: int = 5) -> list[dict]:
    return collection().search(embed([text])[0], limit)
