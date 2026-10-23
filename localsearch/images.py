"""Photo indexing and text-to-image search with CLIP."""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

from PIL import Image, ImageOps

from .qdrant import Collection, point_id

MODEL_NAME = "clip-ViT-B-32"
DIM = 512
COLLECTION = "images"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BATCH_SIZE = 32

Progress = Callable[[int, int], None]


@cache
def model():
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(os.cpu_count() or 1)
    return SentenceTransformer(MODEL_NAME, device="cpu")


def collection() -> Collection:
    return Collection(COLLECTION, DIM)


def find_images(root: Path) -> Iterator[Path]:
    for dirpath, _, files in os.walk(root):
        for name in files:
            if Path(name).suffix.lower() in EXTENSIONS:
                yield Path(dirpath, name).resolve()


def load_thumbnail(path: Path) -> Image.Image | None:
    """Open, auto-rotate, and immediately shrink an image.

    CLIP only sees 224x224, so decoding full-size photos and keeping them around is
    what blows up memory on large libraries. Shrinking here keeps each batch small.
    """
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((448, 448), Image.Resampling.BILINEAR)
            return img
    except Exception as e:  # corrupt or unsupported file: skip, don't abort the run
        print(f"skipping {path}: {e}")
        return None


def index(root: Path, force: bool = False, prune: bool = True, progress: Progress | None = None) -> int:
    """Embed every new image under `root`. Returns the number of images indexed.

    Point IDs are derived from the absolute path, so only images not already in the
    collection are embedded. With `prune`, points for files under `root` that no
    longer exist are removed.
    """
    col = collection()
    col.ensure()

    files = {point_id(f"file://{p}"): p for p in find_images(root)}
    existing = set() if force else set(col.ids())
    todo = [(pid, p) for pid, p in files.items() if pid not in existing]

    if prune and not force:
        prefix = str(root.resolve())
        stale = [
            point_id(f"file://{pl['path']}")
            for pl in col.payloads(["path"])
            if pl.get("path", "").startswith(prefix) and not Path(pl["path"]).exists()
        ]
        col.delete(stale)

    # Decode on a thread pool while the model embeds the previous batch. maxsize=1 keeps
    # at most one decoded batch waiting, which bounds memory regardless of library size.
    batches: queue.Queue = queue.Queue(maxsize=1)

    def produce() -> None:
        with ThreadPoolExecutor() as pool:
            for start in range(0, len(todo), BATCH_SIZE):
                chunk = todo[start : start + BATCH_SIZE]
                imgs = list(pool.map(lambda item: load_thumbnail(item[1]), chunk))
                batches.put([(pid, p, img) for (pid, p), img in zip(chunk, imgs) if img is not None])
        batches.put(None)

    threading.Thread(target=produce, daemon=True).start()

    done = 0
    seen = 0
    while (batch := batches.get()) is not None:
        seen += BATCH_SIZE
        if batch:
            vectors = model().encode([img for _, _, img in batch], batch_size=len(batch)).tolist()
            col.upsert(
                [
                    {"id": pid, "vector": vec, "payload": {"path": str(p), "filename": p.name}}
                    for (pid, p, _), vec in zip(batch, vectors)
                ]
            )
            done += len(batch)
        if progress:
            progress(min(seen, len(todo)), len(todo))
    return done


def search(text: str, limit: int = 8) -> list[dict]:
    vector = model().encode(text).tolist()
    return collection().search(vector, limit)
