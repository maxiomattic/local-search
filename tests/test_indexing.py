import os

from PIL import Image

from localsearch import documents, images
from localsearch.qdrant import point_id


class FakeCollection:
    """In-memory stand-in for localsearch.qdrant.Collection."""

    def __init__(self):
        self.points = {}

    def ensure(self):
        pass

    def ids(self):
        return iter(list(self.points))

    def payloads(self, fields):
        return iter([{k: p["payload"][k] for k in fields if k in p["payload"]} for p in self.points.values()])

    def upsert(self, points, wait=False):
        for p in points:
            self.points[p["id"]] = p

    def delete(self, ids):
        for i in ids:
            self.points.pop(i, None)

    def delete_where(self, key, value):
        self.points = {i: p for i, p in self.points.items() if p["payload"].get(key) != value}


def test_point_id_is_deterministic():
    assert point_id("file:///a.jpg") == point_id("file:///a.jpg")
    assert point_id("file:///a.jpg") != point_id("file:///b.jpg")


def test_documents_reindex_is_idempotent_and_tracks_changes(tmp_path, monkeypatch):
    col = FakeCollection()
    calls = []
    monkeypatch.setattr(documents, "collection", lambda: col)
    monkeypatch.setattr(documents, "embed", lambda texts: calls.append(len(texts)) or [[0.0]] * len(texts))

    a = tmp_path / "a.txt"
    a.write_text(" ".join(f"Sentence number {i}." for i in range(200)))
    (tmp_path / "b.md").write_text("Just one line.")
    (tmp_path / "ignored.bin").write_bytes(b"\0")

    first = documents.index(tmp_path)
    assert first == len(col.points) > 2

    # Unchanged files are skipped entirely: no new embeddings, no duplicates.
    calls.clear()
    assert documents.index(tmp_path) == 0
    assert calls == []
    assert len(col.points) == first

    # A shorter rewrite replaces the file's old chunks instead of leaving stale ones.
    a.write_text("Now it is short.")
    os.utime(a, (1, 1))
    documents.index(tmp_path)
    a_chunks = [p for p in col.points.values() if p["payload"]["filename"] == "a.txt"]
    assert [p["payload"]["text"] for p in a_chunks] == ["Now it is short."]


def test_images_index_skips_known_and_prunes_deleted(tmp_path, monkeypatch):
    col = FakeCollection()
    monkeypatch.setattr(images, "collection", lambda: col)

    class FakeModel:
        def encode(self, imgs, batch_size=None):
            import numpy as np

            assert all(max(i.size) <= 448 for i in imgs)  # thumbnails, not full-size decodes
            return np.zeros((len(imgs), 2))

    monkeypatch.setattr(images, "model", lambda: FakeModel())

    (tmp_path / "sub").mkdir()
    Image.new("RGB", (2000, 1000), "red").save(tmp_path / "big.jpg")
    Image.new("RGB", (10, 10), "blue").save(tmp_path / "sub" / "small.PNG")
    (tmp_path / "broken.jpg").write_bytes(b"not an image")

    assert images.index(tmp_path) == 2
    assert images.index(tmp_path) == 0

    (tmp_path / "big.jpg").unlink()
    images.index(tmp_path)
    assert [p["payload"]["filename"] for p in col.points.values()] == ["small.PNG"]
