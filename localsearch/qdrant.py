"""Minimal Qdrant REST client: just the calls the indexers and search need."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterable, Iterator

import requests

DEFAULT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def point_id(key: str) -> str:
    """Deterministic point ID, so re-indexing the same item overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class Collection:
    def __init__(self, name: str, dim: int, url: str = DEFAULT_URL, timeout: float = 30):
        self.name = name
        self.dim = dim
        self.base = f"{url.rstrip('/')}/collections/{name}"
        self.timeout = timeout
        self.http = requests.Session()

    def ensure(self) -> None:
        res = self.http.get(self.base, timeout=self.timeout)
        if res.status_code == 404:
            self.http.put(
                self.base,
                json={"vectors": {"size": self.dim, "distance": "Cosine"}},
                timeout=self.timeout,
            ).raise_for_status()
        else:
            res.raise_for_status()

    def ids(self) -> Iterator[str]:
        """Yield every point ID in the collection (paged scroll, no payloads or vectors)."""
        offset = None
        while True:
            body: dict = {"limit": 1000, "with_payload": False, "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            res = self.http.post(f"{self.base}/points/scroll", json=body, timeout=self.timeout)
            if res.status_code == 404:
                return
            res.raise_for_status()
            result = res.json()["result"]
            for p in result.get("points", []):
                yield str(p["id"])
            offset = result.get("next_page_offset")
            if offset is None:
                return

    def upsert(self, points: list[dict], wait: bool = False) -> None:
        if not points:
            return
        self.http.put(
            f"{self.base}/points",
            params={"wait": str(wait).lower()},
            json={"points": points},
            timeout=self.timeout,
        ).raise_for_status()

    def delete(self, ids: Iterable[str]) -> None:
        ids = list(ids)
        if not ids:
            return
        self.http.post(
            f"{self.base}/points/delete",
            params={"wait": "true"},
            json={"points": ids},
            timeout=self.timeout,
        ).raise_for_status()

    def delete_where(self, key: str, value: str) -> None:
        """Delete every point whose payload field `key` equals `value`."""
        self.http.post(
            f"{self.base}/points/delete",
            params={"wait": "true"},
            json={"filter": {"must": [{"key": key, "match": {"value": value}}]}},
            timeout=self.timeout,
        ).raise_for_status()

    def payloads(self, fields: list[str]) -> Iterator[dict]:
        """Yield the selected payload fields of every point (paged scroll)."""
        offset = None
        while True:
            body: dict = {"limit": 1000, "with_payload": {"include": fields}, "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            res = self.http.post(f"{self.base}/points/scroll", json=body, timeout=self.timeout)
            if res.status_code == 404:
                return
            res.raise_for_status()
            result = res.json()["result"]
            for p in result.get("points", []):
                yield p.get("payload") or {}
            offset = result.get("next_page_offset")
            if offset is None:
                return

    def search(self, vector: list[float], limit: int) -> list[dict]:
        res = self.http.post(
            f"{self.base}/points/search",
            json={"vector": vector, "limit": limit, "with_payload": True},
            timeout=self.timeout,
        )
        res.raise_for_status()
        return res.json()["result"]
