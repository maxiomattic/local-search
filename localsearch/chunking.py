"""Sentence-aligned sliding-window chunking."""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 100) -> list[str]:
    """Split text into chunks of whole sentences, about `chunk_words` words each.

    Each chunk starts with up to `overlap_words` words of trailing sentences from the
    previous chunk, so a fact that straddles a boundary appears whole in at least one
    chunk. A single sentence longer than `chunk_words` becomes its own chunk.
    """
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    sentences = split_sentences(text)
    lengths = [len(s.split()) for s in sentences]
    chunks: list[str] = []
    i = 0
    while i < len(sentences):
        j, words = i, 0
        while j < len(sentences) and (j == i or words + lengths[j] <= chunk_words):
            words += lengths[j]
            j += 1
        chunks.append(" ".join(sentences[i:j]))
        if j == len(sentences):
            break

        # Step back over trailing sentences to build the overlap, but always move forward.
        k, overlap = j, 0
        while k > i + 1 and overlap + lengths[k - 1] <= overlap_words:
            overlap += lengths[k - 1]
            k -= 1
        i = k
    return chunks
