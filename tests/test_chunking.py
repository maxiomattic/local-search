import pytest

from localsearch.chunking import chunk_text, split_sentences


def sentences(n, words=10):
    return [" ".join(f"s{i}w{j}" for j in range(words)) + "." for i in range(n)]


def test_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("One. Two! Three?") == ["One. Two! Three?"]


def test_split_sentences_normalizes_newlines():
    assert split_sentences("A.\r\nB.\rC.") == ["A.", "B.", "C."]


def test_chunks_respect_size_and_cover_everything():
    sents = sentences(60)
    chunks = chunk_text(" ".join(sents), chunk_words=50, overlap_words=20)
    assert all(len(c.split()) <= 50 for c in chunks)
    joined = " ".join(chunks)
    assert all(s in joined for s in sents)


def test_consecutive_chunks_overlap():
    chunks = chunk_text(" ".join(sentences(30)), chunk_words=50, overlap_words=20)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:]):
        assert split_sentences(b)[0] in split_sentences(a)


def test_long_sentence_becomes_its_own_chunk_and_terminates():
    long = " ".join(["word"] * 500) + "."
    chunks = chunk_text(f"Short one. {long} Short two.", chunk_words=50, overlap_words=20)
    assert long in chunks
    assert chunks[-1].endswith("Short two.")


def test_overlap_must_be_smaller():
    with pytest.raises(ValueError):
        chunk_text("a.", chunk_words=10, overlap_words=10)
