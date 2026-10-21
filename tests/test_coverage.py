"""Coverage tests: no PDF text may be lost, and none may be invented.

The chunker's own ``build_blocks()`` output is the reference, so these tests
cannot drift from the implementation.
"""

import re
import sys
from collections import Counter
from pathlib import Path

import pytest
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from indexing.chunker import PDFChunker, _BULLET_RE  # noqa: E402

PDF = REPO_ROOT / "datasource" / "Rose Gladson AG_.NETAzureOpsFullStack.pdf"

#: Block kinds emitted verbatim as chunk text.
TEXT_KINDS = {"para", "bullet"}

#: Block kinds that are structure, captured in the metadata.
STRUCTURAL_KINDS = {"heading", "project", "role"}


def words(text):
    """Lower-cased word tokens, bullet markers removed."""
    text = _BULLET_RE.sub(" ", text)
    return re.findall(r"[A-Za-z0-9]+", text.lower())


@pytest.fixture(scope="module")
def chunker():
    if not PDF.is_file():
        pytest.skip(f"source PDF not present: {PDF}")
    return PDFChunker(str(PDF))


@pytest.fixture(scope="module")
def chunks(chunker):
    return chunker.chunk()


@pytest.fixture(scope="module")
def blocks(chunker):
    return chunker.build_blocks()


@pytest.fixture(scope="module")
def reader():
    if not PDF.is_file():
        pytest.skip(f"source PDF not present: {PDF}")
    return PdfReader(str(PDF))


def test_every_body_word_appears_in_the_chunk_text(chunks, blocks):
    body_words = Counter()
    for block in blocks:
        if block.kind in TEXT_KINDS:
            body_words.update(words(block.text))

    chunk_words = Counter()
    for chunk in chunks:
        chunk_words.update(words(chunk.chunk_text))

    missing = body_words - chunk_words
    assert not missing, f"body text dropped from the chunks: {missing.most_common(20)}"


def test_no_pdf_word_is_lost_entirely(chunks, blocks):
    """Every PDF word must be in a chunk text or a chunk's metadata."""
    pdf_words = Counter()
    for block in blocks:
        pdf_words.update(words(block.text))

    seen = set()
    for chunk in chunks:
        for key, value in chunk.metadata.items():
            seen.update(words(str(key)))
            seen.update(words(str(value)))
    for chunk in chunks:
        seen.update(words(chunk.chunk_text))

    missing = {w: c for w, c in pdf_words.items() if w not in seen}
    assert not missing, f"PDF text lost: {sorted(missing.items())[:20]}"


def test_no_word_is_invented(chunks, blocks):
    pdf_words = set()
    for block in blocks:
        pdf_words.update(words(block.text))

    for chunk in chunks:
        extra = set(words(chunk.chunk_text)) - pdf_words
        assert not extra, f"chunk invents words: {sorted(extra)}"


def test_every_text_block_produced_a_chunk(chunker, chunks, blocks):
    chunk_texts = [c.chunk_text for c in chunks]
    for block in blocks:
        if block.kind not in TEXT_KINDS:
            continue
        if len(block.text) < chunker.min_useful_chars:
            continue
        assert any(block.text[:60] in text for text in chunk_texts), (
            f"p{block.page} block produced no chunk: {block.text[:60]!r}"
        )


def test_every_section_heading_is_a_metadata_section(chunks, blocks):
    sections = {c.metadata["section"] for c in chunks}
    for block in blocks:
        if block.kind != "heading":
            continue
        title = block.text.rstrip(":").strip()
        assert title in sections, f"heading {title!r} is not a section"


def test_every_project_header_is_in_the_metadata(chunks, blocks):
    for block in blocks:
        if block.kind != "project":
            continue
        captured = " ".join(
            " ".join(
                str(c.metadata.get(k, "")) for k in ("project", "client", "period")
            )
            for c in chunks
        ).lower()
        residual = [
            w for w in words(block.text) if w not in ("project", "role") and w not in captured
        ]
        assert not residual, f"project header not captured: {block.text[:50]!r} {residual}"


def test_every_role_header_is_in_the_metadata(chunks, blocks):
    captured = " ".join(str(c.metadata.get("role", "")) for c in chunks).lower()
    for block in blocks:
        if block.kind != "role":
            continue
        residual = [w for w in words(block.text) if w != "role" and w not in captured]
        assert not residual, f"role header not captured: {block.text[:50]!r} {residual}"


def test_every_page_produces_at_least_one_chunk(chunks, reader, blocks):
    page_chunks = Counter()
    for chunk in chunks:
        for page in range(chunk.metadata["page"], chunk.metadata["end_page"] + 1):
            page_chunks[page] += 1

    for page in range(1, len(reader.pages) + 1):
        assert page_chunks.get(page), f"page {page} produced no chunk"


def test_chunk_page_numbers_are_valid(chunks, reader):
    for chunk in chunks:
        assert 1 <= chunk.metadata["page"] <= len(reader.pages)
        assert 1 <= chunk.metadata["end_page"] <= len(reader.pages)
        assert chunk.metadata["end_page"] >= chunk.metadata["page"]


def test_every_chunk_maps_back_to_a_source_block(chunker, chunks, blocks):
    normalised = {" ".join(b.text.split()).lower() for b in blocks}
    for chunk in chunks:
        needle = " ".join(chunk.chunk_text.split()).lower()[:60]
        assert any(needle in text for text in normalised), (
            f"chunk maps to no block: {chunk.chunk_text[:60]!r}"
        )
