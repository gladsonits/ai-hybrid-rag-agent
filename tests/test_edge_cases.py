"""Edge-case tests for the PDF chunker."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from indexing.chunker import PDFChunker  # noqa: E402

PDF = REPO_ROOT / "datasource" / "Rose Gladson AG_.NETAzureOpsFullStack.pdf"


@pytest.fixture(scope="module")
def chunks():
    if not PDF.is_file():
        pytest.skip(f"source PDF not present: {PDF}")
    return PDFChunker(str(PDF)).chunk()


def test_missing_pdf_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        PDFChunker("datasource/does-not-exist.pdf")


def test_empty_source_pdf_env_raises_value_error(monkeypatch):
    monkeypatch.setenv("SOURCE_PDF", "")
    with pytest.raises(ValueError):
        PDFChunker()


def test_chunking_is_deterministic(chunks):
    again = PDFChunker(str(PDF)).chunk()
    assert [c.to_dict() for c in chunks] == [c.to_dict() for c in again]


def test_max_chunk_chars_is_respected():
    if not PDF.is_file():
        pytest.skip(f"source PDF not present: {PDF}")
    small = PDFChunker(str(PDF), max_chunk_chars=200, split_overlap_chars=40).chunk()
    assert small
    assert all(len(c.chunk_text) <= 200 for c in small)


def test_every_chunk_has_text_section_and_page(chunks):
    assert chunks
    for chunk in chunks:
        assert chunk.chunk_text.strip()
        assert chunk.metadata.get("section")
        assert chunk.metadata["page"] >= 1
        assert chunk.metadata["end_page"] >= chunk.metadata["page"]


def test_contact_details_are_kept(chunks):
    """The header line is short but meaningful, so it must survive."""
    header = [c for c in chunks if "372-3475" in c.chunk_text]
    assert header, "the phone number was dropped from the header chunk"


def test_no_bullet_markers_leak_into_chunk_text(chunks):
    for chunk in chunks:
        for marker in ("●", "•", "▪", "◦", "○"):
            assert marker not in chunk.chunk_text


def test_source_pdf_env_var_is_used(monkeypatch):
    if not PDF.is_file():
        pytest.skip(f"source PDF not present: {PDF}")
    monkeypatch.setenv("SOURCE_PDF", str(PDF))
    assert PDFChunker().chunk()
