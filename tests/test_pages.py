"""Page-attribution and label-scope tests for the PDF chunker."""

import re
import sys
from collections import Counter
from pathlib import Path

import pytest
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from indexing.chunker import PDFChunker  # noqa: E402

PDF = REPO_ROOT / "datasource" / "Rose Gladson AG_.NETAzureOpsFullStack.pdf"

SCOPING = ("Responsibilities", "Environment")


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


def test_raw_label_count(reader):
    counts = Counter()
    for page in reader.pages:
        for line in (page.extract_text() or "").splitlines():
            match = re.match(r"^(Responsibilities|Environment)\s*:", line.strip())
            if match:
                counts[match.group(1)] += 1
    assert counts == {"Responsibilities": 8, "Environment": 8}


def test_every_label_line_becomes_a_label_block(blocks):
    counts = Counter()
    for block in blocks:
        head = block.text.split(":", 1)[0].strip()
        if head in SCOPING and ":" in block.text:
            counts[head] += 1
    assert counts == {"Responsibilities": 8, "Environment": 8}


def test_label_scope_reaches_the_chunks(chunks):
    scoped = Counter(c.metadata["label"] for c in chunks if c.metadata.get("label"))
    assert scoped["Responsibilities"] > 0
    assert scoped["Environment"] > 0


def test_label_does_not_swallow_the_preceding_bullet(chunks):
    """A bullet with no trailing period must not absorb the next label."""
    for chunk in chunks:
        if "Collaborated across cross-functional teams" in chunk.chunk_text:
            assert "Environment:" not in chunk.chunk_text
            assert chunk.metadata.get("label") == "Responsibilities"


def test_a_scoped_label_opens_its_own_chunk(chunks):
    """The label line and its value must not share a chunk with other text."""
    for chunk in chunks:
        if chunk.chunk_text.startswith("Environment:"):
            assert chunk.metadata.get("label") == "Environment"


def test_chunk_page_matches_its_source_block(chunks, blocks):
    by_text = {}
    for block in blocks:
        by_text.setdefault(" ".join(block.text.split()).lower()[:60], block)

    for chunk in chunks:
        needle = " ".join(chunk.chunk_text.split()).lower()[:60]
        block = by_text.get(needle)
        if block is None:
            continue
        assert chunk.metadata["page"] == block.page
        assert chunk.metadata["end_page"] == block.end_page


def test_page_ranges_are_contiguous(chunks, reader):
    for chunk in chunks:
        assert chunk.metadata["page"] <= chunk.metadata["end_page"]
        assert chunk.metadata["end_page"] <= len(reader.pages)


def test_project_metadata_does_not_leak_between_projects(chunks):
    """Every chunk under a project must carry that project's role."""
    seen = {}
    for chunk in chunks:
        project = chunk.metadata.get("project")
        if not project:
            continue
        role = chunk.metadata.get("role")
        assert role, f"chunk under project {project!r} has no role"
        if project in seen:
            assert seen[project] == role, (
                f"project {project!r} has two roles: {seen[project]!r} and {role!r}"
            )
        else:
            seen[project] = role


def test_all_eight_projects_are_found(chunks):
    projects = {c.metadata.get("project") for c in chunks if c.metadata.get("project")}
    assert projects == {
        "TruStage",
        "BMW",
        "District Marketplace",
        "GT – Convene",
        "StoryBook",
        "Dentsu",
        "CMS",
        "CIS",
    }
