"""Structure-aware chunking for resumes and other headed documents.

Splits a PDF into semantically meaningful chunks that carry metadata rich
enough for both keyword (BM25) and dense (embedding) retrieval:

    {
        "chunk_text": "Masters in Computer Applications from Indira Gandhi ...",
        "metadata": {"section": "EDUCATIONAL QUALIFICATION"},
    }

    {
        "chunk_text": "Designed and implemented GenAI-powered chat assistants ...",
        "metadata": {
            "section": "PROFESSIONAL EXPERIENCE",
            "project": "TruStage",
            "role": "Cloud Engineer with AI Capabilities",
        },
    }

How it works
------------
1.  Extract the text of every page, keeping the physical line breaks that
    the PDF layout engine produced.
2.  Classify each physical line as a section heading, project header, role
    header, label, bullet, or plain paragraph.
3.  Un-wrap the physical lines back into logical blocks, re-joining the
    lines that the layout engine had wrapped mid-sentence.
4.  Walk the blocks carrying the project / role context forward, so every
    chunk below a project header inherits that project and role.
5.  Cut any block that is too long for an embedding model into overlapping
    pieces that break on sentence boundaries.

The chunker is deliberately document-shape agnostic: the section, project and
role patterns are plain regular expressions that can be overridden through
``section_rules`` and the constructor arguments.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

# --------------------------------------------------------------------------
# Tuning knobs
# --------------------------------------------------------------------------

#: Hard ceiling on chunk length in characters. Dense embedding models truncate
#: well before this, so anything larger loses its tail during embedding.
MAX_CHUNK_CHARS = 1200

#: Chunks shorter than this are dropped as noise (stray headers, page numbers).
MIN_USEFUL_CHARS = 25

#: Overlap between adjacent chunks when a block has to be split by size.
SPLIT_OVERLAP_CHARS = 100

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

#: A section heading: a short, mostly-uppercase line. The trailing colon is
#: optional because most resumes write "EDUCATIONAL QUALIFICATION:".
_SECTION_RE = re.compile(
    r"^[A-Z][A-Z0-9&/,\.\+\(\)'\-]*(?:[ ][A-Z0-9&/,\.\+\(\)'\-]+)*:?$"
)

#: Uppercase words that show up on their own line but are not sections.
_SECTION_STOPWORDS = frozenset(
    {
        "USA",
        "UK",
        "NET",
        "API",
        "SQL",
        "AWS",
        "GCP",
        "IBM",
        "EY",
        "BMW",
        "CMS",
        "CIS",
        "IT",
        "HR",
        "QA",
    }
)

#: Bullet markers at the start of a line. Resumes mix several: "●", "•", "▪".
_BULLET_RE = re.compile(
    r"^\s*(?:[-–—•●▪◦○*]|\d{1,2}[.)])\s+"
)

#: A "Label:" line, e.g. "Responsibilities:" or "Environment: .NET 8, ...".
#: Only a small set of labels scopes the content beneath it; a bare
#: "Word:" pattern would also match contact details such as
#: "Email: gladson.bits@gmail.com".
_LABEL_RE = re.compile(r"^(?P<label>[A-Z][A-Za-z /&]{0,30}):(?P<value>.*)$")

#: Labels that scope the blocks beneath them.
_SCOPING_LABELS = frozenset(
    {
        "Responsibilities",
        "Environment",
        "Technologies",
        "Tools",
        "Skills",
        "Highlights",
        "Key Achievements",
        "Achievements",
        "Summary",
        "Profile",
        "Overview",
        "Education",
        "Certifications",
    }
)

#: An explicit role header, e.g. "Role: Cloud Engineer with AI Capabilities".
_ROLE_RE = re.compile(r"^Role\s*:\s*(?P<role>\S.*)$", re.IGNORECASE)

#: A bare role line, e.g. "Dot Net Developer" under a project header.
_BARE_ROLE_RE = re.compile(r"^[A-Z][\w./+#&' \-]{2,38}$")

#: "Project: TruStage, USA   Jul 2025- Till Date"
_PROJECT_PREFIX_RE = re.compile(r"^Project\s*:\s*(?P<rest>\S.*)$", re.IGNORECASE)

#: "District Marketplace: (Danske Bank, India)   Aug 2022 to Jan 2024"
_PROJECT_PAREN_RE = re.compile(
    r"^(?P<name>[^:()]{2,60}?)\s*:\s*\((?P<client>[^)]*)\)"
)

#: Month names, tolerant of the spaces the PDF layout engine injects
#: ("Ju n 20 19" is "Jun 2019").
_MONTH = (
    r"(?:J\s*a\s*n|F\s*e\s*b|M\s*a\s*r|A\s*p\s*r|M\s*a\s*y|J\s*u\s*n"
    r"|J\s*u\s*l|A\s*u\s*g|S\s*e\s*p|O\s*c\s*t|N\s*o\s*v|D\s*e\s*c)"
)

#: A year, tolerant of injected spaces ("20 19" is "2019").
_YEAR = r"\d[\d ]{0,5}"

#: A date range such as "Jul 2025- Till Date" or "Aug 2022 to Jan 2024".
_PERIOD_RE = re.compile(
    rf"{_MONTH}\s*{_YEAR}\s*(?:[-–—]|to)\s*(?:{_MONTH}\s*)?(?:{_YEAR})?",
    re.IGNORECASE,
)

#: Lines that look like headings but carry no signal.
_NOISE_LINE_RE = re.compile(
    r"^(?:page\s+\d+\s*of\s*\d+|\d+|[ivxlcdm]+|curriculum vitae|resume)$",
    re.IGNORECASE,
)

#: Separators (rules, underlines) dropped from the text.
_SEPARATOR_RE = re.compile(r"^[\s_=~|\-–—]+$")

#: Characters that close a sentence, and therefore end a logical block.
_SENTENCE_END = ".!?"

# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Chunk:
    """A retrievable unit of text plus the metadata used to index it."""

    chunk_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_text": self.chunk_text, "metadata": self.metadata}

    def __str__(self) -> str:  # pragma: no cover - display helper
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass(slots=True)
class Block:
    """A logical unit of text, re-assembled from the PDF's physical lines."""

    kind: str
    text: str
    page: int = 1
    end_page: int = 1
    label: str = ""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Normalise a raw extracted string: strip nulls, collapse blank runs."""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_bullet(line: str) -> str:
    return _BULLET_RE.sub("", line).strip()


def _is_section_heading(line: str) -> bool:
    """True if the line looks like an ALL-CAPS section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return False
    if _NOISE_LINE_RE.match(stripped):
        return False
    if _SEPARATOR_RE.match(stripped):
        return False
    if not _SECTION_RE.match(stripped):
        return False
    letters = [ch for ch in stripped if ch.isalpha()]
    if len(letters) < 4:
        return False
    if stripped.rstrip(":").upper() in _SECTION_STOPWORDS:
        return False
    return True


def _is_bare_role(line: str) -> bool:
    """True if a short, title-case line reads as a bare job title."""
    stripped = line.strip()
    if not _BARE_ROLE_RE.match(stripped):
        return False
    if stripped.endswith(tuple(_SENTENCE_END)) or stripped.endswith(","):
        return False
    if len(stripped.split()) > 6:
        return False
    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return uppercase_ratio < 0.6


def _ends_sentence(line: str) -> bool:
    """True if the line ends on a sentence terminator."""
    stripped = line.rstrip().strip()
    return bool(stripped) and stripped[-1] in _SENTENCE_END


def _split_period(text: str) -> tuple[str, str]:
    """Split "TruStage, USA   Jul 2025- Till Date" into head and period."""
    match = _PERIOD_RE.search(text)
    if not match:
        return text.strip(), ""
    return text[: match.start()].strip(), text[match.start() :].strip()


def _parse_project_header(text: str) -> dict[str, str]:
    """Pull the project name, client and period out of a project header line."""
    fields: dict[str, str] = {}

    remainder = text.strip()
    prefix = _PROJECT_PREFIX_RE.match(remainder)
    if prefix:
        remainder = prefix.group("rest").strip()

    head, period = _split_period(remainder)
    if period:
        fields["period"] = period

    client_match = re.search(r"\(([^)]*)\)", head)
    if client_match:
        client = client_match.group(1).strip().rstrip(".").strip()
        if client:
            fields["client"] = client
        head = head[: client_match.start()].strip()
    else:
        parts = [part.strip() for part in head.split(",") if part.strip()]
        if parts:
            fields["name"] = parts[0]
            if len(parts) > 1:
                fields["client"] = parts[1]
        return fields

    name = head.rstrip(":").strip()
    if name:
        fields["name"] = name
    return fields


def _looks_like_project_header(line: str) -> bool:
    """True if the line opens a new project entry."""
    stripped = line.strip()
    if _PROJECT_PREFIX_RE.match(stripped):
        # "Project: X" only counts when it also carries a date or a client.
        return bool(_PERIOD_RE.search(stripped) or "(" in stripped)
    if _PROJECT_PAREN_RE.match(stripped):
        return True
    return False


# --------------------------------------------------------------------------
# The chunker
# --------------------------------------------------------------------------


class PDFChunker:
    """Split a PDF into structured, metadata-carrying chunks.

    Usage::

        chunker = PDFChunker("datasource/resume.pdf")
        chunks = chunker.chunk()
    """

    def __init__(
        self,
        pdf_path: str | os.PathLike[str] | None = None,
        *,
        max_chunk_chars: int = MAX_CHUNK_CHARS,
        min_useful_chars: int = MIN_USEFUL_CHARS,
        split_overlap_chars: int = SPLIT_OVERLAP_CHARS,
        section_rules: dict[str, dict[str, re.Pattern[str]]] | None = None,
    ) -> None:
        if pdf_path is None:
            pdf_path = os.getenv("SOURCE_PDF")
        if not pdf_path:
            raise ValueError(
                "No PDF path given and SOURCE_PDF is not set in the environment."
            )

        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        self.max_chunk_chars = max_chunk_chars
        self.min_useful_chars = min_useful_chars
        self.split_overlap_chars = split_overlap_chars
        self.section_rules = section_rules or {}

    # -- reading ----------------------------------------------------------

    def read_pages(self) -> list[tuple[int, str]]:
        """Return (page_number, text) for every page of the PDF."""
        reader = PdfReader(str(self.pdf_path))
        pages: list[tuple[int, str]] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append((index, _clean_text(text)))
        return pages

    def read_lines(self) -> list[tuple[int, str]]:
        """Return (page_number, line) for every meaningful line."""
        lines: list[tuple[int, str]] = []
        for page_number, text in self.read_pages():
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or _NOISE_LINE_RE.match(stripped):
                    continue
                lines.append((page_number, stripped))
        return lines

    # -- structure --------------------------------------------------------

    def build_blocks(self) -> list[Block]:
        """Re-assemble the physical lines into logical blocks.

        The PDF layout engine hard-wraps body text, so a single sentence
        arrives as several lines. A line continues the previous block when
        the previous line did not end on a sentence terminator and the line
        itself does not open a new block.
        """
        blocks: list[Block] = []
        pending: list[str] = []
        pending_kind = "para"
        pending_page = 1
        pending_end_page = 1
        previous_kind = ""
        continues = False
        active_label = ""

        def flush() -> None:
            nonlocal pending, pending_kind, pending_page, pending_end_page
            if pending:
                blocks.append(
                    Block(
                        kind=pending_kind,
                        text=" ".join(pending).strip(),
                        page=pending_page,
                        end_page=pending_end_page,
                        label=active_label,
                    )
                )
            pending = []

        for page_number, line in self.read_lines():
            kind = self._classify(line, previous_kind)

            if kind in ("heading", "project"):
                flush()
                blocks.append(Block(kind=kind, text=line, page=page_number))
                previous_kind = kind
                continues = False
                active_label = ""
                continue

            if kind == "role":
                flush()
                blocks.append(Block(kind=kind, text=line, page=page_number))
                previous_kind = kind
                continues = False
                active_label = ""
                continue

            if kind == "bullet":
                flush()
                pending_kind = "bullet"
                pending_page = page_number
                pending_end_page = page_number
                pending = [_strip_bullet(line)]
                previous_kind = kind
                continues = not _ends_sentence(line)
                continue

            if kind == "label":
                # A label always opens its own block. It must not be absorbed
                # into the preceding one, whose bullet may lack a trailing
                # period and so look like an unfinished sentence.
                flush()
                pending_kind = "label"
                pending_page = page_number
                pending_end_page = page_number
                pending = [line]
                active_label = self._scoping_label(line)
                previous_kind = kind
                continues = not _ends_sentence(line)
                continue

            # Paragraph lines may be continued by the next one.
            if pending and continues and pending_kind in ("para", "bullet"):
                pending.append(line)
                pending_end_page = page_number
            else:
                flush()
                pending_kind = kind
                pending_page = page_number
                pending_end_page = page_number
                pending = [line]

            previous_kind = kind
            continues = not _ends_sentence(line)

        flush()
        return blocks

    def _classify(self, line: str, previous_kind: str) -> str:
        """Classify one physical line."""
        if _is_section_heading(line):
            return "heading"
        if _ROLE_RE.match(line):
            return "role"
        if _looks_like_project_header(line):
            return "project"
        if previous_kind == "project" and _is_bare_role(line):
            return "role"
        if _BULLET_RE.match(line):
            return "bullet"
        if self._is_label(line):
            return "label"
        return "para"

    @staticmethod
    def _is_label(line: str) -> bool:
        """True if the line opens a scoping label.

        A scoping label always opens its own block, whether its value is on
        the same line - "Environment: .NET 8, ..." - or on the next one -
        "Environment:". It must not be absorbed into the preceding block,
        whose bullet may lack a trailing period and so look unfinished.
        """
        match = _LABEL_RE.match(line)
        if not match:
            return False
        return match.group("label").strip() in _SCOPING_LABELS

    @staticmethod
    def _scoping_label(line: str) -> str:
        """The scoping label this line opens, or "" if it opens none.

        Only the labels in ``_SCOPING_LABELS`` scope the content beneath
        them, so a contact line such as "Email: gladson.bits@gmail.com"
        does not turn the rest of the section into email text.
        """
        match = _LABEL_RE.match(line)
        if not match:
            return ""
        label = match.group("label").strip()
        return label if label in _SCOPING_LABELS else ""

    # -- chunking ---------------------------------------------------------

    def chunk(self) -> list[Chunk]:
        """Produce the final list of chunks for the document."""
        chunks: list[Chunk] = []
        section_title = "HEADER"
        section_index = 0
        chunk_index = 0
        section_chunk_index = 0
        context: dict[str, str] = {}

        for block in self.build_blocks():
            if block.kind == "heading":
                section_title = block.text.rstrip(":").strip() or "HEADER"
                section_index += 1
                section_chunk_index = 0
                context = {}
                continue

            if block.kind == "project":
                context = _parse_project_header(block.text)
                continue

            if block.kind == "role":
                match = _ROLE_RE.match(block.text)
                if match:
                    context = dict(context)
                    context["role"] = match.group("role").strip()
                else:
                    context = dict(context)
                    context["role"] = block.text.strip()
                continue

            if not block.text.strip():
                continue

            for piece in self._split_by_size(block.text):
                if len(piece) < self.min_useful_chars:
                    continue
                metadata: dict[str, Any] = {
                    "section": section_title,
                    "source": self.pdf_path.name,
                    "page": block.page,
                    "end_page": block.end_page,
                    "section_index": section_index,
                    "chunk_index": chunk_index,
                    "section_chunk_index": section_chunk_index,
                }
                metadata.update(self._extra_metadata(section_title, block, context))
                chunks.append(Chunk(chunk_text=piece, metadata=metadata))
                chunk_index += 1
                section_chunk_index += 1

        return chunks

    def _extra_metadata(
        self, section_title: str, block: Block, context: dict[str, str]
    ) -> dict[str, Any]:
        """Attach the project / role context, plus any section rules."""
        extra: dict[str, Any] = {}

        for key in ("name", "client", "period"):
            value = context.get(key)
            if value:
                extra["project" if key == "name" else key] = value

        role = context.get("role")
        if role:
            extra["role"] = role

        if block.label:
            extra["label"] = block.label

        rules = self.section_rules.get(section_title.upper(), {})
        if rules:
            haystack = block.text
            for field_name, pattern in rules.items():
                match = pattern.search(haystack)
                if match:
                    value = match.group(1) if match.groups() else match.group(0)
                    extra.setdefault(field_name, value.strip())
        return extra

    def _split_by_size(self, text: str) -> list[str]:
        """Split an oversized block into overlapping pieces."""
        text = text.strip()
        if len(text) <= self.max_chunk_chars:
            return [text] if text else []

        pieces: list[str] = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.max_chunk_chars, length)
            if end < length:
                window = text[start:end]
                boundary = max(window.rfind(". "), window.rfind("; "))
                if boundary > self.max_chunk_chars * 0.6:
                    end = start + boundary + 1

            pieces.append(text[start:end].strip())

            if end >= length:
                break
            start = max(end - self.split_overlap_chars, start + 1)

        return [piece for piece in pieces if piece]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:  # pragma: no cover - manual entry point
    chunker = PDFChunker()
    for chunk in chunker.chunk():
        print(json.dumps(chunk.to_dict(), indent=2, ensure_ascii=False))
        print("-" * 60)


if __name__ == "__main__":  # pragma: no cover
    main()
