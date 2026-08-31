"""Multi-format document loader and intelligent chunker.

Parses local files (PDF, DOCX, TXT, MD, CSV, XLSX) into :class:`Chunk` objects
ready to be ingested into the graph by :class:`~ai_memory.dataset_builder.DatasetBuilder`.

Supported formats
-----------------
* ``.pdf``                — via ``pypdf`` (``pip install pypdf``)
* ``.docx`` / ``.doc``   — via ``python-docx`` (``pip install python-docx``)
* ``.txt`` / ``.md``     — built-in (no extra deps)
* ``.csv``               — built-in ``csv`` module
* ``.xlsx`` / ``.xls``   — via ``pandas`` + ``openpyxl``
  (``pip install pandas openpyxl``)

Chunking strategy
-----------------
Text is first split on double-newlines (paragraph boundaries). Paragraphs
shorter than ``chunk_size`` are merged together up to the size limit.
Paragraphs longer than the limit are split with a sliding window of
``chunk_overlap`` chars so context carries across chunk boundaries.
Chunks shorter than ``min_chunk_len`` are discarded as noise.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
@dataclass
class Chunk:
    """A single text chunk extracted from a document."""

    text: str
    chunk_index: int
    source_path: str
    doc_type: str                              # pdf / docx / txt / md / csv / excel
    page: Optional[int] = None               # PDF page number (1-based)
    section: Optional[str] = None            # DOCX heading / Excel sheet name
    row: Optional[int] = None               # CSV/Excel row number
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        snippet = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(idx={self.chunk_index}, type={self.doc_type}, "
            f"page={self.page}, section={self.section!r}, text={snippet!r}...)"
        )


# --------------------------------------------------------------------------
class DocumentLoader:
    """Parse a local document into a list of :class:`Chunk` objects.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk (default 800).
    chunk_overlap:
        Number of characters to carry over between adjacent chunks when a
        paragraph must be split (default 100).
    min_chunk_len:
        Chunks shorter than this are discarded as noise (default 60).
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        min_chunk_len: int = 60,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

    # ------------------------------------------------------------------ public
    def load(self, path: str) -> List[Chunk]:
        """Parse ``path`` and return chunks. Format is inferred from the suffix."""
        ext = Path(path).suffix.lower()
        dispatch = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".doc": self._load_docx,
            ".txt": self._load_text,
            ".md": self._load_text,
            ".markdown": self._load_text,
            ".csv": self._load_csv,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
        }
        handler = dispatch.get(ext, self._load_text)
        return handler(path)

    # --------------------------------------------------------------- loaders
    def _load_pdf(self, path: str) -> List[Chunk]:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # type: ignore
            except ImportError:
                raise ImportError(
                    "PDF support requires pypdf: pip install pypdf"
                )
        reader = PdfReader(path)
        all_chunks: List[Chunk] = []
        global_idx = 0
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            for c in self._chunk_text(text, path, "pdf", page=page_num):
                c.chunk_index = global_idx
                global_idx += 1
                all_chunks.append(c)
        return all_chunks

    def _load_docx(self, path: str) -> List[Chunk]:
        try:
            from docx import Document  # type: ignore
        except ImportError:
            raise ImportError(
                "DOCX support requires python-docx: pip install python-docx"
            )
        doc = Document(path)
        all_chunks: List[Chunk] = []
        global_idx = 0
        current_section: Optional[str] = None
        buffer: List[str] = []

        def flush():
            nonlocal global_idx
            text = "\n\n".join(buffer).strip()
            if not text:
                return
            for c in self._chunk_text(text, path, "docx", section=current_section):
                c.chunk_index = global_idx
                global_idx += 1
                all_chunks.append(c)

        for para in doc.paragraphs:
            if para.style.name.startswith("Heading"):
                flush()
                buffer.clear()
                current_section = para.text.strip() or current_section
            else:
                stripped = para.text.strip()
                if stripped:
                    buffer.append(stripped)

        flush()
        return all_chunks

    def _load_text(self, path: str) -> List[Chunk]:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        doc_type = Path(path).suffix.lstrip(".").lower() or "txt"
        return self._chunk_text(text, path, doc_type)

    def _load_csv(self, path: str) -> List[Chunk]:
        """Load a CSV file, yielding one :class:`Chunk` per data row.

        CSV rows are structured data — each row is always a valid chunk
        regardless of its character length, so ``min_chunk_len`` is bypassed.
        """
        chunks: List[Chunk] = []
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for row_num, row in enumerate(reader, start=2):  # row 1 = header
                text = "  |  ".join(
                    f"{k}: {v}" for k, v in row.items() if v and str(v).strip()
                )
                if not text.strip():   # skip completely empty rows
                    continue
                chunks.append(
                    Chunk(
                        text=text,
                        chunk_index=row_num - 2,
                        source_path=path,
                        doc_type="csv",
                        row=row_num,
                    )
                )
        return chunks

    def _load_excel(self, path: str) -> List[Chunk]:
        try:
            import pandas as pd  # type: ignore
        except ImportError:
            raise ImportError(
                "Excel support requires pandas + openpyxl: "
                "pip install pandas openpyxl"
            )
        chunks: List[Chunk] = []
        global_idx = 0
        xf = pd.ExcelFile(path)
        for sheet in xf.sheet_names:
            df = xf.parse(sheet).fillna("")
            for i, row in df.iterrows():
                text = f"[Sheet: {sheet}]  " + "  |  ".join(
                    f"{col}: {val}" for col, val in row.items() if str(val).strip()
                )
                if len(text) >= self.min_chunk_len:
                    chunks.append(
                        Chunk(
                            text=text,
                            chunk_index=global_idx,
                            source_path=path,
                            doc_type="excel",
                            row=int(i),
                            section=sheet,
                        )
                    )
                    global_idx += 1
        return chunks

    # ------------------------------------------------------------ chunking
    def _chunk_text(
        self,
        text: str,
        source_path: str,
        doc_type: str,
        page: Optional[int] = None,
        section: Optional[str] = None,
    ) -> List[Chunk]:
        """Split ``text`` into :class:`Chunk` objects respecting ``chunk_size``."""
        # Normalise whitespace while keeping paragraph breaks.
        text = re.sub(r"\n{3,}", "\n\n", text)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        raw_chunks: List[str] = []
        current = ""
        for para in paragraphs:
            if len(para) > self.chunk_size:
                # Flush current buffer first.
                if len(current) >= self.min_chunk_len:
                    raw_chunks.append(current.strip())
                    current = ""
                # Slide over the large paragraph.
                raw_chunks.extend(self._slide(para))
            elif len(current) + len(para) + 2 <= self.chunk_size:
                current = (current + "\n\n" + para).strip() if current else para
            else:
                if len(current) >= self.min_chunk_len:
                    raw_chunks.append(current.strip())
                current = para

        if current and len(current) >= self.min_chunk_len:
            raw_chunks.append(current.strip())

        return [
            Chunk(
                text=c,
                chunk_index=i,
                source_path=source_path,
                doc_type=doc_type,
                page=page,
                section=section,
            )
            for i, c in enumerate(raw_chunks)
        ]

    def _slide(self, text: str) -> List[str]:
        """Sliding-window split for paragraphs that exceed ``chunk_size``."""
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_len:
                chunks.append(chunk)
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        return chunks


__all__ = ["Chunk", "DocumentLoader"]
