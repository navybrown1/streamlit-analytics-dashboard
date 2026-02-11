from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import pdfplumber
from docx import Document

from .models import DocumentBlock

DOCX_MIME_MAGIC = b"PK\x03\x04"
PDF_MAGIC = b"%PDF"


def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_file_type(filename: Optional[str], data: bytes) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext == ".docx":
        return "docx"
    if ext == ".pdf":
        return "pdf"
    if ext in {".txt", ".md", ".rst"}:
        return "txt"

    if data.startswith(PDF_MAGIC):
        return "pdf"
    if data.startswith(DOCX_MIME_MAGIC):
        try:
            with zipfile.ZipFile(io.BytesIO(data)):
                return "docx"
        except zipfile.BadZipFile:
            pass

    try:
        data.decode("utf-8")
        return "txt"
    except UnicodeDecodeError:
        return "unknown"


def _safe_snippet(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _build_block(
    *,
    idx: int,
    text: str,
    source: str,
    page: Optional[int],
    style: Optional[str],
    bold_ratio: float,
    cursor: int,
) -> DocumentBlock:
    clean = re.sub(r"\s+", " ", text).strip()
    start_char = cursor
    end_char = start_char + len(clean)
    return DocumentBlock(
        id=f"b{idx}",
        index=idx,
        text=clean,
        source=source,
        page=page,
        style=style,
        bold_ratio=round(float(bold_ratio), 4),
        start_char=start_char,
        end_char=end_char,
    )


def _extract_docx_xml_paragraphs(data: bytes) -> List[str]:
    paragraphs: List[str] = []
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            xml = zf.read("word/document.xml")
    except Exception:
        return paragraphs

    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return paragraphs

    for para in root.findall(".//w:body/w:p", ns):
        text = "".join((t.text or "") for t in para.findall(".//w:t", ns)).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _extract_common_metadata(first_lines: List[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    date_matcher = re.compile(
        r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{2,4})\b",
        re.IGNORECASE,
    )
    for line in first_lines:
        if "prof" in line.lower() or "professor" in line.lower():
            metadata.setdefault("author", line)
        if "college" in line.lower() or "university" in line.lower():
            metadata.setdefault("organization", line)
        m = date_matcher.search(line)
        if m:
            metadata.setdefault("date", m.group(0))
    return metadata


def parse_docx(data: bytes) -> Tuple[List[DocumentBlock], Dict[str, Any], List[str]]:
    blocks: List[DocumentBlock] = []
    warnings: List[str] = []
    cursor = 0

    doc = Document(io.BytesIO(data))
    idx = 1
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        char_count = sum(len(run.text or "") for run in para.runs) or len(text)
        bold_chars = sum(len(run.text or "") for run in para.runs if bool(run.bold))
        bold_ratio = (bold_chars / char_count) if char_count else 0.0
        style_name = getattr(getattr(para, "style", None), "name", None)
        block = _build_block(
            idx=idx,
            text=text,
            source="docx",
            page=None,
            style=style_name,
            bold_ratio=bold_ratio,
            cursor=cursor,
        )
        blocks.append(block)
        cursor = block.end_char + 1
        idx += 1

    if len(blocks) < 4:
        xml_paragraphs = _extract_docx_xml_paragraphs(data)
        if xml_paragraphs:
            warnings.append(
                "DOCX paragraph styles were sparse; fallback XML parsing was used for additional coverage."
            )
        for text in xml_paragraphs:
            if any(existing.text == text for existing in blocks):
                continue
            block = _build_block(
                idx=idx,
                text=text,
                source="docx-xml",
                page=None,
                style=None,
                bold_ratio=0.0,
                cursor=cursor,
            )
            blocks.append(block)
            cursor = block.end_char + 1
            idx += 1

    metadata = _extract_common_metadata([block.text for block in blocks[:8]])
    return blocks, metadata, warnings


def parse_pdf(data: bytes) -> Tuple[List[DocumentBlock], Dict[str, Any], List[str]]:
    blocks: List[DocumentBlock] = []
    warnings: List[str] = []
    cursor = 0
    idx = 1

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            lines = [line.strip() for line in page_text.splitlines() if line.strip()]
            if not lines:
                continue
            for line in lines:
                block = _build_block(
                    idx=idx,
                    text=line,
                    source=f"pdf-page-{page_no}",
                    page=page_no,
                    style=None,
                    bold_ratio=0.0,
                    cursor=cursor,
                )
                blocks.append(block)
                cursor = block.end_char + 1
                idx += 1

    if not blocks:
        warnings.append("No extractable text found in PDF.")

    metadata = _extract_common_metadata([block.text for block in blocks[:10]])
    metadata["pages"] = max((block.page or 0) for block in blocks) if blocks else 0
    return blocks, metadata, warnings


def parse_txt(data: bytes) -> Tuple[List[DocumentBlock], Dict[str, Any], List[str]]:
    warnings: List[str] = []
    blocks: List[DocumentBlock] = []
    cursor = 0
    idx = 1
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
        warnings.append("Input text decoded using latin-1 fallback.")

    lines = [line.strip() for line in text.splitlines()]
    buffer: List[str] = []
    for line in lines:
        if not line:
            if buffer:
                chunk = " ".join(buffer)
                block = _build_block(
                    idx=idx,
                    text=chunk,
                    source="txt",
                    page=None,
                    style=None,
                    bold_ratio=0.0,
                    cursor=cursor,
                )
                blocks.append(block)
                cursor = block.end_char + 1
                idx += 1
                buffer = []
            continue
        buffer.append(line)

    if buffer:
        chunk = " ".join(buffer)
        block = _build_block(
            idx=idx,
            text=chunk,
            source="txt",
            page=None,
            style=None,
            bold_ratio=0.0,
            cursor=cursor,
        )
        blocks.append(block)

    metadata = _extract_common_metadata([block.text for block in blocks[:8]])
    return blocks, metadata, warnings


def parse_document(
    *,
    file_path: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
) -> Tuple[str, str, bytes, List[DocumentBlock], Dict[str, Any], List[str]]:
    if file_bytes is None:
        if not file_path:
            raise ValueError("Either file_path or file_bytes must be provided.")
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        file_bytes = path.read_bytes()
        inferred_name = path.name
    else:
        inferred_name = filename or (Path(file_path).name if file_path else "uploaded_document")

    file_type = detect_file_type(inferred_name, file_bytes)
    warnings: List[str] = []

    if file_type == "docx":
        blocks, metadata, parse_warnings = parse_docx(file_bytes)
    elif file_type == "pdf":
        blocks, metadata, parse_warnings = parse_pdf(file_bytes)
    elif file_type == "txt":
        blocks, metadata, parse_warnings = parse_txt(file_bytes)
    else:
        raise ValueError("Unsupported file type. Please upload DOCX, PDF, or TXT.")

    warnings.extend(parse_warnings)
    if not blocks:
        warnings.append("No text blocks were extracted from the document.")

    return inferred_name, file_type, file_bytes, blocks, metadata, warnings


def make_reference_snippet(text: str) -> str:
    return _safe_snippet(text)
