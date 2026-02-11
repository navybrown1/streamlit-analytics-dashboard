from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

from .models import DocumentBlock, Section


HEADING_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:[\.)])?\s+.+")
EQUATION_HINT_RE = re.compile(r"(?:=|\{|\}|\bP\s*\()")


def _normalize_title(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def heading_score(block: DocumentBlock) -> float:
    text = block.text.strip()
    if not text:
        return 0.0

    words = text.split()
    word_count = len(words)
    score = 0.0

    if block.style and str(block.style).lower().startswith("heading"):
        score += 2.5

    number_match = HEADING_NUMBER_RE.match(text)
    if number_match:
        score += 1.8

    if text.lower() in {"contents", "table of contents"}:
        score += 2.5

    if word_count <= 8 and text.istitle():
        score += 1.0

    if word_count <= 8 and text.isupper():
        score += 1.2

    if block.bold_ratio >= 0.45 and word_count <= 10:
        score += 0.6

    if len(text) > 90:
        score -= 1.2
    elif len(text) <= 70:
        score += 0.4

    if text.endswith("."):
        score -= 0.7

    if EQUATION_HINT_RE.search(text):
        score -= 1.4

    alphabetic_words = [w for w in words if re.search(r"[A-Za-z]", w)]
    if len(alphabetic_words) < 2:
        score -= 0.6

    return score


def _detect_toc_indices(blocks: List[DocumentBlock]) -> Set[str]:
    toc_indices: Set[str] = set()
    if not blocks:
        return toc_indices

    for pos, block in enumerate(blocks):
        lower = block.text.lower().strip()
        if lower not in {"contents", "table of contents"}:
            continue
        toc_indices.add(block.id)
        for look_ahead in blocks[pos + 1 : pos + 15]:
            t = look_ahead.text.strip()
            if not t:
                continue
            if len(t) > 70:
                break
            if HEADING_NUMBER_RE.match(t) or (len(t.split()) <= 6 and t[0:1].isupper()):
                toc_indices.add(look_ahead.id)
                continue
            break
    return toc_indices


def _section_level_from_title(title: str) -> int:
    m = HEADING_NUMBER_RE.match(title)
    if not m:
        return 1
    return min(4, m.group(1).count(".") + 1)


def _finalize_section(
    *,
    section_id: int,
    title: str,
    block_list: List[DocumentBlock],
    confidence: float,
    section_type: str = "content",
) -> Section:
    text = "\n".join(block.text for block in block_list).strip()
    return Section(
        id=f"s{section_id}",
        title=_normalize_title(title) if title else f"Section {section_id}",
        level=_section_level_from_title(title),
        section_type=section_type,
        start_block=block_list[0].index,
        end_block=block_list[-1].index,
        block_ids=[block.id for block in block_list],
        text=text,
        parse_confidence=max(0.05, min(1.0, confidence)),
    )


def sectionize_blocks(blocks: List[DocumentBlock]) -> Tuple[List[Section], Dict[str, str], List[str]]:
    warnings: List[str] = []
    if not blocks:
        warnings.append("No blocks available for sectionization.")
        return [], {}, warnings

    toc_ids = _detect_toc_indices(blocks)

    sections: List[Section] = []
    block_to_section: Dict[str, str] = {}
    section_id = 1

    current_title = "Document Body"
    current_blocks: List[DocumentBlock] = []
    current_confidences: List[float] = []

    def flush_current(title: str, section_type: str = "content") -> None:
        nonlocal section_id, current_blocks, current_confidences
        if not current_blocks:
            return
        confidence = sum(current_confidences) / max(1, len(current_confidences))
        section = _finalize_section(
            section_id=section_id,
            title=title,
            block_list=current_blocks,
            confidence=confidence,
            section_type=section_type,
        )
        sections.append(section)
        for bid in section.block_ids:
            block_to_section[bid] = section.id
        section_id += 1
        current_blocks = []
        current_confidences = []

    toc_buffer: List[DocumentBlock] = []
    for block in blocks:
        if block.id in toc_ids:
            toc_buffer.append(block)
            continue

        score = heading_score(block)
        is_heading = score >= 1.6

        if is_heading:
            if current_blocks:
                flush_current(current_title)
            current_title = block.text
            current_confidences.append(min(1.0, max(0.15, score / 3.0)))
            continue

        current_blocks.append(block)
        current_confidences.append(0.65)

    flush_current(current_title)

    if toc_buffer:
        toc_section = _finalize_section(
            section_id=section_id,
            title="Contents",
            block_list=toc_buffer,
            confidence=0.9,
            section_type="toc",
        )
        sections.insert(0, toc_section)
        for bid in toc_section.block_ids:
            block_to_section[bid] = toc_section.id

    if not sections:
        section = _finalize_section(
            section_id=1,
            title="Document Body",
            block_list=blocks,
            confidence=0.25,
            section_type="fallback",
        )
        sections = [section]
        for bid in section.block_ids:
            block_to_section[bid] = section.id
        warnings.append("No heading-like boundaries detected; used fallback single-section parsing.")

    return sections, block_to_section, warnings
