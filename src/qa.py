from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .extractors import split_sentences
from .models import AnalysisResult, Citation, QAResponse


@dataclass
class QAChunk:
    text: str
    citation: Citation


@dataclass
class QAIndex:
    vectorizer: TfidfVectorizer
    matrix: np.ndarray
    chunks: List[QAChunk]


def build_qa_index(analysis: AnalysisResult) -> QAIndex:
    block_map = {block.id: block for block in analysis.blocks}
    section_map = {section.id: section for section in analysis.sections}
    chunks: List[QAChunk] = []

    for section in analysis.sections:
        for block_id in section.block_ids:
            block = block_map.get(block_id)
            if not block:
                continue
            for sentence in split_sentences(block.text):
                if len(sentence.split()) < 4:
                    continue
                chunks.append(
                    QAChunk(
                        text=sentence,
                        citation=Citation(
                            section_id=section.id,
                            section_title=section.title,
                            block_id=block.id,
                            snippet=sentence,
                            score=1.0,
                        ),
                    )
                )

    if not chunks:
        # Fallback to section-level chunks
        for section in analysis.sections:
            if not section.text.strip():
                continue
            chunks.append(
                QAChunk(
                    text=section.text,
                    citation=Citation(
                        section_id=section.id,
                        section_title=section.title,
                        block_id=section.block_ids[0] if section.block_ids else "",
                        snippet=section.text[:220],
                        score=1.0,
                    ),
                )
            )

    corpus = [chunk.text for chunk in chunks] or [" "]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    return QAIndex(vectorizer=vectorizer, matrix=matrix, chunks=chunks)


def _search(query: str, qa_index: QAIndex) -> List[Tuple[float, QAChunk]]:
    query_vec = qa_index.vectorizer.transform([query])
    sims = cosine_similarity(query_vec, qa_index.matrix).flatten()
    ranked_idx = np.argsort(sims)[::-1]
    ranked: List[Tuple[float, QAChunk]] = []
    for idx in ranked_idx[:5]:
        score = float(sims[idx])
        ranked.append((score, qa_index.chunks[int(idx)]))
    return ranked


def _meaningful_tokens(text: str) -> set[str]:
    generic = {
        "this",
        "that",
        "there",
        "where",
        "when",
        "what",
        "which",
        "does",
        "document",
        "claim",
        "claims",
        "exist",
        "exists",
        "about",
    }
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]+", text)
        if len(token) >= 4
    }
    return {t for t in tokens if t not in ENGLISH_STOP_WORDS and t not in generic}


def answer_question(query: str, analysis: AnalysisResult, threshold: float = 0.18) -> QAResponse:
    query = (query or "").strip()
    if not query:
        return QAResponse(answer="", supported=False, score=0.0, citations=[], nearest_sections=[])

    qa_index = build_qa_index(analysis)
    ranked = _search(query, qa_index)
    if not ranked:
        return QAResponse(
            answer="Not found in document",
            supported=False,
            score=0.0,
            citations=[],
            nearest_sections=[],
        )

    best_score, best_chunk = ranked[0]
    nearest = [chunk.citation for _, chunk in ranked[:3]]
    query_tokens = _meaningful_tokens(query)
    chunk_tokens = _meaningful_tokens(best_chunk.text)
    overlap = query_tokens.intersection(chunk_tokens)

    if best_score < threshold or (query_tokens and not overlap):
        return QAResponse(
            answer="Not found in document",
            supported=False,
            score=best_score,
            citations=[],
            nearest_sections=nearest,
        )

    supporting = [chunk for score, chunk in ranked[:2] if score >= threshold * 0.8]
    answer_text = " ".join(chunk.text for chunk in supporting)
    answer_text = answer_text.strip() or best_chunk.text

    return QAResponse(
        answer=answer_text,
        supported=True,
        score=best_score,
        citations=[chunk.citation for chunk in supporting],
        nearest_sections=nearest,
    )
