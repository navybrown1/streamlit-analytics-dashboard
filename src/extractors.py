from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import (
    Citation,
    ConfidenceFlag,
    DocumentBlock,
    Entity,
    InsightPoint,
    RelationshipEdge,
    RequirementItem,
    Section,
    Takeaway,
)
from .parsers import make_reference_snippet


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{2,4})\b",
    re.IGNORECASE,
)

PEOPLE_RE = re.compile(r"\b(?:Prof\.?|Professor)\s+[A-Z][A-Za-z\.-]+(?:\s+[A-Z][A-Za-z\.-]+)*")
ORG_RE = re.compile(r"\b[A-Z][A-Za-z&\s]{1,60}(?:College|University|Department|Institute|School|Bank)\b")
METRIC_RE = re.compile(r"\b\d+\s*/\s*\d+\b|\b\d+(?:\.\d+)?%\b")
PROB_EXPR_RE = re.compile(r"\bP\([^\)]{1,40}\)")
SET_RE = re.compile(r"\b[A-Z]c?\s*=\s*\{[^\}]{1,80}\}")

CONCEPT_TERMS = [
    "mutually exclusive",
    "union",
    "intersection",
    "conditional probability",
    "independence",
    "bayes",
    "complement",
    "compound event",
    "elementary outcomes",
    "addition rule",
]

TOOL_TERMS = ["OR gate", "XOR gate", "logic gate", "computer science"]
DELIVERABLE_TERMS = ["lecture note", "assignment", "checklist", "rubric", "dashboard"]

REQ_KEYWORDS = [
    "must",
    "should",
    "may",
    "required",
    "rule",
    "defined as",
    "definition",
    "if and only if",
    "if",
    "then",
]

HEDGE_WORDS = ["may", "might", "could", "possibly", "perhaps"]
UNCERTAINTY_WORDS = ["not", "unclear", "unknown", "?"]
CONTRAST_WORDS = ["however", "but", "although", "yet"]


def split_sentences(text: str) -> List[str]:
    raw = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if raw:
        return raw
    return [text.strip()] if text.strip() else []


def parse_fuzzy_date(value: str) -> Optional[datetime]:
    value = value.strip()
    formats = [
        "%m-%d-%Y",
        "%m/%d/%Y",
        "%m-%d-%y",
        "%m/%d/%y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _confidence_from_sentence(sentence: str) -> str:
    lower = sentence.lower()
    if any(k in lower for k in ["defined as", "rule", "must", "should", "may", "="]):
        return "explicit"
    return "inferred"


def _priority_from_sentence(sentence: str) -> str:
    lower = sentence.lower()
    if "must" in lower or "required" in lower:
        return "High"
    if "should" in lower:
        return "Medium"
    if "may" in lower:
        return "Low"
    return "Medium"


def _verification_method(sentence: str) -> str:
    lower = sentence.lower()
    if "defined as" in lower or "definition" in lower:
        return "Verify the definition text matches the cited source wording."
    if "must" in lower or "required" in lower:
        return "Test mandatory condition against at least two edge-case examples from the document context."
    if "p(" in lower or "=" in sentence:
        return "Recompute with a sample value and confirm formula behavior aligns with cited statement."
    return "Cross-check the cited section and ensure this statement is consistently supported."


def _make_citation(section: Section, block: DocumentBlock, snippet: str, score: float = 1.0) -> Citation:
    return Citation(
        section_id=section.id,
        section_title=section.title,
        block_id=block.id,
        snippet=make_reference_snippet(snippet),
        score=float(score),
    )


def _section_by_id(sections: List[Section]) -> Dict[str, Section]:
    return {section.id: section for section in sections}


def extract_entities(
    sections: List[Section],
    blocks: List[DocumentBlock],
    block_to_section: Dict[str, str],
) -> Tuple[List[Entity], Dict[str, Dict[str, int]]]:
    section_map = _section_by_id(sections)
    registry: Dict[Tuple[str, str], Dict[str, object]] = {}
    section_entity_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def register(label: str, entity_type: str, block: DocumentBlock, section: Section) -> None:
        clean = re.sub(r"\s+", " ", label).strip(" ,.;:")
        if not clean:
            return
        key = (entity_type, clean.lower())
        ref = f"{section.id}:{block.id}"
        citation = _make_citation(section, block, clean)

        if key not in registry:
            registry[key] = {
                "label": clean,
                "type": entity_type,
                "mentions": 0,
                "first": block.index,
                "last": block.index,
                "first_ref": ref,
                "last_ref": ref,
                "sections": set(),
                "citations": [],
            }
        rec = registry[key]
        rec["mentions"] = int(rec["mentions"]) + 1
        rec["last"] = block.index
        rec["last_ref"] = ref
        rec["sections"].add(section.title)
        citations: List[Citation] = rec["citations"]  # type: ignore[assignment]
        if len(citations) < 8:
            citations.append(citation)

    for block in blocks:
        section_id = block_to_section.get(block.id)
        if not section_id:
            continue
        section = section_map[section_id]
        text = block.text
        lower = text.lower()

        for match in PEOPLE_RE.findall(text):
            register(match, "person", block, section)
        for match in ORG_RE.findall(text):
            register(match, "org", block, section)
        for match in DATE_RE.findall(text):
            register(match, "date", block, section)
        for match in METRIC_RE.findall(text):
            register(match, "metric", block, section)
        for match in PROB_EXPR_RE.findall(text):
            register(match, "concept", block, section)
        for match in SET_RE.findall(text):
            register(match, "constraint", block, section)

        for term in CONCEPT_TERMS:
            if term in lower:
                register(term.title(), "concept", block, section)
        for term in TOOL_TERMS:
            if term.lower() in lower:
                register(term, "tool", block, section)
        for term in DELIVERABLE_TERMS:
            if term in lower:
                register(term.title(), "deliverable", block, section)

        if any(k in lower for k in ["must", "should", "may", "required"]):
            register("Requirement Signal", "requirement", block, section)
        if any(k in lower for k in ["constraint", "cannot", "only when", "if and only if"]):
            register("Constraint Signal", "constraint", block, section)

    entities: List[Entity] = []
    sorted_items = sorted(registry.items(), key=lambda kv: (-int(kv[1]["mentions"]), kv[1]["label"]))
    for idx, ((etype, normalized), rec) in enumerate(sorted_items, start=1):
        entity = Entity(
            id=f"ent_{idx}",
            label=str(rec["label"]),
            normalized=normalized,
            type=etype,  # type: ignore[arg-type]
            mentions_count=int(rec["mentions"]),
            first_occurrence=str(rec["first_ref"]),
            last_occurrence=str(rec["last_ref"]),
            source_sections=sorted(list(rec["sections"])),
            citations=rec["citations"],  # type: ignore[arg-type]
        )
        entities.append(entity)
        for citation in entity.citations:
            section_entity_counts[citation.section_id][entity.id] += 1

    return entities, section_entity_counts


def extract_requirements(
    sections: List[Section],
    blocks: List[DocumentBlock],
    block_to_section: Dict[str, str],
) -> List[RequirementItem]:
    section_map = _section_by_id(sections)
    requirements: List[RequirementItem] = []
    req_id = 1

    for block in blocks:
        section_id = block_to_section.get(block.id)
        if not section_id:
            continue
        section = section_map[section_id]
        for sentence in split_sentences(block.text):
            lower = sentence.lower()
            if len(sentence.split()) < 5:
                continue
            if not any(keyword in lower for keyword in REQ_KEYWORDS):
                continue

            if "if" in lower and "then" not in lower and not any(k in lower for k in ["must", "should", "may"]):
                # Avoid weak conditionals unless clearly normative.
                continue

            citation = _make_citation(section, block, sentence)
            requirements.append(
                RequirementItem(
                    id=f"req_{req_id}",
                    statement=sentence,
                    priority=_priority_from_sentence(sentence),
                    rationale=f"Detected from source statement in {section.title}.",
                    verification_method=_verification_method(sentence),
                    section_id=section.id,
                    section_title=section.title,
                    confidence=_confidence_from_sentence(sentence),
                    citations=[citation],
                )
            )
            req_id += 1

    return requirements


def build_relationships(
    sections: List[Section],
    entities: List[Entity],
    requirements: List[RequirementItem],
    section_entity_counts: Dict[str, Dict[str, int]],
) -> List[RelationshipEdge]:
    edges: List[RelationshipEdge] = []
    entity_by_id = {entity.id: entity for entity in entities}

    edge_id = 1
    for section in sections:
        for entity_id, count in section_entity_counts.get(section.id, {}).items():
            entity = entity_by_id.get(entity_id)
            if not entity:
                continue
            citation = entity.citations[0] if entity.citations else None
            edges.append(
                RelationshipEdge(
                    id=f"rel_{edge_id}",
                    source=section.id,
                    target=entity.id,
                    relation="section_contains_entity",
                    weight=float(count),
                    citations=[citation] if citation else [],
                )
            )
            edge_id += 1

    for section in sections:
        entity_ids = list(section_entity_counts.get(section.id, {}).keys())
        for left, right in itertools.combinations(sorted(entity_ids), 2):
            left_count = section_entity_counts[section.id][left]
            right_count = section_entity_counts[section.id][right]
            weight = float(min(left_count, right_count))
            if weight <= 0:
                continue
            edges.append(
                RelationshipEdge(
                    id=f"rel_{edge_id}",
                    source=left,
                    target=right,
                    relation="entity_cooccurs_entity",
                    weight=weight,
                    citations=[],
                )
            )
            edge_id += 1

    for requirement in requirements:
        req_lower = requirement.statement.lower()
        for entity in entities:
            if entity.label.lower() in req_lower:
                edges.append(
                    RelationshipEdge(
                        id=f"rel_{edge_id}",
                        source=requirement.id,
                        target=entity.id,
                        relation="requirement_refers_entity",
                        weight=1.0,
                        citations=requirement.citations[:1],
                    )
                )
                edge_id += 1

    return edges


def compute_risk_and_ambiguity(sections: List[Section]) -> None:
    for section in sections:
        text = section.text.lower()
        hedge_hits = sum(text.count(word) for word in HEDGE_WORDS)
        uncertain_hits = sum(text.count(word) for word in UNCERTAINTY_WORDS)
        contrast_hits = sum(text.count(word) for word in CONTRAST_WORDS)
        parse_penalty = max(0.0, 1.0 - section.parse_confidence)

        section.ambiguity_score = round(hedge_hits + (0.5 * uncertain_hits) + parse_penalty, 3)
        section.risk_score = round(0.7 * uncertain_hits + 0.4 * contrast_hits + 0.3 * hedge_hits + parse_penalty, 3)


def _rank_sentences(
    sections: List[Section],
    blocks: List[DocumentBlock],
    block_to_section: Dict[str, str],
) -> List[Tuple[float, str, Section, DocumentBlock]]:
    section_map = _section_by_id(sections)
    sentence_records: List[Tuple[str, Section, DocumentBlock]] = []

    for block in blocks:
        section_id = block_to_section.get(block.id)
        if not section_id:
            continue
        section = section_map[section_id]
        for sentence in split_sentences(block.text):
            if len(sentence.split()) < 6:
                continue
            sentence_records.append((sentence, section, block))

    if not sentence_records:
        return []

    corpus = [record[0] for record in sentence_records]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)

    scores = np.asarray(matrix.sum(axis=1)).ravel()
    ranked = sorted(
        [
            (float(score), sentence, section, block)
            for score, (sentence, section, block) in zip(scores, sentence_records)
        ],
        key=lambda row: row[0],
        reverse=True,
    )
    return ranked


def build_summary_takeaways_and_flags(
    sections: List[Section],
    blocks: List[DocumentBlock],
    block_to_section: Dict[str, str],
    entities: List[Entity],
) -> Tuple[str, List[Takeaway], List[ConfidenceFlag]]:
    ranked = _rank_sentences(sections, blocks, block_to_section)
    if not ranked:
        return (
            "No summary could be generated because the document did not provide enough sentence-level text.",
            [],
            [],
        )

    used = set()
    takeaway_items: List[Takeaway] = []
    summary_sentences: List[str] = []

    for _, sentence, section, block in ranked:
        key = sentence.lower()
        if key in used:
            continue
        used.add(key)
        citation = _make_citation(section, block, sentence)
        confidence = "explicit" if _confidence_from_sentence(sentence) == "explicit" else "inferred"
        takeaway_items.append(Takeaway(text=sentence, confidence=confidence, citation=citation))
        if len(summary_sentences) < 3:
            summary_sentences.append(sentence)
        if len(takeaway_items) >= 6:
            break

    summary = " ".join(summary_sentences)
    confidence_flags: List[ConfidenceFlag] = []
    for takeaway in takeaway_items[:5]:
        confidence_flags.append(
            ConfidenceFlag(
                statement=takeaway.text,
                confidence=takeaway.confidence,
                reason="Direct text span extracted from document." if takeaway.confidence == "explicit" else "Heuristic synthesis from ranked sentence extraction.",
                citation=takeaway.citation,
            )
        )

    if entities:
        top_entity = max(entities, key=lambda entity: entity.mentions_count)
        citation = top_entity.citations[0] if top_entity.citations else takeaway_items[0].citation
        confidence_flags.append(
            ConfidenceFlag(
                statement=f"The document heavily emphasizes '{top_entity.label}'.",
                confidence="inferred",
                reason="Inferred from entity frequency aggregation across sections.",
                citation=citation,
            )
        )

    return summary, takeaway_items, confidence_flags


def build_topic_frequencies(entities: List[Entity]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for entity in entities:
        if entity.type in {"concept", "constraint", "requirement"}:
            counts[entity.label] += entity.mentions_count
    return dict(counts.most_common(25))


def build_timeline(entities: List[Entity]) -> List[Dict[str, object]]:
    points: List[Tuple[datetime, Dict[str, object]]] = []
    for entity in entities:
        if entity.type != "date":
            continue
        parsed = parse_fuzzy_date(entity.label)
        if not parsed:
            continue
        citation = entity.citations[0] if entity.citations else None
        points.append(
            (
                parsed,
                {
                    "date": parsed.strftime("%Y-%m-%d"),
                    "label": entity.label,
                    "section_id": citation.section_id if citation else "",
                    "section_title": citation.section_title if citation else "",
                    "snippet": citation.snippet if citation else "",
                },
            )
        )

    points.sort(key=lambda row: row[0])
    return [payload for _, payload in points]


def build_insights(sections: List[Section], topic_freq: Dict[str, int]) -> List[InsightPoint]:
    insights: List[InsightPoint] = []

    for section in sections:
        citation = Citation(
            section_id=section.id,
            section_title=section.title,
            block_id=section.block_ids[0] if section.block_ids else "",
            snippet=make_reference_snippet(section.text),
            score=1.0,
        )
        insights.append(
            InsightPoint(
                id=f"ins-risk-{section.id}",
                kind="risk",
                title="Risk / Ambiguity",
                value=float(section.risk_score + section.ambiguity_score),
                label=section.title,
                section_id=section.id,
                citation=citation,
                metadata={
                    "risk_score": section.risk_score,
                    "ambiguity_score": section.ambiguity_score,
                },
            )
        )

    for idx, (topic, count) in enumerate(topic_freq.items(), start=1):
        insights.append(
            InsightPoint(
                id=f"ins-topic-{idx}",
                kind="topic",
                title="Topic Frequency",
                value=float(count),
                label=topic,
            )
        )

    return insights
