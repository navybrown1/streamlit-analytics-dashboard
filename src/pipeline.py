from __future__ import annotations

from datetime import datetime
from typing import Optional

from .extractors import (
    build_insights,
    build_relationships,
    build_summary_takeaways_and_flags,
    build_timeline,
    build_topic_frequencies,
    compute_risk_and_ambiguity,
    extract_entities,
    extract_requirements,
)
from .models import AnalysisResult
from .parsers import compute_hash, parse_document
from .sectionizer import sectionize_blocks

PARSER_VERSION = "1.0.0"


def analyze_document(
    file_path: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
) -> AnalysisResult:
    file_name, file_type, raw_bytes, blocks, metadata, parse_warnings = parse_document(
        file_path=file_path,
        file_bytes=file_bytes,
        filename=filename,
    )

    sections, block_to_section, section_warnings = sectionize_blocks(blocks)
    compute_risk_and_ambiguity(sections)

    entities, section_entity_counts = extract_entities(sections, blocks, block_to_section)
    requirements = extract_requirements(sections, blocks, block_to_section)
    relationships = build_relationships(sections, entities, requirements, section_entity_counts)

    summary, takeaways, confidence_flags = build_summary_takeaways_and_flags(
        sections,
        blocks,
        block_to_section,
        entities,
    )
    topic_frequencies = build_topic_frequencies(entities)
    timeline = build_timeline(entities)
    insights = build_insights(sections, topic_frequencies)

    warnings = parse_warnings + section_warnings
    if not sections:
        warnings.append("No sections detected.")
    if not entities:
        warnings.append("No entities detected.")
    if not requirements:
        warnings.append("No requirement-like statements detected.")

    metadata = {
        **metadata,
        "parser_version": PARSER_VERSION,
        "block_count": len(blocks),
        "section_count": len(sections),
    }

    result = AnalysisResult(
        file_name=file_name,
        file_type=file_type,
        file_hash=compute_hash(raw_bytes),
        created_at=datetime.utcnow(),
        metadata=metadata,
        blocks=blocks,
        sections=sections,
        entities=entities,
        requirements=requirements,
        relationships=relationships,
        insights=insights,
        timeline=timeline,
        topic_frequencies=topic_frequencies,
        summary=summary,
        takeaways=takeaways,
        confidence_flags=confidence_flags,
        warnings=warnings,
    )

    # Ensure schema validation is enforced at runtime.
    return AnalysisResult.model_validate(result.model_dump())
