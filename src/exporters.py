from __future__ import annotations

import json
from typing import Dict

import pandas as pd

from .models import AnalysisResult


def entities_to_dataframe(analysis: AnalysisResult) -> pd.DataFrame:
    rows = []
    for entity in analysis.entities:
        rows.append(
            {
                "id": entity.id,
                "label": entity.label,
                "type": entity.type,
                "mentions_count": entity.mentions_count,
                "first_occurrence": entity.first_occurrence,
                "last_occurrence": entity.last_occurrence,
                "source_sections": " | ".join(entity.source_sections),
            }
        )
    return pd.DataFrame(rows)


def requirements_to_dataframe(analysis: AnalysisResult) -> pd.DataFrame:
    rows = []
    for req in analysis.requirements:
        citation = req.citations[0] if req.citations else None
        rows.append(
            {
                "id": req.id,
                "statement": req.statement,
                "priority": req.priority,
                "rationale": req.rationale,
                "verification_method": req.verification_method,
                "section_id": req.section_id,
                "section_title": req.section_title,
                "confidence": req.confidence,
                "citation_snippet": citation.snippet if citation else "",
            }
        )
    return pd.DataFrame(rows)


def summary_payload(analysis: AnalysisResult) -> Dict[str, object]:
    return {
        "file_name": analysis.file_name,
        "file_type": analysis.file_type,
        "summary": analysis.summary,
        "takeaways": [
            {
                "text": item.text,
                "confidence": item.confidence,
                "section": item.citation.section_title,
                "snippet": item.citation.snippet,
            }
            for item in analysis.takeaways
        ],
        "confidence_flags": [
            {
                "statement": item.statement,
                "confidence": item.confidence,
                "reason": item.reason,
                "section": item.citation.section_title,
            }
            for item in analysis.confidence_flags
        ],
        "warnings": analysis.warnings,
    }


def export_entities_csv(analysis: AnalysisResult) -> bytes:
    return entities_to_dataframe(analysis).to_csv(index=False).encode("utf-8")


def export_entities_json(analysis: AnalysisResult) -> bytes:
    payload = [entity.model_dump() for entity in analysis.entities]
    return json.dumps(payload, indent=2).encode("utf-8")


def export_requirements_csv(analysis: AnalysisResult) -> bytes:
    return requirements_to_dataframe(analysis).to_csv(index=False).encode("utf-8")


def export_requirements_json(analysis: AnalysisResult) -> bytes:
    payload = [req.model_dump() for req in analysis.requirements]
    return json.dumps(payload, indent=2).encode("utf-8")


def export_summary_json(analysis: AnalysisResult) -> bytes:
    return json.dumps(summary_payload(analysis), indent=2).encode("utf-8")


def export_summary_markdown(analysis: AnalysisResult) -> bytes:
    payload = summary_payload(analysis)
    lines = [
        f"# Analysis Summary: {analysis.file_name}",
        "",
        "## Executive Summary",
        str(payload["summary"]),
        "",
        "## Key Takeaways",
    ]
    takeaways = payload["takeaways"]
    if not takeaways:
        lines.append("- No takeaways were extracted.")
    else:
        for item in takeaways:
            lines.append(
                f"- {item['text']} (confidence: {item['confidence']}, section: {item['section']})"
            )

    lines.extend(["", "## Confidence Flags"])
    flags = payload["confidence_flags"]
    if not flags:
        lines.append("- No confidence flags were generated.")
    else:
        for item in flags:
            lines.append(f"- {item['statement']} [{item['confidence']}] - {item['reason']}")

    if analysis.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {warning}" for warning in analysis.warnings])

    return "\n".join(lines).encode("utf-8")
