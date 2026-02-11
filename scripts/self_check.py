#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.exporters import (
    export_entities_csv,
    export_entities_json,
    export_requirements_csv,
    export_requirements_json,
    export_summary_json,
)
from src.pipeline import analyze_document
from src.qa import answer_question


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-check for Document Intelligence Dashboard pipeline")
    parser.add_argument("--doc", required=True, help="Path to input document (DOCX/PDF/TXT)")
    args = parser.parse_args()

    path = Path(args.doc)
    require(path.exists(), f"Document does not exist: {path}")

    analysis = analyze_document(file_path=str(path))

    # Acceptance-aligned structural checks
    require(len(analysis.sections) > 0, "No sections parsed")
    require(len(analysis.blocks) > 0, "No text blocks parsed")

    # Search viability: at least one entity label appears in section text
    if analysis.entities:
        first_label = analysis.entities[0].label.lower()
        combined = "\n".join(section.text.lower() for section in analysis.sections)
        require(first_label in combined, "Entity label not searchable in section text")

    # Clickability viability: entities and requirements should carry citations when present
    if analysis.entities:
        require(any(entity.citations for entity in analysis.entities), "Entities missing citations")
    if analysis.requirements:
        require(any(item.citations for item in analysis.requirements), "Requirements missing citations")

    # Export checks
    require(len(export_entities_csv(analysis)) > 0, "Entity CSV export is empty")
    require(len(export_entities_json(analysis)) > 0, "Entity JSON export is empty")
    require(len(export_requirements_csv(analysis)) > 0, "Requirements CSV export is empty")
    require(len(export_requirements_json(analysis)) > 0, "Requirements JSON export is empty")
    require(len(export_summary_json(analysis)) > 0, "Summary export is empty")

    # Q&A grounded behavior
    unsupported = answer_question("zxqvjk unknown unsupported claim", analysis)
    require(
        unsupported.answer.strip() == "Not found in document" or not unsupported.supported,
        "Unsupported Q&A did not refuse unsupported claim",
    )

    supported = answer_question("What is conditional probability?", analysis)
    if supported.supported:
        require(len(supported.citations) > 0, "Supported answer missing citations")

    print("Self-check passed")
    print(
        f"Sections={len(analysis.sections)} Entities={len(analysis.entities)} "
        f"Requirements={len(analysis.requirements)} Warnings={len(analysis.warnings)}"
    )


if __name__ == "__main__":
    main()
