from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import analyze_document
from src.qa import answer_question


DEFAULT_DOC = Path("/Users/edwinbrown/Downloads/STA 9708 LN3.1 Rules of Probability 2-10-2026.docx")


@pytest.mark.skipif(not DEFAULT_DOC.exists(), reason="Default document not present")
def test_pipeline_returns_sections_entities_requirements() -> None:
    analysis = analyze_document(file_path=str(DEFAULT_DOC))
    assert analysis.sections
    assert analysis.blocks
    assert analysis.file_type in {"docx", "pdf", "txt"}
    assert isinstance(analysis.summary, str)


@pytest.mark.skipif(not DEFAULT_DOC.exists(), reason="Default document not present")
def test_qa_refuses_unsupported_claim() -> None:
    analysis = analyze_document(file_path=str(DEFAULT_DOC))
    response = answer_question("this claim does not exist qqqqxxxx", analysis)
    assert response.answer == "Not found in document" or response.supported is False


def test_txt_fallback_works(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text(
        "Section One\nProbability is a measure.\n\nSection Two\nA rule says must validate outcomes.\n",
        encoding="utf-8",
    )
    analysis = analyze_document(file_path=str(sample))
    assert analysis.sections
    assert analysis.blocks
    assert analysis.file_type == "txt"
