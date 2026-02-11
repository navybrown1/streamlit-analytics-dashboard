from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


EntityType = Literal[
    "person",
    "org",
    "date",
    "requirement",
    "metric",
    "tool",
    "constraint",
    "deliverable",
    "concept",
]

Priority = Literal["High", "Medium", "Low"]
ConfidenceLabel = Literal["explicit", "inferred"]


class DocumentBlock(BaseModel):
    id: str
    index: int
    text: str
    source: str
    page: Optional[int] = None
    style: Optional[str] = None
    bold_ratio: float = 0.0
    start_char: int = 0
    end_char: int = 0


class Citation(BaseModel):
    section_id: str
    section_title: str
    block_id: str
    snippet: str
    score: float = 1.0


class Section(BaseModel):
    id: str
    title: str
    level: int = 1
    section_type: str = "content"
    start_block: int
    end_block: int
    block_ids: List[str] = Field(default_factory=list)
    text: str
    parse_confidence: float = 0.5
    risk_score: float = 0.0
    ambiguity_score: float = 0.0


class Entity(BaseModel):
    id: str
    label: str
    normalized: str
    type: EntityType
    mentions_count: int = 1
    first_occurrence: str
    last_occurrence: str
    source_sections: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)


class RequirementItem(BaseModel):
    id: str
    statement: str
    priority: Priority
    rationale: str
    verification_method: str
    section_id: str
    section_title: str
    confidence: ConfidenceLabel
    citations: List[Citation] = Field(default_factory=list)


class RelationshipEdge(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    weight: float = 1.0
    citations: List[Citation] = Field(default_factory=list)


class InsightPoint(BaseModel):
    id: str
    kind: str
    title: str
    value: float
    label: str
    section_id: Optional[str] = None
    citation: Optional[Citation] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Takeaway(BaseModel):
    text: str
    confidence: ConfidenceLabel
    citation: Citation


class ConfidenceFlag(BaseModel):
    statement: str
    confidence: ConfidenceLabel
    reason: str
    citation: Citation


class AnalysisResult(BaseModel):
    file_name: str
    file_type: str
    file_hash: str
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    blocks: List[DocumentBlock] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    requirements: List[RequirementItem] = Field(default_factory=list)
    relationships: List[RelationshipEdge] = Field(default_factory=list)
    insights: List[InsightPoint] = Field(default_factory=list)
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    topic_frequencies: Dict[str, int] = Field(default_factory=dict)
    summary: str = ""
    takeaways: List[Takeaway] = Field(default_factory=list)
    confidence_flags: List[ConfidenceFlag] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class QAResponse(BaseModel):
    answer: str
    supported: bool
    score: float
    citations: List[Citation] = Field(default_factory=list)
    nearest_sections: List[Citation] = Field(default_factory=list)
