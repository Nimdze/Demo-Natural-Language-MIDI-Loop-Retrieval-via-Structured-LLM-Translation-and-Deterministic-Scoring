"""Pydantic models for the search engine."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Target(BaseModel):
    """One LLM target: a concept at a specific level with scoring policy."""

    concept_name: str = Field(..., description="Exact concept name from the taxonomy.")
    level_name: str = Field(..., description="Exact level name from the taxonomy.")
    importance: int = Field(..., ge=1, le=5, description="Target importance, 1-5.")
    fallback: Literal["down", "up", "nearest"] = Field(
        "nearest",
        description="Fallback direction: down (at least), up (at most), nearest.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "concept_name": "rhythmic_density_average_events_per_beat",
                "level_name": "Dense",
                "importance": 4,
                "fallback": "down",
            }
        }
    )


class QueryTranslation(BaseModel):
    """Validated LLM output for a query."""

    family_classification: Literal["drums", "pitched"] | None = Field(
        None,
        description="Classified instrument family: drums or pitched.",
    )
    semantic_analysis: list[dict] = Field(
        default_factory=list,
        description="Per-concept explanation (off by default).",
    )
    targets: list[Target] = Field(default_factory=list)


class LLMCallRecord(BaseModel):
    """Record of one LLM translation call."""

    query: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    response_time_ms: float
    translation: QueryTranslation | None
    error: str | None = None
    raw: str | None = None
    attempts: list[dict] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SearchResultItem(BaseModel):
    """One ranked file."""

    file_path: str
    file_id: int | None = None
    score: float = Field(..., ge=0.0, le=1.0)
    raw_score: float
    max_possible: float
    all_tags: dict[str, str] = Field(default_factory=dict, description="All concept->level tags for this file")
    match_reasons: list[str] = Field(default_factory=list)
    matched_tags: list[str] = Field(default_factory=list)
    missing_must_have: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Full response to a search query."""

    query: str
    instrument_family: str
    targets: list[Target]
    results: list[SearchResultItem]
    duration_ms: float
    llm_call_record: LLMCallRecord | None = None


class ConceptCoverage(BaseModel):
    """Coverage of one concept by the validation suite."""

    concept_name: str
    triggered: int
    total: int
    coverage: float


class ValidationReport(BaseModel):
    """Output of a validation run."""

    total_tests: int
    passed_schema: int
    failed_schema: int
    hallucinations: int
    coverage: list[ConceptCoverage]
    stability_avg: float
    paraphrase_consistency_avg: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
