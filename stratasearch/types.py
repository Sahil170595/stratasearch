from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

FilterOp = Literal["Eq", "NotEq", "In", "NotIn", "Contains", "Gte", "Lte"]


@dataclass(frozen=True)
class Filter:
    field: str
    op: FilterOp
    value: Any


@dataclass(frozen=True)
class QueryConfig:
    query_id: str
    nl_description: str
    hard_criteria: list[str] = field(default_factory=list)
    soft_criteria: list[str] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)


@dataclass(frozen=True)
class QueryPlan:
    vector_query: str | None
    keyword_query: str | None
    filters: list[Filter]
    top_k: int = 200
    rerank_top_k: int = 50
    lexical_channel: str = "body"


@dataclass(frozen=True)
class SearchHit:
    id: str
    score: float
    attributes: dict[str, Any]


@dataclass
class Candidate:
    id: str
    score: float
    attributes: dict[str, Any]
    rerank_score: float | None = None


@dataclass(frozen=True)
class CriterionScore:
    candidate_id: str
    hard_pass: list[bool]
    soft_scores: list[int]
    raw: dict[str, Any]


class PipelineError(Exception):
    pass


class TPUFError(PipelineError):
    pass


class EmbedError(PipelineError):
    pass


class PlanError(PipelineError):
    pass


class RerankError(PipelineError):
    pass


class CriterionScoreError(PipelineError):
    pass


@runtime_checkable
class Embedder(Protocol):
    def embed(
        self, texts: list[str], input_type: Literal["query", "document"] = "query"
    ) -> list[list[float]]: ...


@runtime_checkable
class Planner(Protocol):
    def plan(self, config: QueryConfig) -> QueryPlan: ...


@runtime_checkable
class Retriever(Protocol):
    def search(self, config: QueryConfig) -> list[Candidate]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, config: QueryConfig, candidates: list[Candidate]) -> list[Candidate]: ...
