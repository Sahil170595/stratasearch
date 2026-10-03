"""Typed configuration boundary; imports never load credentials or .env files."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    mode: str = "lexical-offline"
    hybrid: bool = True
    relax_threshold: int = 6
    minimum_candidates: int = 1
    voyage_dim: int = 1024
    voyage_model: str = ""
    rerank_model: str = ""
    planner_model: str = ""
    scorer_model: str = ""
    hyde_enabled: bool = False
    soft_criteria: bool = False
    rich_document: bool = True
    evidence_bonus: bool = False
    voyage_api_key: str = field(default="", repr=False)
    openai_api_key: str = field(default="", repr=False)
    turbopuffer_api_key: str = field(default="", repr=False)
    turbopuffer_namespace: str = ""
    turbopuffer_region: str = ""

    def __post_init__(self):
        if self.mode not in {"lexical-offline", "provider-live"}:
            raise ValueError("Unknown runtime mode")
        for name in ("hybrid", "hyde_enabled", "soft_criteria", "rich_document", "evidence_bonus"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("relax_threshold", "minimum_candidates", "voyage_dim"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_environment(cls):
        def require(name):
            value = os.environ.get(name, "").strip()
            if not value:
                raise ValueError(f"Live mode requires {name}; no credential values are logged")
            return value

        return cls(
            mode="provider-live",
            voyage_api_key=require("VOYAGE_API_KEY"),
            openai_api_key=require("OPENAI_API_KEY"),
            turbopuffer_api_key=require("TURBOPUFFER_API_KEY"),
            turbopuffer_namespace=require("STRATASEARCH_NAMESPACE"),
            turbopuffer_region=require("STRATASEARCH_REGION"),
            voyage_model=require("STRATASEARCH_EMBED_MODEL"),
            rerank_model=require("STRATASEARCH_RERANK_MODEL"),
            planner_model=require("STRATASEARCH_PLANNER_MODEL"),
            scorer_model=require("STRATASEARCH_SCORER_MODEL"),
            voyage_dim=int(require("STRATASEARCH_EMBED_DIM")),
        )
