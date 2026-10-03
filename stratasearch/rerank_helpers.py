import re
from typing import Any

from stratasearch.types import Candidate, QueryConfig

_EVIDENCE_FIELDS = ("title", "body", "tags", "kind", "topic")
_EVIDENCE_MAX_BONUS = 0.05
_EVIDENCE_TOKEN_RE = re.compile("[A-Za-z0-9]+")
_EVIDENCE_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "with",
    "at",
    "for",
    "to",
    "and",
    "or",
    "on",
    "from",
}


class RerankHelpers:
    @staticmethod
    def _build_query(config: QueryConfig, *, soft_on: bool) -> str:
        if not soft_on:
            query_parts: list[str] = []
            if config.nl_description:
                query_parts.append(config.nl_description.strip())
            if config.hard_criteria:
                query_parts.append(" ".join((c.strip() for c in config.hard_criteria if c)))
            return "\n".join((p for p in query_parts if p)).strip()
        sections: list[str] = []
        desc = (config.nl_description or "").strip()
        if desc:
            sections.append(f"Search description:\n{desc}")
        hards = [c.strip() for c in config.hard_criteria or [] if c and c.strip()]
        if hards:
            numbered = "\n".join((f"{i + 1}. {c}" for i, c in enumerate(hards)))
            sections.append(f"Must-have criteria:\n{numbered}")
        softs = [c.strip() for c in config.soft_criteria or [] if c and c.strip()]
        if softs:
            numbered = "\n".join((f"{i + 1}. {c}" for i, c in enumerate(softs)))
            sections.append(f"Nice-to-have criteria:\n{numbered}")
        return "\n\n".join(sections).strip()

    @staticmethod
    def _as_str_list(val: Any) -> list[str]:
        if val is None:
            return []
        if isinstance(val, list):
            return [str(x).strip() for x in val if x is not None and str(x).strip()]
        s = str(val).strip()
        return [s] if s else []

    @classmethod
    def _evidence_bonus(cls, c: Candidate, hard_criteria: list[str]) -> float:
        if not hard_criteria:
            return 0.0
        attrs = c.attributes or {}
        blob_parts: list[str] = []
        for f in _EVIDENCE_FIELDS:
            for v in cls._as_str_list(attrs.get(f)):
                blob_parts.append(v.lower())
        blob = " ".join(blob_parts)
        if not blob:
            return 0.0
        blob_tokens = set(_EVIDENCE_TOKEN_RE.findall(blob))
        matched = 0
        for crit in hard_criteria:
            if not crit:
                continue
            tokens = [
                t.lower()
                for t in _EVIDENCE_TOKEN_RE.findall(crit)
                if len(t) >= 3 and t.lower() not in _EVIDENCE_STOPWORDS
            ]
            if not tokens:
                continue
            if any((t in blob_tokens for t in tokens)):
                matched += 1
        denom = max(1, len(hard_criteria))
        return _EVIDENCE_MAX_BONUS * (matched / denom)

    @staticmethod
    def _identity_fallback(candidates: list[Candidate]) -> list[Candidate]:
        out = [
            Candidate(id=c.id, score=c.score, attributes=c.attributes, rerank_score=float(c.score))
            for c in candidates
        ]
        RerankHelpers._assert_invariants(candidates, out)
        return out

    @staticmethod
    def _assert_invariants(original: list[Candidate], returned: list[Candidate]) -> None:
        assert len(returned) == len(original), (
            f"rerank length changed: {len(original)} -> {len(returned)}"
        )
        assert {c.id for c in returned} == {c.id for c in original}, (
            "rerank dropped or duplicated candidate ids"
        )
        assert all((c.rerank_score is not None for c in returned)), (
            "rerank left a candidate with rerank_score=None"
        )
