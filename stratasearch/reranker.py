"""Optional learned reranker, retained query/bonus logic, explicit identity fallback."""

import math

from .rerank_helpers import RerankHelpers
from .types import Candidate, RerankError


class Reranker(RerankHelpers):
    def __init__(self, settings, client=None):
        self._settings, self._client = settings, client
        self.last_mode = "not-run"

    def _get_client(self):
        if self._client is None:
            if not self._settings.voyage_api_key or not self._settings.rerank_model:
                raise RerankError("Explicit reranking key and model required")
            from voyageai import Client

            self._client = Client(api_key=self._settings.voyage_api_key, max_retries=0, timeout=30)
        return self._client

    @classmethod
    def _build_rich_document(cls, candidate):
        attrs = candidate.attributes
        body = str(attrs.get("body", ""))
        metadata = [
            f"{key}: {attrs[key]}"
            for key in ("title", "kind", "topic", "collection", "year")
            if key in attrs
        ]
        tags = cls._as_str_list(attrs.get("tags"))
        # Trim metadata tags first, then the assembled document if necessary.
        for count in (20, 10, 5, 0):
            text = "\n".join([body, *metadata, "Tags: " + "; ".join(tags[:count])])
            if len(text) <= 8000:
                return text
        return text[:8000]

    def _fallback(self, candidates):
        self.last_mode = "identity-fallback"
        return self._identity_fallback(candidates)

    def rerank(self, config, candidates):
        if not candidates or len({c.id for c in candidates}) != len(candidates):
            raise RerankError("Nonempty distinct candidates required")
        query = self._build_query(config, soft_on=self._settings.soft_criteria)
        if not query:
            return self._fallback(candidates)
        documents = [
            self._build_rich_document(c)
            if self._settings.rich_document
            else c.attributes.get("body", "")
            for c in candidates
        ]
        try:
            results = (
                self._get_client()
                .rerank(
                    query=query,
                    documents=documents,
                    model=self._settings.rerank_model,
                    top_k=len(candidates),
                )
                .results
            )
            if len(results) != len(candidates):
                raise ValueError("Reranker count mismatch")
            seen = set()
            scored = []
            for result in results:
                index, value = result.index, result.relevance_score
                if type(index) is not int or index not in range(len(candidates)) or index in seen:
                    raise ValueError("Invalid or duplicate reranker index")
                if type(value) not in {int, float} or not math.isfinite(value):
                    raise ValueError("Invalid reranking score")
                seen.add(index)
                c = candidates[index]
                bonus = (
                    self._evidence_bonus(c, config.hard_criteria)
                    if self._settings.evidence_bonus
                    else 0
                )
                scored.append(Candidate(c.id, c.score, c.attributes, float(value) + bonus))
        except Exception:
            return self._fallback(candidates)
        scored.sort(key=lambda c: (-(c.rerank_score or 0), c.id))
        self._assert_invariants(candidates, scored)
        self.last_mode = "provider-reranker"
        return scored
