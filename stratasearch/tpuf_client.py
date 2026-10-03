"""Optional official Turbopuffer SDK adapter for a caller-owned neutral namespace."""

import math
from dataclasses import asdict

from .data import FIELDS, parse_corpus, parse_filter
from .types import SearchHit, TPUFError


def _compose_filters(filters):
    tuples = [(f.field, f.op, f.value) for f in [parse_filter(asdict(f)) for f in filters]]
    return ("And", tuples) if len(tuples) > 1 else tuples[0] if tuples else None


class TPUFClient:
    def __init__(self, settings, namespace=None):
        self._settings = settings
        self._ns = namespace
        self._client = None

    def _namespace(self):
        if self._ns is None:
            if (
                not self._settings.turbopuffer_api_key
                or not self._settings.turbopuffer_namespace
                or not self._settings.turbopuffer_region
            ):
                raise TPUFError("Explicit owned namespace, region and key required")
            from turbopuffer import Turbopuffer

            self._client = Turbopuffer(
                api_key=self._settings.turbopuffer_api_key,
                region=self._settings.turbopuffer_region,
                max_retries=0,
                timeout=30,
            )
            self._ns = self._client.namespace(self._settings.turbopuffer_namespace)
        return self._ns

    def query(self, plan, embedding):
        if type(plan.top_k) is not int or not 1 <= plan.top_k <= 1000:
            raise TPUFError("Invalid retrieval depth")
        has_vector = embedding is not None and plan.vector_query is not None
        if has_vector:
            if (
                not isinstance(embedding, list)
                or len(embedding) != self._settings.voyage_dim
                or any(type(v) not in {int, float} or not math.isfinite(v) for v in embedding)
            ):
                raise TPUFError("Invalid query embedding")
            rank_by = ("vector", "ANN", embedding)
        elif plan.keyword_query:
            rank_by = ("body", "BM25", plan.keyword_query)
        else:
            raise TPUFError("No retrieval signal; arbitrary sampling is not search")
        kwargs = {"top_k": plan.top_k, "include_attributes": sorted(FIELDS), "rank_by": rank_by}
        filters = _compose_filters(plan.filters)
        if filters is not None:
            kwargs["filters"] = filters
        try:
            response = self._namespace().query(**kwargs)
            hits = []
            for row in response.rows or []:
                raw = row.model_dump() if hasattr(row, "model_dump") else dict(row)
                score = raw.get("$dist") if has_vector else raw.get("$score", raw.get("$dist"))
                if type(score) not in {int, float} or not math.isfinite(score):
                    raise ValueError("Missing or non-finite provider score")
                attrs = {key: raw[key] for key in FIELDS if key in raw}
                doc = parse_corpus([{**attrs, "id": str(raw["id"])}])[0]
                hits.append(
                    SearchHit(doc["id"], -float(score) if has_vector else float(score), doc)
                )
            if len({h.id for h in hits}) != len(hits):
                raise ValueError("Duplicate provider identifiers")
            return sorted(hits, key=lambda h: (-h.score, h.id))[: plan.top_k]
        except Exception as exc:
            raise TPUFError(f"Search provider failed ({type(exc).__name__})") from None

    def count(self, filters=None):
        kwargs = {"aggregate_by": {"total": ("Count",)}}
        composed = _compose_filters(filters or [])
        if composed is not None:
            kwargs["filters"] = composed
        try:
            total = self._namespace().query(**kwargs).aggregations["total"]
            if type(total) is not int or total < 0:
                raise ValueError("Invalid provider count")
            return total
        except Exception as exc:
            raise TPUFError(f"Count provider failed ({type(exc).__name__})") from None
