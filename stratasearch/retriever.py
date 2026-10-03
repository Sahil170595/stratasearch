from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from stratasearch.types import Candidate, Filter, PipelineError, QueryConfig, QueryPlan, SearchHit

logger = logging.getLogger("stratasearch.retriever")
_RRF_K = 60
_RELAX_RANK = {"year": 0, "kind": 2, "title": 2, "topic": 3, "tags": 3, "collection": 4}
_DEFAULT_RELAX_RANK = 2


def _relax_rank(flt: Filter) -> int:
    return _RELAX_RANK.get(flt.field, _DEFAULT_RELAX_RANK)


def _format_filter(flt: Filter) -> str:
    return f"{flt.field} {flt.op} {flt.value!r}"


class SearchRetriever:
    """Relax primary filters first, reuse the final set, then fuse ranked channels."""

    def __init__(self, tpuf, embedder, planner, settings):
        self._tpuf, self._embedder, self._planner = (tpuf, embedder, planner)
        self.settings = settings
        self.final_filters = []
        self.dropped_filters = []
        self.attempts = []
        self.channels = {}

    def search(self, config: QueryConfig) -> list[Candidate]:
        self.final_filters, self.dropped_filters, self.attempts = ([], [], [])
        self.channels = {}
        plan = self._planner.plan(config)
        self.plan = plan
        logger.info(
            "planned config=%s top_k=%d rerank_top_k=%d filters=%d vector=%s keyword=%s",
            config.query_id,
            plan.top_k,
            plan.rerank_top_k,
            len(plan.filters),
            bool(plan.vector_query),
            bool(plan.keyword_query),
        )
        embedding: list[float] | None = None
        if plan.vector_query:
            vectors = self._embedder.embed([plan.vector_query], input_type="query")
            if not vectors:
                raise PipelineError(
                    f"Embedder returned no vectors for vector_query in {config.query_id}."
                )
            embedding = vectors[0]
        hybrid_on = self.settings.hybrid
        if hybrid_on:
            candidates = self._search_hybrid_union(config, plan, embedding)
        else:
            candidates = self._search_vector_only(config, plan, embedding)
        logger.info("retrieved %d candidates for config=%s", len(candidates), config.query_id)
        if len(candidates) < self.settings.minimum_candidates:
            raise PipelineError(
                f"Retriever returned only {len(candidates)} candidates for {config.query_id}; need at least {self.settings.minimum_candidates}."
            )
        return candidates

    def _search_vector_only(
        self, config: QueryConfig, plan: QueryPlan, embedding: list[float] | None
    ) -> list[Candidate]:
        hits, _final_filters = self._vector_with_relaxation(config, plan, embedding)
        self.channels["primary"] = [{"id": h.id, "score": h.score} for h in hits]
        return [
            Candidate(id=h.id, score=h.score, attributes=h.attributes, rerank_score=None)
            for h in hits
        ]

    def _search_hybrid_union(
        self, config: QueryConfig, plan: QueryPlan, embedding: list[float] | None
    ) -> list[Candidate]:
        vector_hits, final_filters = self._vector_with_relaxation(config, plan, embedding)
        self.channels["primary"] = [{"id": h.id, "score": h.score} for h in vector_hits]
        keyword_text = plan.keyword_query or (config.nl_description[:200] or None)
        if not keyword_text:
            logger.info(
                "hybrid union: no keyword signal for %s; skipping BM25 branch", config.query_id
            )
            return [
                Candidate(id=h.id, score=h.score, attributes=h.attributes, rerank_score=None)
                for h in vector_hits
            ]
        bm25_plan = QueryPlan(
            vector_query=None,
            keyword_query=keyword_text,
            filters=final_filters,
            top_k=plan.top_k,
            rerank_top_k=plan.rerank_top_k,
            lexical_channel="metadata",
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(self._tpuf.query, bm25_plan, None)
            bm25_hits: list[SearchHit] = fut.result()
        self.channels["secondary"] = [{"id": h.id, "score": h.score} for h in bm25_hits]
        merged = _rrf_merge(vector_hits, bm25_hits, top_k=plan.top_k)
        logger.info(
            "hybrid union: vector=%d, bm25=%d, merged=%d for %s",
            len(vector_hits),
            len(bm25_hits),
            len(merged),
            config.query_id,
        )
        return merged

    def _vector_with_relaxation(
        self, config: QueryConfig, plan: QueryPlan, embedding: list[float] | None
    ) -> tuple[list[SearchHit], list[Filter]]:
        hits = self._tpuf.query(plan, embedding)
        self.attempts.append({"filters": list(plan.filters), "count": len(hits)})
        logger.info(
            "retrieved %d hits for config=%s (initial, %d filters)",
            len(hits),
            config.query_id,
            len(plan.filters),
        )
        current_filters = list(plan.filters)
        while len(hits) < self.settings.relax_threshold and current_filters:
            drop_idx = min(
                range(len(current_filters)), key=lambda i: (_relax_rank(current_filters[i]), i)
            )
            dropped = current_filters.pop(drop_idx)
            self.dropped_filters.append(dropped)
            logger.warning(
                "relaxing filters for config=%s: only %d hits (< %d); dropping %s; remaining=%d",
                config.query_id,
                len(hits),
                self.settings.relax_threshold,
                _format_filter(dropped),
                len(current_filters),
            )
            relaxed_plan = QueryPlan(
                vector_query=plan.vector_query,
                keyword_query=plan.keyword_query,
                filters=current_filters,
                top_k=plan.top_k,
                rerank_top_k=plan.rerank_top_k,
            )
            hits = self._tpuf.query(relaxed_plan, embedding)
            self.attempts.append({"filters": list(current_filters), "count": len(hits)})
            logger.info(
                "retrieved %d hits for config=%s (after relaxation, %d filters remaining)",
                len(hits),
                config.query_id,
                len(current_filters),
            )
        if len(hits) < self.settings.relax_threshold:
            logger.warning(
                "filter relaxation exhausted for config=%s: %d hits with 0 filters",
                config.query_id,
                len(hits),
            )
        self.final_filters = list(current_filters)
        return (hits, current_filters)


def _rrf_merge(
    vector_hits: list[SearchHit], bm25_hits: list[SearchHit], *, top_k: int
) -> list[Candidate]:
    """One-based RRF with k=60; absent branches add nothing; ties use document ID."""
    scores: dict[str, float] = {}
    attrs: dict[str, dict] = {}
    for rank, hit in enumerate(vector_hits, start=1):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (_RRF_K + rank)
        attrs.setdefault(hit.id, hit.attributes)
    for rank, hit in enumerate(bm25_hits, start=1):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (_RRF_K + rank)
        attrs.setdefault(hit.id, hit.attributes)
    ordered_ids = sorted(scores, key=lambda cid: (-scores[cid], cid))
    ordered_ids = ordered_ids[:top_k]
    return [
        Candidate(id=cid, score=scores[cid], attributes=attrs[cid], rerank_score=None)
        for cid in ordered_ids
    ]
