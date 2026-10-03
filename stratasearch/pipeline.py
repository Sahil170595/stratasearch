"""Orchestrate retrieval, reranking, optional scoring, final inspection, local output."""

from dataclasses import asdict, dataclass

from .data import parse_query
from .types import Candidate, QueryConfig


@dataclass
class RunReport:
    query_id: str
    mode: str
    status: str
    selected: list[Candidate]
    rejected: list[Candidate]
    reranking_mode: str
    scoring_mode: str
    requested_filters: list
    final_filters: list
    dropped_filters: list
    planned_filters: list
    attempts: list
    channels: dict
    criteria: dict
    planner_mode: str
    hyde_status: str

    def to_dict(self):
        return {"version": 1, **asdict(self)}


class Pipeline:
    def __init__(self, retriever, reranker, inspector=None, scorer=None, limit=10):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Output limit must be an integer in 1..100")
        self.retriever = retriever
        self.reranker = reranker
        self.inspector = inspector
        self.scorer = scorer
        self.limit = limit

    def run(self, config: QueryConfig):
        config = parse_query(
            {
                "query_id": config.query_id,
                "description": config.nl_description,
                "hard_criteria": config.hard_criteria,
                "soft_criteria": config.soft_criteria,
                "filters": [asdict(f) for f in config.filters],
            }
        )
        candidates = self.retriever.search(config)
        ranked = self.reranker.rerank(config, candidates)
        if len(ranked) != len(candidates) or {c.id for c in ranked} != {c.id for c in candidates}:
            raise ValueError("Reranking must preserve candidate count and identifiers")
        rejected = []
        eligible = ranked
        scores = {}
        if self.scorer:
            scored = self.scorer.score_batch(config, ranked[: self.retriever.plan.rerank_top_k])
            if len({s.candidate_id for s in scored}) != len(scored):
                raise ValueError("Duplicate criterion-score identifiers")
            scores = {s.candidate_id: s for s in scored}
            eligible = []
            for c in ranked:
                score = scores.get(c.id)
                valid = (
                    score is not None
                    and len(score.hard_pass) == len(config.hard_criteria)
                    and len(score.soft_scores) == len(config.soft_criteria)
                )
                valid = (
                    valid
                    and all(type(p) is bool for p in score.hard_pass)
                    and all(type(p) is int and 0 <= p <= 3 for p in score.soft_scores)
                )
                if not valid or score.raw.get("error") or not all(score.hard_pass):
                    rejected.append(c)
                else:
                    eligible.append(c)
            eligible.sort(
                key=lambda c: (
                    -sum(scores[c.id].hard_pass),
                    -sum(scores[c.id].soft_scores),
                    -(c.rerank_score if c.rerank_score is not None else c.score),
                    c.id,
                )
            )
        selected = eligible[: self.limit]
        if self.inspector:
            self.inspector.write(config, selected, top_n=self.limit)
        mode = self.retriever.settings.mode
        return RunReport(
            config.query_id,
            mode,
            "ready" if len(selected) >= self.limit else "shortfall",
            selected,
            rejected,
            self.reranker.last_mode,
            getattr(self.scorer, "mode", "provider-criterion-scorer")
            if self.scorer
            else "disabled",
            list(config.filters),
            list(self.retriever.final_filters),
            list(self.retriever.dropped_filters),
            list(self.retriever.plan.filters),
            list(self.retriever.attempts),
            dict(self.retriever.channels),
            {
                key: {
                    "hard_pass": s.hard_pass,
                    "soft_scores": s.soft_scores,
                    "failed": bool(s.raw.get("error")),
                }
                for key, s in scores.items()
            },
            getattr(self.retriever._planner, "last_mode", "lexical-no-llm"),
            getattr(self.retriever._planner, "hyde_status", "disabled"),
        )
