"""Computed lexical offline components. No synthetic semantic model."""

import math
import re

from .data import parse_corpus
from .types import Candidate, CriterionScore, QueryPlan, SearchHit

TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text):
    return TOKEN.findall(text.lower())


def matches(doc, f):
    value = doc[f.field]
    if f.op == "Eq":
        return value == f.value
    if f.op == "NotEq":
        return value != f.value
    if f.op == "Contains":
        return f.value in value
    if f.op in {"In", "NotIn"}:
        present = (
            any(item in f.value for item in value) if isinstance(value, list) else value in f.value
        )
        return present if f.op == "In" else not present
    return value >= f.value if f.op == "Gte" else value <= f.value


class LexicalPlanner:
    def __init__(self, settings):
        self.settings = settings

    def plan(self, config):
        return QueryPlan(None, config.nl_description, list(config.filters))


class LexicalClient:
    def __init__(self, corpus):
        self.corpus = parse_corpus(corpus)
        self.calls = []
        self.bodies = [tokens(doc["body"]) for doc in self.corpus]
        self.average_length = sum(map(len, self.bodies)) / len(self.bodies)

    def query(self, plan, embedding):
        if embedding is not None:
            raise ValueError("Offline lexical client does not accept an embedding")
        query = list(dict.fromkeys(tokens(plan.keyword_query or "")))
        dfs = {t: sum(t in body for body in self.bodies) for t in query}
        hits = []
        for doc, body in zip(self.corpus, self.bodies):
            if not all(matches(doc, f) for f in plan.filters):
                continue
            if plan.lexical_channel == "metadata":
                title = set(tokens(doc["title"]))
                tags = set(tokens(" ".join(doc["tags"])))
                score = sum(3 * (t in title) + 2 * (t in tags) for t in query) / max(1, len(query))
            else:
                score = sum(
                    math.log(1 + (len(self.corpus) + 1) / (dfs[t] + 1))
                    * body.count(t)
                    / (body.count(t) + 1)
                    / math.sqrt(max(1, len(body)) / max(1, self.average_length))
                    for t in query
                )
            if score > 0:
                hits.append(
                    SearchHit(doc["id"], score, {k: v for k, v in doc.items() if k != "id"})
                )
        ordered = sorted(hits, key=lambda hit: (-hit.score, hit.id))[: plan.top_k]
        self.calls.append(
            {
                "channel": plan.lexical_channel,
                "filters": list(plan.filters),
                "hits": [h.id for h in ordered],
            }
        )
        return ordered


class IdentityReranker:
    last_mode = "identity-no-learned-reranker"

    def rerank(self, config, candidates):
        return [Candidate(c.id, c.score, c.attributes, c.score) for c in candidates]


class TokenCriterionScorer:
    mode = "exact-token-gate"

    def score_batch(self, config, candidates):
        scores = []
        for candidate in candidates:
            attrs = candidate.attributes
            observed = set(
                tokens(
                    f"{attrs.get('title', '')} {attrs.get('body', '')} {' '.join(attrs.get('tags', []))}"
                )
            )
            required = [set(tokens(c)) for c in config.hard_criteria]
            preferred = [set(tokens(c)) for c in config.soft_criteria]
            scores.append(
                CriterionScore(
                    candidate.id,
                    [bool(c) and c <= observed for c in required],
                    [round(3 * len(c & observed) / max(1, len(c))) for c in preferred],
                    {"method": self.mode},
                )
            )
        return scores
