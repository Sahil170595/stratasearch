"""Offline export/recompute contract. Integrity checking is not authentication."""

import hashlib
import json
from dataclasses import asdict

from . import __version__
from .config import Settings
from .data import parse_corpus, parse_query
from .local import IdentityReranker, LexicalClient, LexicalPlanner, TokenCriterionScorer
from .pipeline import Pipeline
from .retriever import SearchRetriever

ALGORITHM_KEYS = {"hybrid", "relax_threshold", "minimum_candidates"}


def query_dict(config):
    return {
        "query_id": config.query_id,
        "description": config.nl_description,
        "hard_criteria": config.hard_criteria,
        "soft_criteria": config.soft_criteria,
        "filters": [asdict(f) for f in config.filters],
    }


def corpus_digest(corpus):
    canonical = json.dumps(
        parse_corpus(corpus), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def offline_run(corpus, query, settings, limit, inspector=None):
    if settings.mode != "lexical-offline":
        raise ValueError("Offline replay cannot call providers")
    return Pipeline(
        SearchRetriever(LexicalClient(corpus), None, LexicalPlanner(settings), settings),
        IdentityReranker(),
        inspector,
        TokenCriterionScorer(),
        limit=limit,
    ).run(query)


def export_bundle(corpus, query, settings, limit, inspector=None):
    corpus = parse_corpus(corpus)
    query = parse_query(query_dict(query))
    report = offline_run(corpus, query, settings, limit, inspector)
    # Round-trip converts tuples consistently; credentials are never part of this contract.
    return json.loads(
        json.dumps(
            {
                "version": 1,
                "engine_version": __version__,
                "corpus_sha256": corpus_digest(corpus),
                "query": query_dict(query),
                "settings": {k: getattr(settings, k) for k in sorted(ALGORITHM_KEYS)},
                "limit": limit,
                "report": report.to_dict(),
            },
            allow_nan=False,
        )
    )


def replay_bundle(bundle, corpus):
    expected = {
        "version",
        "engine_version",
        "corpus_sha256",
        "query",
        "settings",
        "limit",
        "report",
    }
    if (
        not isinstance(bundle, dict)
        or set(bundle) != expected
        or type(bundle["version"]) is not int
        or bundle["version"] != 1
        or bundle["engine_version"] != __version__
    ):
        raise ValueError("Unsupported replay contract")
    if bundle["corpus_sha256"] != corpus_digest(corpus):
        raise ValueError("Replay corpus digest mismatch")
    if not isinstance(bundle["settings"], dict) or set(bundle["settings"]) != ALGORITHM_KEYS:
        raise ValueError("Invalid replay algorithm settings")
    s = Settings(**bundle["settings"])
    actual = export_bundle(corpus, parse_query(bundle["query"]), s, bundle["limit"])
    if actual["report"] != bundle["report"]:
        raise ValueError("Recomputed result mismatch")
    return actual["report"]
