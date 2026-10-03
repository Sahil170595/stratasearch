import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from stratasearch.config import Settings
from stratasearch.criterion_scorer import CriterionScorer
from stratasearch.data import load_corpus, parse_corpus, parse_query
from stratasearch.embedder import VoyageEmbedder
from stratasearch.inspector import CandidateInspector
from stratasearch.local import IdentityReranker, LexicalClient, LexicalPlanner, TokenCriterionScorer
from stratasearch.pipeline import Pipeline
from stratasearch.query_planner import QueryPlanner
from stratasearch.reranker import Reranker
from stratasearch.retriever import SearchRetriever, _rrf_merge
from stratasearch.types import Candidate, EmbedError, Filter, QueryConfig, QueryPlan, SearchHit

ROOT = Path(__file__).resolve().parents[1]


def document(doc_id, body="search latency cache", **attributes):
    return dict(
        id=doc_id,
        title="Search notebook",
        body=body,
        tags=["search"],
        kind="guide",
        topic="retrieval",
        collection="core",
        year=2025,
        **attributes,
    )


@pytest.fixture
def corpus():
    return parse_corpus(
        [
            document("doc-a"),
            {
                **document("doc-b", "search index"),
                "year": 2022,
                "kind": "experiment",
                "collection": "lab",
            },
            {**document("doc-c", "storage replication"), "title": "Search latency", "tags": []},
            {**document("doc-d", "latency cache"), "year": 2023, "topic": "storage"},
        ]
    )


def settings(**kwargs):
    return Settings(relax_threshold=3, minimum_candidates=1, **kwargs)


def offline(corpus, config, tmp_path=None):
    s = settings()
    client = LexicalClient(corpus)
    retriever = SearchRetriever(client, None, LexicalPlanner(s), s)
    inspector = CandidateInspector(tmp_path) if tmp_path else None
    report = Pipeline(
        retriever, IdentityReranker(), inspector, TokenCriterionScorer(), limit=4
    ).run(config)
    return report, client


def test_offline_entire_pipeline_and_inspection(corpus, tmp_path):
    report, client = offline(
        corpus, QueryConfig("synthetic-search", "search latency", ["cache"]), tmp_path
    )
    assert report.mode == "lexical-offline"
    assert report.status == "shortfall"
    assert {c.id for c in report.selected} == {"doc-a", "doc-d"}
    assert {c.id for c in report.rejected} == {"doc-b", "doc-c"}
    assert report.scoring_mode == "exact-token-gate"
    assert all(c.score > 0 for c in report.selected)
    artifact = json.loads((tmp_path / "synthetic-search.json").read_text())
    assert [c["id"] for c in artifact["candidates"]] == [c.id for c in report.selected]
    assert len(client.calls) >= 2


def test_relaxation_uses_final_filters_and_records_attempts(corpus):
    s = settings()
    filters = [
        Filter("collection", "Eq", "lab"),
        Filter("year", "Gte", 2025),
        Filter("kind", "Eq", "experiment"),
    ]
    client = LexicalClient(corpus)
    retriever = SearchRetriever(client, None, LexicalPlanner(s), s)
    result = retriever.search(QueryConfig("synthetic-filters", "search latency", filters=filters))
    assert [f.field for f in retriever.dropped_filters] == ["year", "kind", "collection"]
    assert client.calls[-1]["filters"] == retriever.final_filters == []
    assert len(result) >= 3


def test_rrf_one_based_missing_channel_and_stable_tie():
    a = SearchHit("a", 0.1, {})
    b = SearchHit("b", 999, {})
    c = SearchHit("c", 3, {})
    merged = _rrf_merge([b, a], [a, c], top_k=10)
    assert merged[0].id == "a"
    assert merged[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert [x.id for x in _rrf_merge([b], [a], top_k=2)] == ["a", "b"]


def test_offline_channels_compute_scores_without_embeddings(corpus):
    client = LexicalClient(corpus)
    plan = QueryPlan(None, "search latency", [], top_k=10)
    body = client.query(plan, None)
    metadata = client.query(
        QueryPlan(None, "search latency", [], top_k=10, lexical_channel="metadata"), None
    )
    assert [x.id for x in body] != [x.id for x in metadata]
    assert "doc-c" not in [x.id for x in body]
    assert "doc-c" in [x.id for x in metadata]
    with pytest.raises(ValueError, match="embedding"):
        client.query(plan, [0.1, 0.2])


def test_no_lexical_matches_fail_without_invented_candidates(corpus):
    with pytest.raises(Exception, match="only 0"):
        offline(corpus, QueryConfig("synthetic-oov", "quasar"))


def test_fixture_is_fresh_validated_and_repeatable(tmp_path):
    docs = load_corpus(ROOT / "examples" / "corpus.json")
    query = parse_query(json.loads((ROOT / "examples" / "query.json").read_text()))
    a, _ = offline(docs, query)
    b, _ = offline(docs, query)
    assert a.to_dict() == b.to_dict()
    assert len(docs) == 18
    assert a.selected


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [document("a"), document("a")],
        [{**document("a"), "year": math.nan}],
        [{**document("a"), "profile": "unsupported"}],
        [{**document("a"), "body": "x" * 4001}],
        [{**document("a"), "year": True}],
    ],
)
def test_rejects_bad_corpus(bad):
    with pytest.raises(ValueError):
        parse_corpus(bad)


@pytest.mark.parametrize(
    "field,op,value",
    [
        ("unknown", "Eq", "x"),
        ("year", "Gte", True),
        ("tags", "Eq", "cache"),
        ("year", "Gte", math.inf),
        ("kind", "In", "guide"),
    ],
)
def test_rejects_bad_filters(field, op, value):
    with pytest.raises(ValueError):
        parse_query(
            {
                "query_id": "synthetic-query",
                "description": "search",
                "filters": [{"field": field, "op": op, "value": value}],
            }
        )


def test_query_planner_provider_mock_and_neutral_schema():
    planner = QueryPlanner(settings())
    planner._call_llm = lambda _: {
        "filters": [{"field": "year", "op": "Gte", "value": 2025}],
        "unmapped": ["cache evidence"],
    }
    plan = planner.plan(QueryConfig("synthetic-query", "search", ["recent cache"]))
    assert plan.filters == [Filter("year", "Gte", 2025)]
    assert "cache evidence" in plan.vector_query
    assert QueryPlanner._validate_filter({"field": "unknown", "op": "Eq", "value": "x"}) is None


def test_planner_failed_provider_preserves_criteria_as_query_text():
    planner = QueryPlanner(settings())
    planner._call_llm = lambda _: None
    plan = planner.plan(QueryConfig("synthetic-query", "search", ["cache"]))
    assert plan.filters == []
    assert "cache" in plan.vector_query


def test_embeddings_mock_checks_dimensions_order_and_finiteness():
    class MockSDK:
        def embed(self, texts, **kwargs):
            return SimpleNamespace(embeddings=[[float(i), 1.0] for i in range(len(texts))])

    embedder = VoyageEmbedder(settings(voyage_dim=2), client=MockSDK())
    assert embedder.embed(["a", "b"]) == [[0.0, 1.0], [1.0, 1.0]]
    with pytest.raises(EmbedError):
        embedder._validate_vectors([[float("nan"), 0]], 1)
    with pytest.raises(EmbedError):
        embedder._validate_vectors([[0]], 1)


def test_reranker_mock_outputs_real_adapter_path_without_network():
    class MockSDK:
        def rerank(self, **kwargs):
            return SimpleNamespace(
                results=[
                    SimpleNamespace(index=1, relevance_score=0.8),
                    SimpleNamespace(index=0, relevance_score=0.2),
                ]
            )

    reranker = Reranker(settings(), client=MockSDK())
    ranked = reranker.rerank(
        QueryConfig("synthetic-query", "search"),
        [Candidate("a", 1, {"body": "search"}), Candidate("b", 2, {"body": "latency"})],
    )
    assert [x.id for x in ranked] == ["b", "a"]
    assert [x.rerank_score for x in ranked] == [0.8, 0.2]


def test_bad_reranker_provider_uses_visible_identity_fallback():
    class MockSDK:
        def rerank(self, **kwargs):
            return SimpleNamespace(results=[SimpleNamespace(index=0, relevance_score=math.nan)])

    reranker = Reranker(settings(), client=MockSDK())
    ranked = reranker.rerank(
        QueryConfig("synthetic-query", "search"), [Candidate("a", 1, {"body": "search"})]
    )
    assert ranked[0].rerank_score == 1
    assert reranker.last_mode == "identity-fallback"


def test_scorer_mock_and_invalid_rubric_fail_closed():
    scorer = CriterionScorer(settings())
    scorer._call_with_retries = lambda *_: {"hard": [{"pass": True}], "soft": [{"score": 3}]}
    q = QueryConfig("synthetic-query", "search", ["cache"], ["latency"])
    result = scorer.score(q, Candidate("a", 1, {"body": "cache latency"}))
    assert result.hard_pass == [True] and result.soft_scores == [3]
    scorer._call_with_retries = lambda *_: {"hard": [{"pass": True}], "soft": [{"score": True}]}
    assert scorer.score(q, Candidate("a", 1, {"body": "cache"})).hard_pass == [False]


def test_no_quota_padding_from_scorer_rejections(corpus):
    report, _ = offline(corpus, QueryConfig("synthetic-query", "search", ["quasar"]))
    assert report.selected == []
    assert report.status == "shortfall"
    assert report.rejected


def test_inspection_filename_cannot_escape_output(corpus, tmp_path):
    with pytest.raises(ValueError):
        offline(corpus, QueryConfig("../escape", "search"), tmp_path)
    assert not (tmp_path.parent / "escape.json").exists()


def test_import_does_not_load_live_clients_or_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    assert Settings().mode == "lexical-offline"
