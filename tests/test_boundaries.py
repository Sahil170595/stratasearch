import copy
import json
from types import SimpleNamespace

import pytest
from test_system import ROOT, document, settings

from stratasearch.cli import main
from stratasearch.config import Settings
from stratasearch.criterion_scorer import CriterionScorer
from stratasearch.data import load_corpus, parse_filter, read_json
from stratasearch.embedder import VoyageEmbedder
from stratasearch.pipeline import Pipeline
from stratasearch.query_planner import QueryPlanner
from stratasearch.replay import export_bundle, replay_bundle
from stratasearch.reranker import Reranker
from stratasearch.retriever import SearchRetriever
from stratasearch.tpuf_client import TPUFClient, _compose_filters
from stratasearch.types import (
    Candidate,
    CriterionScore,
    EmbedError,
    Filter,
    QueryConfig,
    QueryPlan,
    TPUFError,
)


def test_export_replay_recomputes_and_rejects_rewritten_result():
    docs = load_corpus(ROOT / "examples/corpus.json")
    query = QueryConfig("synthetic-replay", "search cache", ["cache"])
    s = settings()
    bundle = export_bundle(docs, query, s, 4)
    assert replay_bundle(bundle, docs) == bundle["report"]
    changed = copy.deepcopy(bundle)
    changed["report"]["selected"][0]["score"] += 1
    with pytest.raises(ValueError, match="result"):
        replay_bundle(changed, docs)
    changed = copy.deepcopy(docs)
    changed[0]["body"] += " changed"
    with pytest.raises(ValueError, match="corpus"):
        replay_bundle(bundle, changed)
    assert "api_key" not in json.dumps(bundle)


def test_cli_default_and_replay(tmp_path, capsys):
    out = tmp_path / "run.json"
    args = [
        "--corpus",
        str(ROOT / "examples/corpus.json"),
        "--query",
        str(ROOT / "examples/query.json"),
        "--output",
        str(out),
        "--limit",
        "4",
    ]
    assert main(args) == 0
    bundle = read_json(out)
    assert bundle["report"]["mode"] == "lexical-offline"
    assert bundle["report"]["selected"]
    assert main(["--corpus", str(ROOT / "examples/corpus.json"), "--replay", str(out)]) == 0
    assert "Replay verified" in capsys.readouterr().out


def test_cli_requires_separate_network_permission(capsys):
    assert main(["--live"]) == 2
    assert "allow-network" in capsys.readouterr().err


def test_sdk_adapter_score_sign_filters_and_attribute_allowlist():
    calls = []

    class Namespace:
        def query(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                rows=[{**document("note-a"), "$dist": 0.2, "vector": [1, 2], "extra": "excluded"}]
            )

    client = TPUFClient(settings(voyage_dim=2), Namespace())
    plan = QueryPlan("search", "search", [Filter("kind", "Eq", "guide")], top_k=2)
    hits = client.query(plan, [1, 2])
    assert hits[0].score == -0.2
    assert "extra" not in hits[0].attributes and "vector" not in hits[0].attributes
    assert calls[0]["filters"] == ("kind", "Eq", "guide")
    assert calls[0]["rank_by"] == ("vector", "ANN", [1, 2])
    assert calls[0]["include_attributes"] == [
        "body",
        "collection",
        "kind",
        "tags",
        "title",
        "topic",
        "year",
    ]
    assert (
        _compose_filters([Filter("kind", "Eq", "guide"), Filter("tags", "Contains", "cache")])[0]
        == "And"
    )


def test_sdk_row_and_bm25_score_from_installed_row_type():
    from turbopuffer.types.row import Row

    class Namespace:
        def query(self, **_):
            return SimpleNamespace(rows=[Row.from_dict({**document("note-a"), "$dist": 4.5})])

    hit = TPUFClient(settings(), Namespace()).query(QueryPlan(None, "search", []), None)[0]
    assert hit.score == 4.5


@pytest.mark.parametrize(
    "rows",
    [
        [{**document("note-a"), "$dist": float("nan")}],
        [{**document("note-a"), "$dist": 1}, {**document("note-a"), "$dist": 2}],
        [{**document("note-a"), "$dist": 1, "body": None}],
    ],
)
def test_sdk_bad_rows_fail_closed(rows):
    class Namespace:
        def query(self, **_):
            return SimpleNamespace(rows=rows)

    with pytest.raises(TPUFError):
        TPUFClient(settings(), Namespace()).query(QueryPlan(None, "search", []), None)


def test_count_and_no_arbitrary_search():
    class Namespace:
        def query(self, **_):
            return SimpleNamespace(aggregations={"total": 18})

    client = TPUFClient(settings(), Namespace())
    assert client.count() == 18
    with pytest.raises(TPUFError, match="No retrieval signal"):
        client.query(QueryPlan(None, None, []), None)


def test_embedding_batching_retry_and_terminal_failure(monkeypatch):
    monkeypatch.setattr("stratasearch.embedder.time.sleep", lambda _: None)

    class RateLimit(Exception):
        http_status = 429

    class SDK:
        calls = []

        def embed(self, texts, **kwargs):
            self.calls.append(len(texts))
            if len(self.calls) == 1:
                raise RateLimit("do not disclose response")
            return SimpleNamespace(embeddings=[[1, 2] for _ in texts])

    sdk = SDK()
    assert len(VoyageEmbedder(settings(voyage_dim=2), sdk).embed(["search"] * 129)) == 129
    assert sdk.calls == [128, 128, 1]

    class Denied(Exception):
        http_status = 401

    class BadSDK:
        def embed(self, *_args, **_kwargs):
            raise Denied("sensitive transport content")

    with pytest.raises(EmbedError) as exc:
        VoyageEmbedder(settings(), BadSDK()).embed(["search"])
    assert "sensitive" not in str(exc.value)


def test_full_provider_pipeline_with_injected_clients(tmp_path):
    class Namespace:
        def query(self, **kwargs):
            return SimpleNamespace(rows=[{**document("note-a"), "$dist": 0.2, "$score": 2.0}])

    class SDK:
        def embed(self, texts, **kwargs):
            return SimpleNamespace(embeddings=[[1, 2] for _ in texts])

        def rerank(self, **kwargs):
            return SimpleNamespace(results=[SimpleNamespace(index=0, relevance_score=0.8)])

    s = Settings(mode="provider-live", voyage_dim=2, minimum_candidates=1)
    planner = QueryPlanner(s)
    planner._call_llm = lambda _: {"filters": [], "unmapped": ["cache"]}
    scorer = CriterionScorer(s)
    scorer._call_with_retries = lambda *_: {"hard": [{"pass": True}], "soft": []}
    r = SearchRetriever(TPUFClient(s, Namespace()), VoyageEmbedder(s, SDK()), planner, s)
    report = Pipeline(r, Reranker(s, SDK()), scorer=scorer, limit=1).run(
        QueryConfig("synthetic-provider", "search", ["cache"])
    )
    assert report.mode == "provider-live"
    assert report.reranking_mode == "provider-reranker"
    assert report.selected[0].rerank_score == 0.8


def test_bad_scorer_array_counts_cannot_bypass_hard_gate(corpus):
    from stratasearch.local import IdentityReranker, LexicalClient, LexicalPlanner

    class BrokenScorer:
        def score_batch(self, q, candidates):
            return [CriterionScore(c.id, [], [], {}) for c in candidates]

    s = settings()
    report = Pipeline(
        SearchRetriever(LexicalClient(corpus), None, LexicalPlanner(s), s),
        IdentityReranker(),
        scorer=BrokenScorer(),
    ).run(QueryConfig("synthetic-malformed", "search", ["cache"]))
    assert not report.selected and report.rejected


def test_json_duplicate_and_nonfinite_rejected(tmp_path):
    path = tmp_path / "input.json"
    for body in ['{"a":1,"a":2}', '{"year":NaN}', '{"year":Infinity}']:
        path.write_text(body)
        with pytest.raises(ValueError):
            read_json(path)


@pytest.mark.parametrize(
    "value",
    [{"field": [], "op": "Eq", "value": "x"}, {"field": "kind", "op": [], "value": "guide"}],
)
def test_unhashable_filter_inputs_are_actionable(value):
    with pytest.raises(ValueError):
        parse_filter(value)


def test_rich_query_bonus_and_duplicate_rerank_indices():
    c = Candidate("note-a", 1, document("note-a"))
    q = QueryConfig("synthetic-query", "search", ["cache"], ["latency"])
    assert "Nice-to-have" in Reranker._build_query(q, soft_on=True)
    assert 0 < Reranker._evidence_bonus(c, q.hard_criteria) <= 0.05
    assert "Tags:" in Reranker._build_rich_document(c)

    class SDK:
        def rerank(self, **_):
            return SimpleNamespace(results=[SimpleNamespace(index=0, relevance_score=1)] * 2)

    r = Reranker(settings(), SDK())
    assert len(r.rerank(q, [c, Candidate("note-b", 2, document("note-b"))])) == 2
    assert r.last_mode == "identity-fallback"


def test_scorer_batch_preserves_order_and_all_fail_state():
    scorer = CriterionScorer(settings())

    def response(summary, *_):
        if summary == "missing":
            raise ValueError("synthetic worker failure")
        return {"hard": [{"pass": "true"}], "soft": [{"score": 2.0}]}

    scorer._call_with_retries = response
    candidates = [
        Candidate("note-a", 1, {"body": "cache"}),
        Candidate("note-b", 2, {"body": "missing"}),
    ]
    scores = scorer.score_batch(
        QueryConfig("synthetic-workers", "search", ["cache"], ["latency"]), candidates
    )
    assert [s.candidate_id for s in scores] == ["note-a", "note-b"]
    assert scores[0].hard_pass == [True] and scores[0].soft_scores == [2]
    assert scores[1].hard_pass == [False] and scores[1].raw["error"]


def test_planner_real_sdk_shape_and_hyde_are_optional():
    def completion(**kwargs):
        content = (
            '{"filters":[],"unmapped":["cache"]}'
            if "response_format" in kwargs
            else "Hypothetical cache document"
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    planner = QueryPlanner(settings(hyde_enabled=True))
    planner._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=completion))
    )
    plan = planner.plan(QueryConfig("synthetic-hyde", "search", ["cache"]))
    assert "Hypothetical cache document" in plan.vector_query
    assert planner.hyde_status == "hypothetical-document-generated"
    assert planner.last_mode == "provider-filter-extraction"


def test_import_default_cli_never_imports_provider_sdks():
    import subprocess
    import sys

    code = 'import sys; import stratasearch.cli; assert not ({"voyageai","openai","turbopuffer"} & set(sys.modules))'
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_scorer_sdk_mock_retries_transport_and_parses_json(monkeypatch):
    import httpx
    from openai import APIConnectionError

    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise APIConnectionError(request=httpx.Request("POST", "/synthetic"))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"hard":[{"pass":true}],"soft":[]}')
                )
            ]
        )

    monkeypatch.setattr("stratasearch.criterion_scorer.time.sleep", lambda _: None)
    scorer = CriterionScorer(settings())
    scorer._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=completion))
    )
    result = scorer.score(
        QueryConfig("synthetic-scorer", "search", ["cache"]),
        Candidate("note-a", 1, {"body": "cache"}),
    )
    assert result.hard_pass == [True]
    assert len(calls) == 2 and calls[-1]["temperature"] == 0
