"""Local-first CLI; a separate explicit switch authorizes optional network calls."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .criterion_scorer import CriterionScorer
from .data import load_corpus, parse_query, read_json
from .embedder import VoyageEmbedder
from .inspector import CandidateInspector
from .pipeline import Pipeline
from .query_planner import QueryPlanner
from .replay import export_bundle, query_dict, replay_bundle
from .reranker import Reranker
from .retriever import SearchRetriever
from .tpuf_client import TPUFClient
from .types import PipelineError


def main(argv=None):
    parser = argparse.ArgumentParser(prog="stratasearch")
    parser.add_argument("--corpus", default="examples/corpus.json")
    parser.add_argument("--query", default="examples/query.json")
    parser.add_argument("--output", help="Write versioned run JSON (offline runs are replayable)")
    parser.add_argument(
        "--inspect", help="Write final post-gate Markdown and JSON in this directory"
    )
    parser.add_argument("--replay", help="Recompute an offline bundle against --corpus")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--relax-threshold", type=int, default=6)
    parser.add_argument("--single-channel", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--hyde", action="store_true")
    parser.add_argument("--include-soft", action="store_true")
    parser.add_argument("--evidence-bonus", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.live and not args.allow_network:
            raise ValueError("Live mode requires --allow-network; provider calls may incur costs")
        if args.replay and (args.live or args.allow_network):
            raise ValueError("Replay is offline only")
        if not args.live and (args.hyde or args.include_soft or args.evidence_bonus):
            raise ValueError("Learned-ranking switches require --live --allow-network")
        if args.replay:
            report = replay_bundle(read_json(args.replay), load_corpus(args.corpus))
            print(f"Replay verified: {report['query_id']}, {len(report['selected'])} selected")
            return 0
        query = parse_query(read_json(args.query))
        inspector = CandidateInspector(args.inspect) if args.inspect else None
        if args.live:
            settings = replace(
                Settings.from_environment(),
                hybrid=not args.single_channel,
                relax_threshold=args.relax_threshold,
                hyde_enabled=args.hyde,
                soft_criteria=args.include_soft,
                evidence_bonus=args.evidence_bonus,
            )
            report = Pipeline(
                SearchRetriever(
                    TPUFClient(settings), VoyageEmbedder(settings), QueryPlanner(settings), settings
                ),
                Reranker(settings),
                inspector,
                CriterionScorer(settings),
                limit=args.limit,
            ).run(query)
            payload = {
                "version": 1,
                "query": query_dict(query),
                "report": report.to_dict(),
                "replayable": False,
                "reason": "External index and model responses are not frozen",
            }
        else:
            settings = Settings(
                hybrid=not args.single_channel, relax_threshold=args.relax_threshold
            )
            payload = export_bundle(
                load_corpus(args.corpus), query, settings, args.limit, inspector
            )
        text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return 0
    except (ValueError, OSError, PipelineError, TypeError) as exc:
        # Local validation messages are useful. Never print provider response bodies or keys.
        print(f"StrataSearch: {exc}", file=sys.stderr)
        return 2
