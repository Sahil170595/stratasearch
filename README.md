# StrataSearch

An inspectable Python document-search pipeline: query planning, filter relaxation,
two-channel retrieval, reciprocal rank fusion, reranking, criterion scoring and
final local inspection. This is full source, not a browser-only implementation.

The default runtime computes **lexical features on synthetic engineering notes**.
It does not make embeddings, call an LLM, load a learned model or contact a server.
Optional adapters call real Voyage, Turbopuffer and OpenAI SDKs when explicitly
configured and authorized. Those adapters were tested with injected clients, not
live services.

[Portfolio](https://chimeraforge.vercel.app/work) |
[Interactive demo and write-up PR #65](https://github.com/Sahil170595/Banterblogs/pull/65)
(pending merge at source publication; publication does not imply deployment).

## Run Without Providers

Python 3.11+; tested with Python 3.13.1. From this checkout, no installation is
needed for the offline CLI:

```sh
python -B -m stratasearch
python -B -m stratasearch --output run.json --inspect outputs
python -B -m stratasearch --replay run.json --corpus examples/corpus.json
python -B -m stratasearch --single-channel --relax-threshold 3 --limit 2
```

Edit `examples/query.json` or pass `--query your-query.json`; pass `--corpus` for
your own authorized neutral documents. Inputs are bounded and validated before
running. Out-of-vocabulary search fails instead of inventing matches. A result
shortfall is reported without padding with documents that failed hard criteria.

Exports contain query, algorithm settings, engine version, corpus SHA-256,
channel ranks and scores, relaxation attempts, criterion judgments, selected and
rejected results. Offline replay recomputes and compares the result against the
same normalized corpus. This is integrity checking, not an authenticated audit
log; a self-consistently rewritten export is not proof of historical execution.
Live reports are explicitly not replayable because external responses are not
frozen. `--inspect` writes only the final selected, post-gate JSON/Markdown view.

## Mechanics

1. Offline planning uses the description and explicit filters, with no inferred
   semantic intent. Live planning uses a temperature-zero JSON LLM call for
   allowlisted filters, retains unmapped hard criteria in vector-query text, and
   builds a short keyword query. Failed extraction retains criteria as query text
   and reports its fallback mode. Temperature zero is not a reproducibility proof.
2. The primary channel runs with all planned filters. Below the candidate
   threshold, one filter is dropped and the search reruns. Drop order is year,
   kind/title, topic/tags, collection; input order breaks ties. **All filters,
   including explicit ones, can relax.** They are discovery hints, not immutable
   access-control or hard-policy boundaries. Put evidence requirements in hard
   criteria; those gates run later.
3. The second channel uses the **final relaxed filter set**. The primary branch
   completes before the secondary call; this is not simultaneous retrieval.
   RRF adds `1 / (60 + rank)` with one-based ranks, no contribution from an
   absent channel, and deterministic ID tie-breaking. Native scores are not
   normalized into a shared semantic scale.
4. Offline reranking is an identity pass-through, clearly recorded as such. Live
   reranking calls Voyage, optionally includes soft criteria and rich document
   fields, and can add an explicit token-evidence bonus capped at 0.05. Invalid
   or failed provider responses yield a visible identity fallback, never a
   manufactured learned score. Count and identifier invariants are checked.
5. Offline hard gates require every ASCII token of each criterion in title,
   body or tags. Soft scores are rounded token coverage on a 0-3 scale. This is
   not language understanding, entailment or LLM judgment. The live scorer uses
   one JSON LLM judgment per document, ordered batch fan-out (up to 25 workers),
   two retries on transient errors, strict rubric count/type validation and
   all-fail results on malformed/failed judgments. Only the first 50 reranked
   documents are scored by default; unscored documents are not selected.
6. Eligible results sort by hard-pass count, soft total, prior ranking score and
   ID. Local versioned output replaces any external submission contract.

The offline body feature for each unique query token `t` is:

```text
idf(t) = log(1 + (N + 1) / (df(t) + 1))
score = sum(idf(t) * tf(t)/(tf(t)+1) / sqrt(length/mean_length))
```

Lengths are floored at one. Corpus statistics are computed before filtering.
The second feature sums `3 * title_match + 2 * tag_match` per query token,
divided by the number of unique query tokens. Only positive matches enter a
channel. Neither channel is a semantic embedding or a claim of canonical BM25.
In live mode those channels are real vector ANN and provider BM25 over `body`.

## Optional Live Providers

Use an environment with the optional `providers` dependencies in
`pyproject.toml`. No keys, endpoints, namespaces, model downloads or provider
responses are shipped. Nothing reads a `.env` file automatically. Importing the
default CLI does not import SDKs or discover credentials.

Explicitly set these environment variables using your normal secret-management
workflow; never commit their values:

- `VOYAGE_API_KEY`, `OPENAI_API_KEY`, `TURBOPUFFER_API_KEY`
- `STRATASEARCH_NAMESPACE`, `STRATASEARCH_REGION`
- `STRATASEARCH_EMBED_MODEL`, `STRATASEARCH_EMBED_DIM`
- `STRATASEARCH_RERANK_MODEL`, `STRATASEARCH_PLANNER_MODEL`, `STRATASEARCH_SCORER_MODEL`

The namespace must be yours, populated with documents conforming to
`data.py`, and contain a `vector` attribute plus a full-text-indexed `body`.
Document vectors must use the same Voyage model, dimensions, normalization and
distance configuration as queries. `VoyageEmbedder.embed(texts,
input_type="document")` is retained for actual document embedding; the CLI does
not upload or index a corpus. Choose models that support the requested dimension,
chat JSON response format and temperature parameter. There are intentionally no
assumed live model defaults or migrated private namespace settings.

```sh
python -B -m stratasearch --live --allow-network --query your-query.json --output live-run.json
```

Optional switches: `--hyde`, `--include-soft`, `--evidence-bonus`.
HyDE appends a generated hypothetical technical document to query text; it is
never stored evidence. The `--corpus` file is used only by offline runs/replay;
live search reads the caller-prepared namespace. Calls may incur costs and send
your query/document text to providers. SDKs own service routing; this code does
not contain custom provider endpoint overrides. Search transport fails closed;
only planner/HyDE and reranker have the explicitly described fallbacks.

This release made **no live provider calls**. Mock tests and installed-SDK shape
checks do not verify model availability, billing, index schema, authentication,
real provider recall, or deployed behavior.

## Source Fidelity And Publication Boundary

This is a neutralized release of a staged private search prototype. Its Python
dataclasses/protocols, LLM filter-extraction and query-building methods, optional
HyDE, relaxation loop, final-filter reuse, RRF, embedding batch/retry logic,
reranker query/evidence/fallback helpers, ordered concurrent criterion scoring,
rubric parsing and JSON/Markdown inspection remain substantive code paths.

Schema-specific prompts and documents were replaced, HTTP wiring moved to
optional SDK adapters, unknown fields/invalid numbers now fail validation, and
provider-only debug sampling was removed. The external submission client and
its output contract are absent. Final inspection now follows the hard gate;
failed or unscored documents never pad a result quota. These are deliberate
changes, not claims of bit-identical output on the old system.

Only 18 freshly authored fictional engineering notes and a fresh synthetic query
are supplied. No original dataset, query set, reference answers, evaluator,
reports, response logs, organization identifiers, briefs or Git history are
included. No original benchmark or per-query-selected result is promoted as a
held-out measurement. This corpus is illustrative, not a relevance benchmark.

## Verification And Limits

On the supplied fixture, with threshold 6 and limit 4, the primary channel counts
are 3, 3, 5, 9 as year, kind and collection relax. Both final channels contain 9
matches. The cache gate leaves 5 eligible documents; 4 are selected in order:
`note-a`, `note-p`, `note-e`, `note-c`. Four retrieved documents fail the hard
criterion. These are reproducible fixture outcomes, **not accuracy numbers**.
`ready` means the requested count was met, not that quality was validated.

The release was checked with the existing installed environment (no installs):
46 tests passed, including end-to-end offline/provider-mock pipelines, provider
SDK row shapes, retry/batching, finite values, malformed rubrics, relaxation,
RRF, no quota padding, CLI network gating, export/replay and mismatch rejection.
Tests block socket connections. Ruff scoped lint/format checks are required.
The full tests use installed optional SDKs; the offline CLI needs only Python.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -m pytest tests -q -p no:cacheprovider --basetemp .test-tmp
python -B -m ruff check stratasearch tests --config pyproject.toml --no-cache
python -B -m ruff format stratasearch tests --config pyproject.toml --no-cache --check
```

Limits: ASCII-only tokenization; no semantic synonym handling offline; relaxation
can erase useful constraints; optional LLM outputs can be wrong or prompt-injected;
synthetic coverage is not held-out recall; candidate caps can miss relevant
documents; no multi-tenant authorization, production monitoring or service is
provided. Separate pipeline instances should be used for concurrent runs because
inspection state is per instance. Export/inspection writes are local files, not
transactional or authenticated storage. Publish only authorized documents and
review your own generated artifacts before sharing them.

## Licensing

No project license grant has been inferred or added: the source reviewed did not
contain a verified project license. Public visibility is not an open-source
license. The replacement fixtures were authored for this release, not derived
from the excluded dataset. Third-party SDK/tool licenses remain theirs; see
[THIRD_PARTY.md](THIRD_PARTY.md). No third-party implementation is vendored.
