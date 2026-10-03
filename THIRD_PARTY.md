# Third-Party Dependencies

No SDK implementation, model weights, tokenizer assets or third-party dataset
is redistributed in this repository. Installed packages and their license
files were left untouched. Integration code imports optional dependencies;
their licenses are not a license grant for StrataSearch itself.

The inspected existing environment used:

| Dependency | Version | Verified License | Use |
|---|---|---|---|
| openai | 2.36.0 | Apache-2.0, installed distribution metadata | Optional planner/criterion-scoring SDK |
| voyageai | 0.3.7 | MIT, installed LICENSE; copyright 2023 OpenAI and 2023 VoyageAI | Optional embeddings/reranking SDK |
| turbopuffer | 1.21.0 | MIT, installed distribution metadata | Optional index-query SDK |
| pytest | 9.0.3 | MIT, installed distribution metadata | Tests only |
| ruff | 0.16.2 | MIT, installed distribution metadata | Lint/format checks only |

Their transitive dependencies are supplied and licensed by their distributions,
not copied into this repository. Preserve the respective license/NOTICE files
when redistributing any dependency; installing extras does not transfer this
repository's licensing status to them.

The fusion formula uses Reciprocal Rank Fusion, with `k=60` and one-based ranks
(Cormack, Clarke and Buettcher, SIGIR 2009). This is algorithm attribution, not a
claim of copied third-party implementation or reproduced benchmark results.
