"""Real optional Voyage embeddings; batching, bounded retries, finite validation."""

import math
import time

from .types import EmbedError


class VoyageEmbedder:
    def __init__(self, settings, client=None):
        self._settings = settings
        self._client = client

    def _get_client(self):
        if self._client is None:
            if not self._settings.voyage_api_key or not self._settings.voyage_model:
                raise EmbedError("Explicit embedding key and model required")
            from voyageai import Client

            self._client = Client(api_key=self._settings.voyage_api_key, max_retries=0, timeout=30)
        return self._client

    def embed(self, texts, input_type="query"):
        if (
            input_type not in {"query", "document"}
            or not isinstance(texts, list)
            or any(not isinstance(t, str) or not t.strip() for t in texts)
        ):
            raise EmbedError("Invalid embedding input")
        results = []
        for start in range(0, len(texts), 128):
            results.extend(self._embed_batch(texts[start : start + 128], input_type))
        return results

    def _embed_batch(self, batch, input_type):
        client = self._get_client()
        for attempt in range(4):
            try:
                response = client.embed(
                    batch,
                    model=self._settings.voyage_model,
                    input_type=input_type,
                    output_dimension=self._settings.voyage_dim,
                )
                return self._validate_vectors(response.embeddings, len(batch))
            except EmbedError:
                raise
            except Exception as exc:
                status = getattr(exc, "http_status", None)
                retryable = (
                    status == 429
                    or isinstance(status, int)
                    and 500 <= status < 600
                    or type(exc).__name__
                    in {
                        "APIConnectionError",
                        "APITimeoutError",
                        "Timeout",
                        "ConnectionError",
                        "TimeoutError",
                    }
                )
                if retryable and attempt < 3:
                    time.sleep((1, 2, 4)[attempt])
                    continue
                raise EmbedError(f"Embedding provider failed ({type(exc).__name__})") from None
        raise EmbedError("Embedding retries exhausted")

    def _validate_vectors(self, vectors, count):
        if not isinstance(vectors, list) or len(vectors) != count:
            raise EmbedError("Embedding count mismatch")
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != self._settings.voyage_dim:
                raise EmbedError("Embedding dimension mismatch")
            if any(type(v) not in {int, float} or not math.isfinite(v) for v in vector):
                raise EmbedError("Embedding values must be finite numbers")
        return [[float(v) for v in vector] for vector in vectors]
