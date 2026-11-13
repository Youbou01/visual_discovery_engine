"""Nearest-neighbor adapter with Scikit-learn default and optional FAISS backend."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class VectorSearchIndex:
    """Hide backend differences behind a simple top-k cosine-similarity interface."""

    def __init__(self, backend: str = "sklearn") -> None:
        self.backend_requested = backend
        self.backend = "sklearn"
        self._index = None
        self._vectors: np.ndarray | None = None

    def fit(self, vectors: np.ndarray) -> "VectorSearchIndex":
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("Expected a non-empty 2D embedding matrix.")
        # Re-normalize defensively because FAISS inner-product search assumes unit vectors.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._vectors = vectors / np.maximum(norms, 1e-12)

        if self.backend_requested in {"faiss", "auto"}:
            try:
                import faiss  # type: ignore

                self._index = faiss.IndexFlatIP(self._vectors.shape[1])
                self._index.add(self._vectors)
                self.backend = "faiss"
                return self
            except ImportError:
                if self.backend_requested == "faiss":
                    raise RuntimeError(
                        "FAISS requested but not installed. Run: pip install -e '.[faiss]'"
                    )

        self._index = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
        self._index.fit(self._vectors)
        self.backend = "sklearn"
        return self

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None or self._vectors is None:
            raise RuntimeError("Call fit() before search().")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        # Copy so normalization below cannot mutate a caller-owned or read-only array.
        vector = np.asarray(query, dtype=np.float32).reshape(1, -1).copy()
        if vector.shape[1] != self._vectors.shape[1]:
            raise ValueError("Query vector dimension does not match indexed vectors.")
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        k = min(top_k, self._vectors.shape[0])

        if self.backend == "faiss":
            similarities, indices = self._index.search(vector, k)
            return 1.0 - similarities[0], indices[0]

        distances, indices = self._index.kneighbors(vector, n_neighbors=k)
        return distances[0], indices[0]
