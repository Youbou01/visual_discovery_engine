"""Offline retrieval metrics using folder labels as weak relevance judgments."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from visual_search.index_store import ImageIndexData


def _average_precision(relevant: np.ndarray, total_relevant: int) -> float:
    """Compute AP@K with denominator min(number of relevant items, K)."""
    if not relevant.any() or total_relevant <= 0:
        return 0.0
    cumulative = np.cumsum(relevant)
    precisions = cumulative / (np.arange(len(relevant)) + 1)
    denominator = min(total_relevant, len(relevant))
    return float((precisions * relevant).sum() / denominator)


def evaluate_retrieval(
    index_data: ImageIndexData,
    top_k: int = 10,
    max_queries: int = 200,
) -> dict[str, float | int]:
    """Compute Precision@K, Recall@K, and mAP over a deterministic query sample.

    Folder/category labels are treated as relevance labels. This is convenient for
    a starter benchmark, but human judgments or product-level labels are better
    when visual similarity does not mean same category.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if max_queries < 1:
        raise ValueError("max_queries must be at least 1.")
    vectors = index_data.hybrid.astype(np.float32)
    k = min(top_k + 1, len(vectors))  # +1 because every image retrieves itself first.
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1).fit(vectors)
    candidate_count = min(max_queries, len(vectors))
    # Evenly spread queries across the index so evaluation is repeatable.
    query_indices = np.linspace(0, len(vectors) - 1, candidate_count, dtype=int)
    precisions, recalls, aps = [], [], []
    for query_idx in query_indices:
        distances, neighbors = model.kneighbors(vectors[query_idx : query_idx + 1], n_neighbors=k)
        ranked = [int(i) for i in neighbors[0] if int(i) != int(query_idx)][:top_k]
        if not ranked:
            continue
        relevant = np.asarray(
            [index_data.categories[item] == index_data.categories[query_idx] for item in ranked],
            dtype=bool,
        )
        total_relevant = (
            sum(category == index_data.categories[query_idx] for category in index_data.categories) - 1
        )
        # A singleton category has no positive counterpart, so it is not a valid query.
        if total_relevant <= 0:
            continue
        precisions.append(float(relevant.mean()))
        recalls.append(float(relevant.sum() / total_relevant))
        aps.append(_average_precision(relevant, total_relevant))

    return {
        "queries_evaluated": len(precisions),
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "map_at_k": float(np.mean(aps)) if aps else 0.0,
        "k": int(top_k),
    }
