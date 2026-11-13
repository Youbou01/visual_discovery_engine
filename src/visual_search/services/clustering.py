"""Unsupervised grouping and 2D projection of the indexed image collection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from visual_search.index_store import ImageIndexData, save_index


def cluster_collection(
    index_data: ImageIndexData,
    n_clusters: int = 8,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cluster labels and a 2D PCA projection for exploratory visualization."""
    if index_data.size < 2:
        raise ValueError("At least two images are required for clustering.")
    n_clusters = min(max(2, int(n_clusters)), index_data.size)
    labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(
        index_data.hybrid
    )
    # PCA is for visualization only; nearest-neighbor search stays in the original space.
    projection = PCA(n_components=2, random_state=random_state).fit_transform(index_data.hybrid)
    index_data.cluster_labels = labels.astype(np.int32)
    return labels, projection


def export_clusters(index_data: ImageIndexData, labels: np.ndarray, projection: np.ndarray, path) -> None:
    frame = pd.DataFrame(
        {
            "image_path": index_data.paths,
            "category": index_data.categories,
            "cluster": labels,
            "pca_x": projection[:, 0],
            "pca_y": projection[:, 1],
        }
    )
    frame.to_csv(path, index=False)
