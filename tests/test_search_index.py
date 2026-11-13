import numpy as np
import pytest

from visual_search.search_index import VectorSearchIndex


def test_sklearn_index_returns_nearest_vector_first():
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    index = VectorSearchIndex("sklearn").fit(vectors)
    distances, indices = index.search(np.array([1, 0]), top_k=3)
    assert indices.tolist() == [0, 1, 2]
    assert distances[0] <= distances[1] <= distances[2]


def test_index_rejects_dimension_mismatch():
    index = VectorSearchIndex().fit(np.eye(3, dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.ones(4, dtype=np.float32), 2)


def test_index_rejects_empty_matrix():
    with pytest.raises(ValueError, match="non-empty"):
        VectorSearchIndex().fit(np.empty((0, 3), dtype=np.float32))
