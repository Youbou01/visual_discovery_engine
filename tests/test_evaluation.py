import numpy as np

from visual_search.index_store import ImageIndexData
from visual_search.services.evaluation import evaluate_retrieval


def test_retrieval_metrics_for_separable_categories():
    index = ImageIndexData(
        paths=[f"{i}.jpg" for i in range(4)],
        categories=["a", "a", "b", "b"],
        deep=np.eye(4, dtype=np.float32),
        color=np.eye(4, dtype=np.float32),
        texture=np.eye(4, dtype=np.float32),
        hybrid=np.array(
            [[1, 0, 0], [0.99, 0.01, 0], [0, 1, 0], [0, 0.99, 0]],
            dtype=np.float32,
        ),
        hashes=["0"] * 4,
        quality=np.ones((4, 4), dtype=np.float32) * 100,
    )
    metrics = evaluate_retrieval(index, top_k=1, max_queries=4)
    assert metrics["queries_evaluated"] == 4
    assert metrics["precision_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["map_at_k"] == 1.0
