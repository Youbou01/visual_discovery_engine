import numpy as np

from visual_search.index_store import ImageIndexData, load_index, save_index


def make_index():
    return ImageIndexData(
        paths=["/images/a.jpg", "/images/b.jpg"],
        categories=["cats", "dogs"],
        deep=np.eye(2, dtype=np.float32),
        color=np.ones((2, 3), dtype=np.float32),
        texture=np.ones((2, 4), dtype=np.float32),
        hybrid=np.eye(2, dtype=np.float32),
        hashes=["0101", "1010"],
        quality=np.array([[100, 100, 55, 110], [200, 150, 80, 95]], dtype=np.float32),
        clip=np.eye(2, dtype=np.float32),
        manifest={"encoder_id": "torchvision:resnet50", "feature_weights": {"deep": 0.75}},
    )


def test_index_round_trip(tmp_path):
    destination = tmp_path / "index.npz"
    expected = make_index()
    save_index(expected, destination)
    actual = load_index(destination)
    assert actual.paths == expected.paths
    assert actual.categories == expected.categories
    np.testing.assert_allclose(actual.hybrid, expected.hybrid)
    np.testing.assert_allclose(actual.clip, expected.clip)
    assert actual.manifest == expected.manifest
    assert actual.size == 2
