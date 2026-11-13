from dataclasses import replace

import numpy as np
from PIL import Image

from visual_search.config import settings
from visual_search.db.repository import FeedbackRepository
from visual_search.services import indexing as indexing_module
from visual_search.services import search as search_module
from visual_search.services.indexing import IndexBuilder
from visual_search.services.search import SearchService


class FakeEncoder:
    """Tiny deterministic stand-in so integration tests never download model weights."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, image):
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        mean_rgb = pixels.mean(axis=(0, 1)) / 255.0
        return mean_rgb / max(float(np.linalg.norm(mean_rgb)), 1e-12)


def test_index_build_and_image_search_without_network(tmp_path, monkeypatch):
    data_dir = tmp_path / "images"
    for category, color in [("red", (240, 20, 20)), ("green", (20, 230, 20))]:
        folder = data_dir / category
        folder.mkdir(parents=True)
        for number in range(2):
            Image.new("RGB", (128, 128), color).save(folder / f"{number}.jpg")

    config = replace(
        settings,
        data_dir=data_dir,
        artifact_dir=tmp_path / "artifacts",
        index_file=tmp_path / "artifacts" / "index.npz",
        metadata_file=tmp_path / "artifacts" / "metadata.csv",
        db_path=tmp_path / "artifacts" / "feedback.sqlite3",
        metric_checkpoint=None,
        index_backend="sklearn",
    )
    monkeypatch.setattr(indexing_module, "TorchvisionImageEncoder", FakeEncoder)
    built = IndexBuilder(config).build()
    assert built.size == 4
    assert built.hybrid.shape[0] == 4
    assert config.index_file.exists()
    assert config.metadata_file.exists()

    monkeypatch.setattr(search_module, "TorchvisionImageEncoder", FakeEncoder)
    repository = FeedbackRepository(config.db_path)
    service = SearchService(config, index_data=built, feedback_repository=repository)
    results = service.search_image(
        Image.new("RGB", (128, 128), (245, 15, 15)), top_k=2, categories=["red"]
    )
    assert len(results) == 2
    assert all(item.category == "red" for item in results)

    # A changed hybrid recipe must not silently query an index built with other weights.
    mismatched_config = replace(config, deep_weight=0.60, color_weight=0.25, texture_weight=0.15)
    try:
        SearchService(mismatched_config, index_data=built)
    except ValueError as error:
        assert "feature weights differ" in str(error).lower()
    else:
        raise AssertionError("Expected manifest validation to reject mismatched weights")

    combined_results = service.search_images(
        [
            Image.new("RGB", (128, 128), (250, 10, 10)),
            Image.new("RGB", (128, 128), (230, 25, 25)),
        ],
        top_k=2,
        categories=["red"],
    )
    assert len(combined_results) == 2
    assert all(item.category == "red" for item in combined_results)


def test_object_aware_search_requires_matching_index(tmp_path, monkeypatch):
    from visual_search.index_store import ImageIndexData

    data = ImageIndexData(
        paths=["/tmp/a.jpg"],
        categories=["a"],
        deep=np.array([[1.0, 0.0]], dtype=np.float32),
        color=np.array([[1.0, 0.0]], dtype=np.float32),
        texture=np.array([[1.0, 0.0]], dtype=np.float32),
        hybrid=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        hashes=["0"],
        quality=np.array([[128, 128, 100, 120]], dtype=np.float32),
    )
    config = replace(settings, db_path=tmp_path / "fb.sqlite", metric_checkpoint=None)
    monkeypatch.setattr(search_module, "TorchvisionImageEncoder", FakeEncoder)
    service = SearchService(config, index_data=data)
    try:
        service.search_image(Image.new("RGB", (64, 64)), object_aware=True)
    except RuntimeError as error:
        assert "object-aware vectors are missing" in str(error).lower()
    else:
        raise AssertionError("Expected explicit object-index prerequisite error")
