import numpy as np
from PIL import Image

from visual_search.features import color_histogram, compose_hybrid_embedding, texture_hog
from visual_search.utils.image_io import hamming_distance, perceptual_hash


def test_hybrid_embedding_is_unit_normalized():
    result = compose_hybrid_embedding(
        np.array([3.0, 4.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 2.0]),
        deep_weight=0.7,
        color_weight=0.2,
        texture_weight=0.1,
    )
    assert result.dtype == np.float32
    assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-5)


def test_hybrid_rejects_all_zero_weights():
    try:
        compose_hybrid_embedding([1], [1], [1], 0, 0, 0)
    except ValueError as error:
        assert "at least one" in str(error)
    else:
        raise AssertionError("Expected invalid weights to raise ValueError")


def test_classical_features_have_fixed_dimensions():
    image = Image.new("RGB", (192, 128), (120, 80, 40))
    assert color_histogram(image).shape == (128,)
    assert texture_hog(image).ndim == 1
    assert texture_hog(image).shape[0] > 100


def test_perceptual_hash_is_stable_for_same_image():
    image = Image.new("RGB", (96, 96), "navy")
    first = perceptual_hash(image)
    second = perceptual_hash(image.copy())
    assert first == second
    assert hamming_distance(first, second) == 0
