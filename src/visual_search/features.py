"""Classical visual descriptors and a weighted hybrid embedding composer."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from visual_search.config import settings
from visual_search.utils.image_io import cv2_from_pil


def _unit(vector: np.ndarray) -> np.ndarray:
    """Convert a vector to float32 unit length, safely handling a zero vector."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def color_histogram(image: Image.Image) -> np.ndarray:
    """Create a compact HSV histogram; HSV separates hue from illumination better than RGB."""
    bgr = cv2_from_pil(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 8 hue x 4 saturation x 4 value bins = 128 dimensions.
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    return _unit(cv2.normalize(hist, None).flatten())


def texture_hog(image: Image.Image) -> np.ndarray:
    """Extract HOG gradients as a texture/shape descriptor with fixed output size."""
    gray = cv2.cvtColor(cv2_from_pil(image), cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    descriptor = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    return _unit(descriptor.compute(gray).flatten())


def compose_hybrid_embedding(
    deep: np.ndarray,
    color: np.ndarray,
    texture: np.ndarray,
    deep_weight: float | None = None,
    color_weight: float | None = None,
    texture_weight: float | None = None,
) -> np.ndarray:
    """Fuse unit vectors so cosine similarity becomes a weighted sum of component similarities.

    Multiplying each unit component by sqrt(weight) is important: cosine similarity
    between concatenated vectors then contributes each component approximately in
    proportion to its configured weight. The final normalization protects against
    tiny floating-point drift when the weights do not sum to exactly one.
    """
    weights = np.asarray(
        [
            settings.deep_weight if deep_weight is None else deep_weight,
            settings.color_weight if color_weight is None else color_weight,
            settings.texture_weight if texture_weight is None else texture_weight,
        ],
        dtype=np.float32,
    )
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("Hybrid weights must be non-negative and at least one must be positive.")
    weights /= weights.sum()
    parts = [_unit(deep), _unit(color), _unit(texture)]
    weighted = [part * np.sqrt(weight) for part, weight in zip(parts, weights) if weight > 0]
    return _unit(np.concatenate(weighted).astype(np.float32))
