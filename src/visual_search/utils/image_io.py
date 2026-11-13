"""Safe image loading, quality checks, and preprocessing helpers."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass(frozen=True)
class ImageQuality:
    width: int
    height: int
    blur_score: float
    brightness: float
    is_low_quality: bool


def load_pil_image(source: str | Path | bytes | bytearray | io.BytesIO) -> Image.Image:
    """Load an image safely and apply EXIF orientation before converting to RGB."""
    try:
        if isinstance(source, (bytes, bytearray)):
            image = Image.open(io.BytesIO(source))
        else:
            image = Image.open(source)
        # EXIF orientation metadata is common in phone photos; normalize it once.
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc


def cv2_from_pil(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image into an OpenCV BGR uint8 array."""
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def assess_image_quality(image: Image.Image, blur_threshold: float = 35.0) -> ImageQuality:
    """Estimate blur and brightness; these are heuristics, not a quality truth label."""
    bgr = cv2_from_pil(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    width, height = image.size
    low_quality = width < 64 or height < 64 or blur_score < blur_threshold
    return ImageQuality(width, height, blur_score, brightness, low_quality)


def center_or_object_crop(
    image: Image.Image,
    box: tuple[int, int, int, int] | None = None,
    padding: float = 0.06,
) -> Image.Image:
    """Crop a detected object with a little margin, or return the original image."""
    if box is None:
        return image

    x1, y1, x2, y2 = box
    width, height = image.size
    pad_x = int(max(0, x2 - x1) * padding)
    pad_y = int(max(0, y2 - y1) * padding)
    bounds = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return image
    return image.crop(bounds)


def perceptual_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute a compact DCT perceptual hash for near-duplicate screening."""
    gray = cv2.cvtColor(cv2_from_pil(image), cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    # DCT low-frequency coefficients capture coarse image structure and ignore small edits.
    dct = cv2.dct(np.float32(resized))
    low_freq = dct[:hash_size, :hash_size]
    median = np.median(low_freq[1:, 1:])
    bits = low_freq > median
    return "".join("1" if bit else "0" for bit in bits.flatten())


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Count differing bits between two equal-length perceptual hashes."""
    if len(hash_a) != len(hash_b):
        raise ValueError("Perceptual hashes must have the same length.")
    return sum(bit_a != bit_b for bit_a, bit_b in zip(hash_a, hash_b))
