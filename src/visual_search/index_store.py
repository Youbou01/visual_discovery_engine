"""Versioned NPZ index persistence with atomic writes."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ImageIndexData:
    paths: list[str]
    categories: list[str]
    deep: np.ndarray
    color: np.ndarray
    texture: np.ndarray
    hybrid: np.ndarray
    hashes: list[str]
    quality: np.ndarray
    clip: np.ndarray | None = None
    object_deep: np.ndarray | None = None
    object_hybrid: np.ndarray | None = None
    cluster_labels: np.ndarray | None = None
    manifest: dict | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Ensure every feature matrix and metadata list shares the same row ordering."""
        count = len(self.paths)
        if count == 0:
            raise ValueError("An image index cannot be empty.")
        if len(self.categories) != count or len(self.hashes) != count:
            raise ValueError("Index paths, categories, and perceptual hashes must align.")
        for name in ("deep", "color", "texture", "hybrid"):
            matrix = np.asarray(getattr(self, name))
            if matrix.ndim != 2 or matrix.shape[0] != count:
                raise ValueError(f"Index matrix '{name}' must be 2D with {count} rows.")
        quality = np.asarray(self.quality)
        if quality.ndim != 2 or quality.shape[0] != count or quality.shape[1] < 4:
            raise ValueError(f"Quality metadata must have shape ({count}, 4 or more).")
        for name in ("clip", "object_deep", "object_hybrid", "cluster_labels"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape[0] != count:
                raise ValueError(f"Optional index component '{name}' must have {count} rows.")

    @property
    def size(self) -> int:
        return len(self.paths)


def save_index(index: ImageIndexData, path: str | Path) -> Path:
    """Write all vectors and aligned metadata atomically to avoid partial indexes."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": np.asarray([1], dtype=np.int32),
        "paths": np.asarray(index.paths, dtype=np.str_),
        "categories": np.asarray(index.categories, dtype=np.str_),
        "deep": np.asarray(index.deep, dtype=np.float32),
        "color": np.asarray(index.color, dtype=np.float32),
        "texture": np.asarray(index.texture, dtype=np.float32),
        "hybrid": np.asarray(index.hybrid, dtype=np.float32),
        "hashes": np.asarray(index.hashes, dtype=np.str_),
        "quality": np.asarray(index.quality, dtype=np.float32),
        "manifest_json": np.asarray([json.dumps(index.manifest or {}, sort_keys=True)]),
    }
    for name in ("clip", "object_deep", "object_hybrid", "cluster_labels"):
        value = getattr(index, name)
        if value is not None:
            payload[name] = np.asarray(value)

    # Writing through an open file handle prevents NumPy from silently appending .npz.
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", dir=destination.parent, delete=False
    ) as temporary:
        temp_path = Path(temporary.name)
        np.savez_compressed(temporary, **payload)
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temp_path, destination)
    return destination


def load_index(path: str | Path) -> ImageIndexData:
    """Load a schema-checked index without enabling unsafe pickle deserialization."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Index not found at {source}. Build it with: visual-search index"
        )
    with np.load(source, allow_pickle=False) as archive:
        version = int(archive["schema_version"][0])
        if version != 1:
            raise ValueError(f"Unsupported index schema version: {version}")
        optional = lambda name: archive[name].copy() if name in archive.files else None
        return ImageIndexData(
            paths=archive["paths"].astype(str).tolist(),
            categories=archive["categories"].astype(str).tolist(),
            deep=archive["deep"].astype(np.float32),
            color=archive["color"].astype(np.float32),
            texture=archive["texture"].astype(np.float32),
            hybrid=archive["hybrid"].astype(np.float32),
            hashes=archive["hashes"].astype(str).tolist(),
            quality=archive["quality"].astype(np.float32),
            clip=optional("clip"),
            object_deep=optional("object_deep"),
            object_hybrid=optional("object_hybrid"),
            cluster_labels=optional("cluster_labels"),
            manifest=(
                json.loads(str(archive["manifest_json"][0]))
                if "manifest_json" in archive.files else None
            ),
        )
