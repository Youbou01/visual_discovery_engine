"""Offline dataset ingestion and embedding-index construction."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from visual_search.config import Settings, settings
from visual_search.features import (
    color_histogram,
    compose_hybrid_embedding,
    texture_hog,
)
from visual_search.index_store import ImageIndexData, save_index
from visual_search.models.encoders import ClipEncoder, MetricLearningEncoder, TorchvisionImageEncoder
from visual_search.models.object_detection import ObjectCropper
from visual_search.utils.image_io import assess_image_quality, load_pil_image, perceptual_hash

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class IndexBuilder:
    """Scan the image tree, compute aligned feature matrices, and persist the index."""

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.encoder = (
            MetricLearningEncoder(config.metric_checkpoint, config.device)
            if config.metric_checkpoint else TorchvisionImageEncoder(config.encoder, config.device)
        )

    def build(
        self,
        with_clip: bool = False,
        object_aware: bool = False,
        skip_low_quality: bool = False,
        blur_threshold: float = 35.0,
    ) -> ImageIndexData:
        if not self.config.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset folder does not exist: {self.config.data_dir}. "
                "Create it and add images (subfolders become categories)."
            )

        paths = sorted(
            path for path in self.config.data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"No supported images found under {self.config.data_dir}")

        clip_encoder = ClipEncoder(self.config.clip_model_name, self.config.device) if with_clip else None
        cropper = ObjectCropper(self.config.device) if object_aware else None

        records: dict[str, list] = {
            "paths": [], "categories": [], "deep": [], "color": [], "texture": [],
            "hybrid": [], "hashes": [], "quality": [], "clip": [],
            "object_deep": [], "object_hybrid": [],
        }

        for path in tqdm(paths, desc="Indexing images"):
            try:
                image = load_pil_image(path)
                quality = assess_image_quality(image, blur_threshold)
                if skip_low_quality and quality.is_low_quality:
                    continue

                deep = self.encoder.encode(image)
                color = color_histogram(image)
                texture = texture_hog(image)
                hybrid = compose_hybrid_embedding(
                    deep, color, texture,
                    self.config.deep_weight,
                    self.config.color_weight,
                    self.config.texture_weight,
                )

                # Compute optional branches before appending any records. If one branch
                # fails for a file, all matrices stay row-aligned when that file is skipped.
                clip_vector = clip_encoder.encode_image(image) if clip_encoder is not None else None
                object_deep = None
                object_hybrid = None
                if cropper is not None:
                    object_image, _label = cropper.crop_primary_object(image)
                    object_deep = self.encoder.encode(object_image)
                    object_hybrid = compose_hybrid_embedding(
                        object_deep,
                        color_histogram(object_image),
                        texture_hog(object_image),
                        self.config.deep_weight,
                        self.config.color_weight,
                        self.config.texture_weight,
                    )

                records["paths"].append(str(path.resolve()))
                relative = path.relative_to(self.config.data_dir)
                # A file directly under the root belongs to the "uncategorized" group.
                records["categories"].append(
                    relative.parts[0] if len(relative.parts) > 1 else "uncategorized"
                )
                records["deep"].append(deep)
                records["color"].append(color)
                records["texture"].append(texture)
                records["hybrid"].append(hybrid)
                records["hashes"].append(perceptual_hash(image))
                records["quality"].append(
                    [quality.width, quality.height, quality.blur_score, quality.brightness]
                )
                if clip_vector is not None:
                    records["clip"].append(clip_vector)
                if object_deep is not None and object_hybrid is not None:
                    records["object_deep"].append(object_deep)
                    records["object_hybrid"].append(object_hybrid)

            except Exception as exc:  # Keep a single corrupt file from aborting a long index build.
                print(f"[skip] {path}: {exc}")

        if not records["paths"]:
            raise ValueError("No images could be indexed. Check file formats and image readability.")

        to_matrix = lambda key: np.asarray(records[key], dtype=np.float32)
        index_data = ImageIndexData(
            paths=records["paths"],
            categories=records["categories"],
            deep=to_matrix("deep"),
            color=to_matrix("color"),
            texture=to_matrix("texture"),
            hybrid=to_matrix("hybrid"),
            hashes=records["hashes"],
            quality=to_matrix("quality"),
            clip=to_matrix("clip") if records["clip"] else None,
            object_deep=to_matrix("object_deep") if records["object_deep"] else None,
            object_hybrid=to_matrix("object_hybrid") if records["object_hybrid"] else None,
            manifest={
                "encoder_id": (
                    f"metric:{Path(self.config.metric_checkpoint).expanduser().resolve()}"
                    if self.config.metric_checkpoint else f"torchvision:{self.config.encoder}"
                ),
                "feature_weights": {
                    "deep": self.config.deep_weight,
                    "color": self.config.color_weight,
                    "texture": self.config.texture_weight,
                },
                "clip_model_name": self.config.clip_model_name,
                "with_clip": clip_encoder is not None,
                "object_aware": cropper is not None,
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "images_indexed": len(records["paths"]),
            },
        )

        save_index(index_data, self.config.index_file)
        metadata = pd.DataFrame(
            {
                "image_path": index_data.paths,
                "category": index_data.categories,
                "width": index_data.quality[:, 0].astype(int),
                "height": index_data.quality[:, 1].astype(int),
                "blur_score": index_data.quality[:, 2],
                "brightness": index_data.quality[:, 3],
                "phash": index_data.hashes,
            }
        )
        self.config.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(self.config.metadata_file, index=False)
        return index_data
