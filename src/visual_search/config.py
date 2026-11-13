"""Centralized, environment-driven application configuration."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _path_from_env(name: str, default: str) -> Path:
    """Resolve a path relative to the repository unless it is already absolute."""
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True)
class Settings:
    data_dir: Path = _path_from_env("VISUAL_SEARCH_DATA_DIR", "data/images")
    artifact_dir: Path = _path_from_env("VISUAL_SEARCH_ARTIFACT_DIR", "artifacts")
    db_path: Path = _path_from_env(
        "VISUAL_SEARCH_DB_PATH", "artifacts/visual_search.sqlite3"
    )
    device: str = os.getenv("VISUAL_SEARCH_DEVICE", "auto")
    encoder: str = os.getenv("VISUAL_SEARCH_ENCODER", "resnet50")
    index_backend: str = os.getenv("VISUAL_SEARCH_INDEX_BACKEND", "sklearn")
    deep_weight: float = float(os.getenv("VISUAL_SEARCH_DEEP_WEIGHT", "0.75"))
    color_weight: float = float(os.getenv("VISUAL_SEARCH_COLOR_WEIGHT", "0.15"))
    texture_weight: float = float(os.getenv("VISUAL_SEARCH_TEXTURE_WEIGHT", "0.10"))
    max_upload_mb: int = int(os.getenv("VISUAL_SEARCH_MAX_UPLOAD_MB", "15"))
    index_file: Path = _path_from_env(
        "VISUAL_SEARCH_INDEX_FILE", "artifacts/image_index.npz"
    )
    metadata_file: Path = _path_from_env(
        "VISUAL_SEARCH_METADATA_FILE", "artifacts/image_metadata.csv"
    )
    metric_checkpoint: str | None = os.getenv("VISUAL_SEARCH_METRIC_CHECKPOINT") or None
    clip_model_name: str = os.getenv(
        "VISUAL_SEARCH_CLIP_MODEL", "openai/clip-vit-base-patch32"
    )

    def validate(self) -> None:
        """Fail early on invalid retrieval weights or unsupported configuration."""
        weights = (self.deep_weight, self.color_weight, self.texture_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Hybrid feature weights must be finite and non-negative.")
        if sum(weights) <= 0:
            raise ValueError("At least one hybrid feature weight must be positive.")
        if self.encoder not in {"resnet50", "mobilenet_v3", "efficientnet_b0"}:
            raise ValueError(
                "VISUAL_SEARCH_ENCODER must be resnet50, mobilenet_v3, or efficientnet_b0."
            )
        if self.index_backend not in {"sklearn", "faiss", "auto"}:
            raise ValueError("VISUAL_SEARCH_INDEX_BACKEND must be sklearn, faiss, or auto.")
        if self.max_upload_mb <= 0:
            raise ValueError("VISUAL_SEARCH_MAX_UPLOAD_MB must be greater than zero.")


settings = Settings()
settings.validate()
