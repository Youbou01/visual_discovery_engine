"""Pretrained image encoders used to turn images into dense feature vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
    resnet50,
)


@dataclass
class EncoderBundle:
    model: nn.Module
    preprocess: Any
    device: torch.device
    dimension: int
    name: str


class TorchvisionImageEncoder:
    """A thin, consistent inference wrapper around torchvision classification models."""

    def __init__(self, architecture: str = "resnet50", device: str = "auto") -> None:
        self.architecture = architecture
        self.device = self._resolve_device(device)
        self.bundle = self._load_model(architecture)

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this environment.")
        return torch.device(requested)

    def _load_model(self, architecture: str) -> EncoderBundle:
        if architecture == "resnet50":
            weights = ResNet50_Weights.DEFAULT
            full_model = resnet50(weights=weights)
            # Keeping avgpool while removing fc yields a 2048-value representation.
            model = nn.Sequential(*list(full_model.children())[:-1])
            dimension = 2048
        elif architecture == "mobilenet_v3":
            weights = MobileNet_V3_Large_Weights.DEFAULT
            full_model = mobilenet_v3_large(weights=weights)
            model = nn.Sequential(full_model.features, full_model.avgpool, nn.Flatten(1))
            dimension = 960
        elif architecture == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT
            full_model = efficientnet_b0(weights=weights)
            model = nn.Sequential(full_model.features, full_model.avgpool, nn.Flatten(1))
            dimension = 1280
        else:
            raise ValueError(f"Unsupported encoder architecture: {architecture}")

        model.eval().to(self.device)
        return EncoderBundle(model, weights.transforms(), self.device, dimension, architecture)

    @torch.inference_mode()
    def encode(self, image: Image.Image) -> np.ndarray:
        """Return a unit-normalized float32 vector for one RGB PIL image."""
        tensor = self.bundle.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        vector = self.bundle.model(tensor)
        vector = torch.flatten(vector, start_dim=1)
        # Unit normalization makes cosine similarity independent of vector magnitude.
        vector = F.normalize(vector, p=2, dim=1)
        return vector.squeeze(0).cpu().numpy().astype(np.float32)


class ClipEncoder:
    """Optional CLIP image/text encoder; transformers is loaded only when requested."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLIP requires the optional dependency pack. Run: pip install -e '.[clip]'"
            ) from exc

        self.device = TorchvisionImageEncoder._resolve_device(device)
        self.model_name = model_name
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).eval().to(self.device)

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        features = self.model.get_image_features(**inputs)
        features = F.normalize(features, dim=-1)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Search text cannot be empty.")
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        features = self.model.get_text_features(**inputs)
        features = F.normalize(features, dim=-1)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

class MetricResNet50(nn.Module):
    """ResNet50 with a learnable projection head used by optional metric learning."""

    def __init__(self, embedding_dim: int = 256, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        input_dim = backbone.fc.in_features
        backbone.fc = nn.Linear(input_dim, embedding_dim)
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(images), p=2, dim=1)


class MetricLearningEncoder:
    """Inference adapter for a checkpoint produced by training/train_metric.py."""

    def __init__(self, checkpoint_path: str, device: str = "auto") -> None:
        self.device = TorchvisionImageEncoder._resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        embedding_dim = int(checkpoint.get("embedding_dim", 256))
        self.model = MetricResNet50(embedding_dim=embedding_dim, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval().to(self.device)
        self.preprocess = ResNet50_Weights.DEFAULT.transforms()
        self.dimension = embedding_dim

    @torch.inference_mode()
    def encode(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        return self.model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
