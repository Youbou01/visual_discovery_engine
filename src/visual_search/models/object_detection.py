"""Optional object-aware crop selection using a pretrained Faster R-CNN detector."""

from __future__ import annotations

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)


class ObjectCropper:
    """Detect objects and return a crop around the most confident large object."""

    def __init__(self, device: str = "auto", confidence: float = 0.55) -> None:
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        self.confidence = confidence
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights).eval().to(self.device)
        self.transforms = weights.transforms()

    @torch.inference_mode()
    def crop_primary_object(self, image: Image.Image) -> tuple[Image.Image, str | None]:
        tensor = self.transforms(image.convert("RGB")).to(self.device)
        prediction = self.model([tensor])[0]
        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        candidates = [i for i, score in enumerate(scores) if score >= self.confidence]

        if not candidates:
            return image, None

        # Prefer a confident, reasonably large detection, avoiding tiny background objects.
        selected = max(
            candidates,
            key=lambda idx: float(scores[idx]) * float(
                (boxes[idx, 2] - boxes[idx, 0]) * (boxes[idx, 3] - boxes[idx, 1])
            ),
        )
        x1, y1, x2, y2 = [int(round(value)) for value in boxes[selected]]
        width, height = image.size
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return image, None
        return image.crop((x1, y1, x2, y2)), str(int(labels[selected]))
