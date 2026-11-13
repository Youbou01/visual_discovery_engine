"""Retrieval-conditioned Grad-CAM explanation for the ResNet50 deep feature branch."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


class SimilarityGradCAM:
    """Highlight query regions that contribute to similarity with a retrieved image.

    This is not a proof of semantic reasoning. It is a gradient-based diagnostic of
    the ResNet50 embedding similarity score and is available for the ResNet50 branch.
    """

    def __init__(self, device: str = "auto") -> None:
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        weights = ResNet50_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.model = resnet50(weights=weights)
        # Expose the 2048D pooled representation instead of 1000 ImageNet class logits.
        self.model.fc = torch.nn.Identity()
        self.model = self.model.eval().to(self.device)
        self.activations = None
        self.gradients = None

        def capture_activation(_module, _inputs, output):
            self.activations = output
            output.register_hook(lambda gradient: setattr(self, "gradients", gradient))

        # layer4 is the final convolutional stage before global average pooling.
        self.hook = self.model.layer4.register_forward_hook(capture_activation)

    def explain(self, query_image: Image.Image, target_embedding: np.ndarray) -> Image.Image:
        if len(target_embedding) != 2048:
            raise ValueError("Similarity Grad-CAM currently expects a 2048D ResNet50 embedding.")
        tensor = self.preprocess(query_image.convert("RGB")).unsqueeze(0).to(self.device)
        target = torch.as_tensor(target_embedding, dtype=torch.float32, device=self.device).view(1, -1)

        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            embedding = F.normalize(self.model(tensor), dim=1)
            # Use the selected neighbor's embedding as the target direction.
            score = F.cosine_similarity(embedding, target, dim=1).sum()
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True).relu()
        heatmap = F.interpolate(heatmap, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = heatmap.squeeze().detach().cpu().numpy()
        heatmap -= heatmap.min()
        heatmap /= max(float(heatmap.max()), 1e-12)

        width, height = query_image.size
        heatmap = cv2.resize(heatmap, (width, height))
        color_map = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        original = np.asarray(query_image.convert("RGB"))
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original, 0.58, color_map, 0.42, 0)
        return Image.fromarray(overlay)

    def close(self) -> None:
        self.hook.remove()
