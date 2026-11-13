"""Triplet-loss training for a category-organized image dataset."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights

from visual_search.models.encoders import MetricResNet50


class TripletFolderDataset(Dataset):
    """Sample anchor/positive/negative images from class-named subdirectories."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, root: str | Path, transform, samples_per_epoch: int | None = None):
        self.root = Path(root)
        self.transform = transform
        self.by_class: dict[str, list[Path]] = defaultdict(list)
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.EXTENSIONS:
                relative = path.relative_to(self.root)
                if len(relative.parts) > 1:
                    self.by_class[relative.parts[0]].append(path)
        self.classes = [name for name, files in self.by_class.items() if len(files) >= 2]
        if len(self.classes) < 2:
            raise ValueError(
                "Metric learning needs at least two class folders, each with two or more images."
            )
        self.samples_per_epoch = samples_per_epoch or sum(map(len, self.by_class.values()))
        self.negative_classes = {name: [c for c in self.classes if c != name] for name in self.classes}

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load(self, path: Path):
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, _index: int):
        anchor_class = random.choice(self.classes)
        anchor_path, positive_path = random.sample(self.by_class[anchor_class], 2)
        negative_class = random.choice(self.negative_classes[anchor_class])
        negative_path = random.choice(self.by_class[negative_class])
        return (
            self._load(anchor_path),
            self._load(positive_path),
            self._load(negative_path),
        )


def train_metric_model(
    data_dir: str | Path,
    output_path: str | Path,
    epochs: int = 3,
    batch_size: int = 8,
    embedding_dim: int = 256,
    learning_rate: float = 1e-4,
    margin: float = 0.3,
    device: str = "auto",
) -> Path:
    """Train a compact metric embedding and save a checkpoint usable by the indexer."""
    resolved_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else
        "cpu" if device == "auto" else device
    )
    weights = ResNet50_Weights.DEFAULT
    # Mild augmentation teaches invariance to ordinary camera and orientation changes.
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )
    dataset = TripletFolderDataset(data_dir, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Triplets are randomized inside __getitem__.
        num_workers=0,
        pin_memory=resolved_device.type == "cuda",
    )

    model = MetricResNet50(embedding_dim=embedding_dim, pretrained=True).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for anchor, positive, negative in loader:
            anchor = anchor.to(resolved_device)
            positive = positive.to(resolved_device)
            negative = negative.to(resolved_device)

            optimizer.zero_grad(set_to_none=True)
            anchor_vector = model(anchor)
            positive_vector = model(positive)
            negative_vector = model(negative)
            loss = criterion(anchor_vector, positive_vector, negative_vector)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        mean_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch + 1}/{epochs} - triplet loss: {mean_loss:.4f}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "embedding_dim": embedding_dim,
            "architecture": "resnet50_triplet",
            "class_names": dataset.classes,
        },
        destination,
    )
    return destination
