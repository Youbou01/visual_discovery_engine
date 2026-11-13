"""Download a small Oxford-IIIT Pet sample into category-organized image folders."""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import OxfordIIITPet

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of images to export")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "images")
    args = parser.parse_args()

    dataset = OxfordIIITPet(
        root=str(args.raw_dir),
        split="trainval",
        target_types="category",
        download=True,
    )
    count = min(max(1, args.limit), len(dataset))
    args.output.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        image, label = dataset[index]
        label_value = int(label.item()) if hasattr(label, "item") else int(label)
        category_dir = args.output / f"pet_breed_{label_value:02d}"
        category_dir.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(category_dir / f"pet_{index:06d}.jpg", quality=92)

    print(f"Exported {count} pet images into {args.output}")
    print("Check the dataset's terms before redistribution or commercial use.")


if __name__ == "__main__":
    main()
