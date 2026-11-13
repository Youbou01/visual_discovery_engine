"""Generate a small deterministic synthetic dataset for smoke-testing the app."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def generate(root: Path, images_per_class: int, seed: int) -> int:
    random.seed(seed)
    classes = ["demo_circles", "demo_stripes", "demo_squares"]
    palettes = [(30, 110, 220), (220, 90, 45), (50, 160, 95), (180, 80, 170)]
    total = 0
    for category in classes:
        destination = root / category
        destination.mkdir(parents=True, exist_ok=True)
        for index in range(images_per_class):
            base = random.choice([(245, 245, 240), (230, 235, 245), (248, 235, 225)])
            foreground = random.choice(palettes)
            image = Image.new("RGB", (256, 256), base)
            draw = ImageDraw.Draw(image)
            if category == "demo_circles":
                for _ in range(5 + index % 4):
                    x, y = random.randint(20, 210), random.randint(20, 210)
                    radius = random.randint(12, 35)
                    draw.ellipse((x, y, x + radius, y + radius), fill=foreground)
            elif category == "demo_squares":
                for _ in range(5 + index % 4):
                    x, y = random.randint(10, 205), random.randint(10, 205)
                    side = random.randint(18, 42)
                    draw.rectangle((x, y, x + side, y + side), fill=foreground)
            else:
                spacing = random.randint(18, 34)
                angle_shift = random.randint(-10, 10)
                for x in range(-256, 512, spacing):
                    draw.line(
                        (x + angle_shift, 0, x - angle_shift + 256, 256),
                        fill=foreground,
                        width=random.randint(4, 10),
                    )
            # Add a small, deterministic identifier mark so files are not byte-identical.
            draw.text((8, 8), f"{category[-2:]}-{index:03d}", fill=(20, 20, 20))
            image.save(destination / f"{category}_{index:03d}.jpg", quality=92)
            total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-per-class", type=int, default=12)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    count = generate(args.output, max(2, args.images_per_class), args.seed)
    print(f"Generated {count} synthetic images under {args.output}")


if __name__ == "__main__":
    main()
