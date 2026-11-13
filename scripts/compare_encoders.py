"""Build/evaluate several torchvision encoders and report retrieval metrics + build time."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from visual_search.config import settings
from visual_search.services.evaluation import evaluate_retrieval
from visual_search.services.indexing import IndexBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=["resnet50", "mobilenet_v3", "efficientnet_b0"],
        choices=["resnet50", "mobilenet_v3", "efficientnet_b0"],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument(
        "--output", type=Path, default=settings.artifact_dir / "encoder_comparison.csv"
    )
    args = parser.parse_args()

    rows = []
    for encoder_name in args.encoders:
        index_path = settings.artifact_dir / f"index_{encoder_name}.npz"
        metadata_path = settings.artifact_dir / f"metadata_{encoder_name}.csv"
        config = replace(
            settings,
            encoder=encoder_name,
            metric_checkpoint=None,
            index_file=index_path,
            metadata_file=metadata_path,
        )

        started = time.perf_counter()
        data = IndexBuilder(config).build()
        elapsed = time.perf_counter() - started
        metrics = evaluate_retrieval(data, top_k=args.top_k, max_queries=args.max_queries)
        rows.append(
            {
                "encoder": encoder_name,
                "indexed_images": data.size,
                "embedding_dimension": int(data.deep.shape[1]),
                "index_build_seconds": round(elapsed, 3),
                **metrics,
                "index_path": str(index_path),
            }
        )
        print(f"{encoder_name}: {metrics}; build took {elapsed:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Comparison report written to {args.output}")


if __name__ == "__main__":
    main()
