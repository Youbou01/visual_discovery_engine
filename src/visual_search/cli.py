"""Command-line entry point for indexing, retrieval, clustering, evaluation, and training."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from visual_search.config import settings
from visual_search.index_store import load_index, save_index
from visual_search.services.clustering import cluster_collection, export_clusters
from visual_search.services.evaluation import evaluate_retrieval
from visual_search.services.indexing import IndexBuilder
from visual_search.services.search import SearchService
from visual_search.training.metric_learning import train_metric_model
from visual_search.utils.image_io import load_pil_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visual-search",
        description="Build and query a multimodal visual image-search engine.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="Build or rebuild the image embedding database.")
    index.add_argument("--data-dir", type=Path, default=settings.data_dir)
    index.add_argument("--with-clip", action="store_true", help="Also store CLIP image embeddings.")
    index.add_argument("--object-aware", action="store_true", help="Store detected-object crop embeddings.")
    index.add_argument("--skip-low-quality", action="store_true")
    index.add_argument("--blur-threshold", type=float, default=35.0)
    index.add_argument("--metric-checkpoint", type=str, default=None)

    image = commands.add_parser("search-image", help="Search using a local image file.")
    image.add_argument("image", type=Path)
    image.add_argument("--top-k", type=int, default=10)
    image.add_argument("--object-aware", action="store_true")
    image.add_argument("--user", type=str, default=None)

    text = commands.add_parser("search-text", help="Search using a text prompt (requires CLIP index).")
    text.add_argument("query")
    text.add_argument("--top-k", type=int, default=10)

    cluster = commands.add_parser("cluster", help="Cluster images and export a 2D exploration CSV.")
    cluster.add_argument("--clusters", type=int, default=8)
    cluster.add_argument("--output", type=Path, default=settings.artifact_dir / "clusters.csv")

    evaluate = commands.add_parser("evaluate", help="Evaluate retrieval using folder categories as labels.")
    evaluate.add_argument("--top-k", type=int, default=10)
    evaluate.add_argument("--max-queries", type=int, default=200)

    train = commands.add_parser("train-metric", help="Fine-tune a ResNet50 embedding with triplet loss.")
    train.add_argument("--data-dir", type=Path, default=settings.data_dir)
    train.add_argument("--output", type=Path, default=settings.artifact_dir / "metric_resnet50.pt")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--embedding-dim", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--margin", type=float, default=0.3)
    train.add_argument("--device", default=settings.device)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "index":
        config = replace(
            settings,
            data_dir=args.data_dir.resolve(),
            metric_checkpoint=args.metric_checkpoint or settings.metric_checkpoint,
        )
        result = IndexBuilder(config).build(
            with_clip=args.with_clip,
            object_aware=args.object_aware,
            skip_low_quality=args.skip_low_quality,
            blur_threshold=args.blur_threshold,
        )
        print(f"Indexed {result.size} images -> {config.index_file}")
        if result.clip is not None:
            print(f"CLIP vectors: {result.clip.shape}")
        if result.object_hybrid is not None:
            print(f"Object-aware vectors: {result.object_hybrid.shape}")

    elif args.command == "search-image":
        service = SearchService()
        results = service.search_image(
            load_pil_image(args.image),
            top_k=args.top_k,
            object_aware=args.object_aware,
            personalize_for=args.user,
        )
        for rank, item in enumerate(results, start=1):
            print(f"{rank:>2}. {item.similarity:.4f} [{item.category}] {item.image_path}")

    elif args.command == "search-text":
        service = SearchService()
        for rank, item in enumerate(service.search_text(args.query, args.top_k), start=1):
            print(f"{rank:>2}. {item.similarity:.4f} [{item.category}] {item.image_path}")

    elif args.command == "cluster":
        data = load_index(settings.index_file)
        labels, projection = cluster_collection(data, args.clusters)
        save_index(data, settings.index_file)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export_clusters(data, labels, projection, args.output)
        print(f"Clustered {data.size} images into {len(set(labels.tolist()))} groups.")
        print(f"CSV written to {args.output}")

    elif args.command == "evaluate":
        data = load_index(settings.index_file)
        metrics = evaluate_retrieval(data, top_k=args.top_k, max_queries=args.max_queries)
        for name, value in metrics.items():
            print(f"{name}: {value}")

    elif args.command == "train-metric":
        result = train_metric_model(
            args.data_dir, args.output, args.epochs, args.batch_size,
            args.embedding_dim, args.learning_rate, args.margin, args.device,
        )
        print(f"Metric-learning checkpoint saved to {result}")
        print("Set VISUAL_SEARCH_METRIC_CHECKPOINT to this file, then rebuild the image index.")


if __name__ == "__main__":
    main()
