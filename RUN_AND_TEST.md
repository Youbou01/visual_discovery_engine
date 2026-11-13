# Run, test, and extend the Visual Discovery Engine

This guide covers the complete development workflow. Commands are written for the repository root.

## 1. Requirements

- Python 3.10 or newer (Python 3.11 is a good default).
- 8 GB RAM recommended for CPU use; more RAM is useful for large datasets.
- A CUDA-capable GPU is optional. The project automatically chooses CUDA when available if `VISUAL_SEARCH_DEVICE=auto`.
- Internet access on first model load, because Torchvision and optional CLIP/Faster R-CNN weights are fetched and cached locally.

### Install PyTorch and Torchvision first

Choose the install command matching your OS, Python, and CUDA runtime from the official selector:

<https://pytorch.org/get-started/locally/>

Then verify the pair:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

Torch and Torchvision releases must be compatible. Avoid copying an old CUDA-specific installation command from an unrelated tutorial.

### Install this repository

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e .
```

Install optional packs only when you need them:

```bash
pip install -e '.[clip]'       # CLIP text-to-image search
pip install -e '.[faiss]'      # FAISS vector index
pip install -e '.[api]'        # FastAPI service
pip install -e '.[dev]'       # pytest, coverage, ruff, httpx
pip install -e '.[all]'       # all optional packs except the separately installed torch pair
```

On Windows, FAISS wheels may not be available for every Python version. The default Scikit-learn backend remains fully supported; use a Conda-based FAISS installation if needed.

## 2. Add images

Create one folder per category. You can create the structure manually or use any image collection for which you have permission to use the files.

```text
data/images/
├── sneakers/
├── chairs/
├── dogs/
└── landscapes/
```

Example: put 20 or more images into at least two category folders. A minimum of a few hundred images makes the search gallery more meaningful, but the project also runs on a small collection.

The first-level directory name becomes the metadata category. The app does not scrape images from the web and does not bundle copyrighted image files.

### Optional: download a real sample dataset

```bash
python scripts/download_pet_sample.py --limit 1000
```

This uses Torchvision's Oxford-IIIT Pet dataset and exports images into breed-category folders under `data/images`. The first run downloads the source dataset. Check the dataset's terms before redistribution or commercial use.

### Optional: generate a tiny synthetic smoke-test dataset

```bash
python scripts/generate_demo_dataset.py --images-per-class 12
```

This creates simple geometric images in `data/images/demo_circles`, `demo_stripes`, and `demo_squares`. It is useful to verify the indexing and UI pipeline; it is **not** a realistic benchmark for semantic image search.

## 3. Build the base image index

```bash
visual-search index
```

Equivalent command if the editable package entry point is not available:

```bash
python -m visual_search.cli index
```

The first build downloads ResNet50 weights if they are not cached. It creates:

- `artifacts/image_index.npz` — normalized deep, color, HOG, and hybrid embeddings, paths, categories, perceptual hashes, and quality metrics.
- `artifacts/image_metadata.csv` — a human-readable image manifest.

The index is compressed and written atomically. If a build is interrupted, the old complete index is not replaced by a partially written file.

### Available index build options

```bash
# Include CLIP image vectors, enabling text-to-image search.
visual-search index --with-clip

# Include a second search representation based on detected object crops.
visual-search index --object-aware

# Build both extension indexes.
visual-search index --with-clip --object-aware

# Exclude blurry / too-small images using the heuristic quality check.
visual-search index --skip-low-quality --blur-threshold 45

# Use an image folder other than data/images.
visual-search index --data-dir /path/to/my/images
```

CLIP and object-aware vectors are optional because they increase the initial model download and indexing time. To use object-aware search later, the current index must have been built with `--object-aware` (or the matching checkbox in the UI).

## 4. Run the Streamlit application

```bash
streamlit run app.py
```

The browser interface has four sections:

### Search

- **Image-to-image:** upload a JPG/PNG/WebP image and retrieve the nearest images.
- **Multiple references:** upload a small set of examples; the app averages their embedding vectors before retrieval.
- **Hybrid features:** enable/disable the weighted deep + color + texture vector.
- **Object-aware:** searches the object-crop index if one was created.
- **Personalize:** blends the query with a profile centroid built from images you liked.
- **Hide near-duplicates:** removes results with a very small perceptual-hash distance from the query.
- **Text-to-image:** enter a natural-language prompt. This requires `pip install -e '.[clip]'` and an index built with `--with-clip`.
- Like and dislike buttons persist feedback locally. Disliked items are excluded from that user's later searches.

### Explore collections

Run KMeans over the hybrid vectors, then use PCA to create a 2D projection for visual exploration. Cluster IDs are unsupervised group labels, not semantic names. The command also writes `artifacts/clusters.csv`.

### History & feedback

See recent searches and liked images for the current profile ID. Profile IDs are local labels, not secure accounts.

### Manage index

Add an image to a category folder, see the number of images on disk, and rebuild the index with optional CLIP/object-aware/quality settings.

## 5. Use the command-line search tools

```bash
# Image-to-image search
visual-search search-image path/to/query.jpg --top-k 12

# Object-aware retrieval (requires an object-aware index)
visual-search search-image path/to/query.jpg --object-aware --top-k 12

# Personalized search based on a local feedback profile
visual-search search-image path/to/query.jpg --user guest --top-k 12

# Text search (requires CLIP index)
visual-search search-text "a red vintage sports car" --top-k 10
```

## 6. Evaluate retrieval quality

```bash
visual-search evaluate --top-k 10 --max-queries 200
```

The output includes:

- `precision_at_k`: fraction of the top-K results sharing the query's folder label.
- `recall_at_k`: fraction of same-label items retrieved in the top-K.
- `map_at_k`: mean average precision at K, rewarding relevant results near the top.

**Interpretation:** The current evaluator treats identical category names as relevant. This is a quick baseline only. For more meaningful results, create a hand-judged benchmark with a list of query images and relevant image IDs. For example, two items in a `shoes` folder can still be very different styles; conversely, images in different folders may depict the same product.

## 7. Compare CNN encoders

The project supports ResNet50, MobileNetV3-Large, and EfficientNet-B0. To build a separate index for each and compare retrieval quality/build time:

```bash
python scripts/compare_encoders.py --encoders resnet50 mobilenet_v3 efficientnet_b0 --top-k 10
```

The script writes `artifacts/encoder_comparison.csv`. It uses category folders as weak relevance labels, so interpret the metrics in the same way as the evaluation section. Each model has its own separate index file; the main app continues to use `artifacts/image_index.npz` unless its configuration is changed.

## 8. Clustering

```bash
visual-search cluster --clusters 8 --output artifacts/clusters.csv
```

The operation stores cluster labels back into the NPZ index and exports a CSV with image paths, categories, cluster assignments, and PCA coordinates. PCA coordinates are only for visualization; retrieval still uses the original embedding.

## 9. Fine-tune the embedding model with triplet loss

This is an optional advanced step. Organize the training data in at least two class subfolders, with at least two images in each class:

```text
data/images/
├── sneaker_running/
│   ├── a.jpg
│   └── b.jpg
├── sneaker_casual/
│   ├── c.jpg
│   └── d.jpg
└── boots/
    ├── e.jpg
    └── f.jpg
```

Train a ResNet50 projection head:

```bash
visual-search train-metric --epochs 5 --batch-size 8 --embedding-dim 256
```

It saves `artifacts/metric_resnet50.pt`. Configure the app/indexer to use it:

```bash
# macOS / Linux
export VISUAL_SEARCH_METRIC_CHECKPOINT=artifacts/metric_resnet50.pt

# Windows PowerShell
# $env:VISUAL_SEARCH_METRIC_CHECKPOINT = "artifacts/metric_resnet50.pt"

visual-search index
```

Rebuild the entire index after changing the encoder. Do not mix vectors created by different models or checkpoints. The embedding dimension also changes from the standard 2048D ResNet50 output to the selected projection size.

Training uses random anchor/positive/negative triplets from class folders and TripletMarginLoss. It is a reference training pipeline, not a complete production recipe: add a held-out validation set, hard-negative mining, augmentation experiments, checkpoint selection, and a relevance benchmark before treating the metric model as superior.

## 10. Start the FastAPI service

Install the optional API dependencies:

```bash
pip install -e '.[api]'
```

Run the service:

```bash
uvicorn visual_search.api:app --host 127.0.0.1 --port 8000 --reload
```

Endpoints:

- `GET /health` — index availability.
- `POST /search/image` — multipart image upload; `top_k` is a query parameter.
- `POST /search/text` — JSON body such as `{"query":"a wooden chair","top_k":10}`; requires a CLIP index.
- `POST /feedback` — JSON body such as `{"user_id":"guest","image_path":"/abs/path/item.jpg","action":"like"}`.

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

The sample API binds to localhost by default. Before exposing it to a network, add authentication, request limits, secure storage, CORS policy as required, and protections against path disclosure.

## 11. Choose a search backend

By default the project uses `sklearn.neighbors.NearestNeighbors(metric="cosine")`. This is exact brute-force search and is a good transparent baseline for small and medium collections.

To try FAISS:

```bash
pip install -e '.[faiss]'

# macOS / Linux
export VISUAL_SEARCH_INDEX_BACKEND=faiss

# Windows PowerShell
# $env:VISUAL_SEARCH_INDEX_BACKEND = "faiss"

streamlit run app.py
```

The implementation uses `IndexFlatIP` with unit-normalized vectors, so inner product is cosine similarity. It is still exact search; for much larger data, explore IVF/HNSW/quantized indexes and measure recall-vs-latency tradeoffs. `auto` tries FAISS and falls back to Scikit-learn if FAISS is not installed.

## 12. Run tests and quality checks

Install development dependencies:

```bash
pip install -e '.[dev]'
```

Run the test suite:

```bash
pytest
```

Run coverage:

```bash
pytest --cov=visual_search --cov-report=term-missing
```

Run Ruff:

```bash
ruff check .
```

Compile all Python files (does not execute model downloads):

```bash
python -m compileall -q app.py src scripts tests
```

The tests do not download pretrained models. End-to-end tests that exercise actual CNN inference should be run separately after the model weights have been cached and a dataset has been indexed.

## 13. Environment configuration

Copy `.env.example` to `.env` and adjust values as needed. Common settings:

| Variable | Default | Purpose |
|---|---|---|
| `VISUAL_SEARCH_DATA_DIR` | `data/images` | Input image tree |
| `VISUAL_SEARCH_ARTIFACT_DIR` | `artifacts` | Artifact directory |
| `VISUAL_SEARCH_DB_PATH` | `artifacts/visual_search.sqlite3` | Local feedback database |
| `VISUAL_SEARCH_DEVICE` | `auto` | `auto`, `cpu`, or a CUDA device such as `cuda:0` |
| `VISUAL_SEARCH_ENCODER` | `resnet50` | `resnet50`, `mobilenet_v3`, or `efficientnet_b0` |
| `VISUAL_SEARCH_INDEX_BACKEND` | `sklearn` | `sklearn`, `faiss`, or `auto` |
| `VISUAL_SEARCH_DEEP_WEIGHT` | `0.75` | CNN similarity weight |
| `VISUAL_SEARCH_COLOR_WEIGHT` | `0.15` | HSV histogram weight |
| `VISUAL_SEARCH_TEXTURE_WEIGHT` | `0.10` | HOG feature weight |
| `VISUAL_SEARCH_MAX_UPLOAD_MB` | `15` | UI/API upload limit |
| `VISUAL_SEARCH_METRIC_CHECKPOINT` | unset | Optional triplet-trained checkpoint |
| `VISUAL_SEARCH_CLIP_MODEL` | `openai/clip-vit-base-patch32` | Hugging Face CLIP model identifier |

Weights are normalized at feature-composition time, so changing their absolute scale does not change the relative weighting. Rebuild the index after changing the feature weights or encoder.

## 14. Troubleshooting

**`Index not found`**

Run `visual-search index` from the repository root after placing images under `data/images`.

**`No supported images found`**

Check extensions and directory paths. Supported formats include JPG, JPEG, PNG, WebP, BMP, and TIFF.

**CLIP search says the index has no CLIP vectors**

Install `.[clip]`, run `visual-search index --with-clip`, and restart the app.

**Object-aware checkbox returns no object-specific results**

Rebuild the index with `visual-search index --object-aware`. Object search requires a matching object-crop index.

**CUDA out of memory**

Set `VISUAL_SEARCH_DEVICE=cpu`; use a smaller dataset; avoid indexing CLIP and object crops at the same time; or use MobileNetV3. Restart the process after changing device settings.

**Torch/Torchvision operator mismatch**

Reinstall a compatible pair using the official PyTorch installation selector. Do not install arbitrary mismatched versions.

**FAISS cannot be installed**

Use the default Scikit-learn backend or install FAISS in a supported Conda environment.

**Search feels slow on a large collection**

Use FAISS, cache the loaded service, run on GPU for embedding extraction, and avoid retrieving every vector if you later implement metadata prefilters. For a larger production collection, use a persistent vector database or a FAISS index file and benchmark approximate search.

**Grad-CAM is unavailable**

It is available only when the configured encoder is the standard ResNet50 (not a metric-learning checkpoint), and the stored deep vectors are 2048-dimensional.

## 15. Container run (optional)

The included Dockerfile is a CPU-oriented baseline. For CUDA, start from a PyTorch image that matches the host driver/runtime and adapt the dependency install accordingly.

```bash
docker build -t visual-discovery-engine .

# Build the index with the dataset and artifacts mounted from the host.
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/artifacts:/app/artifacts" \
  visual-discovery-engine python -m visual_search.cli index

# Run the UI with the same persistent mounts.
docker run --rm -p 8501:8501 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/artifacts:/app/artifacts" \
  visual-discovery-engine
```

Open `http://localhost:8501`. Model weights are cached in the container unless you mount a persistent model-cache directory.

## 16. Suggested next engineering improvements

1. Add a judged query/relevance dataset and compare encoder/weight configurations automatically.
2. Add incremental indexing with a content hash so one new image does not require rebuilding every vector.
3. Move the image catalog and feedback store to a managed database for multi-user deployment.
4. Add authentication, authorization, deletion workflows, and private image storage before public deployment.
5. Add hard-negative mining and held-out validation to the metric-learning trainer.
6. Explore region embeddings or segmentation for multi-object photographs.
7. Compare exact and approximate FAISS indexes using recall@K and p95 latency.
