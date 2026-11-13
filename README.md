# Visual Discovery Engine

A modular, locally runnable image-retrieval project built with **Streamlit, PyTorch, Torchvision, OpenCV, Scikit-learn, NumPy, and SQLite**. It starts as an image-to-image search engine and can grow into a multi-modal visual discovery application.

## What is included

### Implemented core

- **CNN embeddings:** pretrained ResNet50 by default; MobileNetV3-Large and EfficientNet-B0 are configurable alternatives.
- **Hybrid visual descriptors:** normalized CNN features + HSV color histogram + HOG texture/shape features.
- **Nearest-neighbor retrieval:** cosine search using Scikit-learn, with optional FAISS inner-product search.
- **Streamlit UI:** single-image and blended multi-image search, text search, category filters, quality and duplicate controls, result gallery, and index management.
- **Index persistence:** versioned compressed NPZ files with a model/weight manifest, aligned image metadata CSV, and atomic writes.
- **Image utilities:** EXIF correction, safe decoding, blur/brightness checks, perceptual hashing, and object crops.
- **Evaluation:** Precision@K, Recall@K, and mAP@K using folder categories as starter relevance labels; an encoder-comparison script reports build time and retrieval metrics.
- **Tests:** unit tests for feature fusion, hashing, vector search, index persistence, SQLite feedback, and retrieval metrics.

### Advanced extensions included

| Extension | Implementation | Optional requirement |
|---|---|---|
| Text-to-image search | CLIP shared image/text embedding space | `pip install -e '.[clip]'`, then rebuild with `--with-clip` |
| Object-aware search | Faster R-CNN selects a primary object crop; stores a parallel object index | Build with `--object-aware` |
| Hybrid similarity | Weighted CNN + HSV + HOG vectors | Included in the default index |
| Duplicate detection | DCT perceptual hash + Hamming distance | Included |
| Automatic image grouping | KMeans clusters + PCA visualization | Included |
| Personalization | SQLite likes/dislikes; liked-image centroid blended into query vector | Included |
| Retrieval explanation | Similarity-conditioned Grad-CAM for the ResNet50 deep branch | Included; ResNet50 index only |
| Metric learning | ResNet50 triplet-loss training and checkpoint loading | Included; needs labeled class folders |
| REST API | FastAPI image/text/feedback endpoints | `pip install -e '.[api]'` |
| Large-catalog search | FAISS flat inner-product index with optional fallback | `pip install -e '.[faiss]'` |

## Architecture at a glance

```text
                         OFFLINE INDEX BUILD
  data/images/<category>/*.jpg
              |
              v
      Safe image loading / EXIF fix
              |
      +-------+----------+------------------+
      |                  |                  |
      v                  v                  v
  CNN encoder       HSV histogram        HOG descriptor
  ResNet50          128 dims              texture/shape
      |                  |                  |
      +------------------+------------------+
                         |
                         v
              Weighted hybrid embedding
                         |
        +----------------+------------------+
        |                |                  |
        v                v                  v
     NPZ index       CSV metadata     optional CLIP / object vectors
        |
        v
  sklearn NearestNeighbors or FAISS

                         QUERY TIME
  Uploaded image OR text prompt
        |                         |
        v                         v
  Same image encoder         CLIP text encoder
        |                         |
        +-------------+-----------+
                      v
               Vector retrieval
                      |
          category / quality / duplicate filters
                      |
          optional liked-items personalization
                      |
                      v
          Streamlit gallery / FastAPI JSON
                      |
             like / dislike feedback
                      |
                      v
                 SQLite store
```

### Why the project is structured this way

1. **Offline and online work are separated.** CNN embeddings are expensive to compute, so the index is built once and reused for queries.
2. **The feature extractor is shared.** Indexed images and query images go through the same model and preprocessing transform. This alignment is essential for meaningful vector comparisons.
3. **Features are fused as normalized vectors.** Each feature family is normalized, scaled by the square root of its weight, concatenated, and normalized again. Cosine similarity of the fused representation behaves like a weighted combination of the component similarities.
4. **Optional dependencies are lazy-loaded.** CLIP, FAISS, object detection, FastAPI, and Grad-CAM do not need to be imported for a basic search workflow.
5. **The index is portable.** The NPZ file holds vectors and path/category references; the source images stay in the dataset folder.

## Quick start

See **[`docs/RUN_AND_TEST.md`](docs/RUN_AND_TEST.md)** for the complete setup, first index build, all feature commands, tests, API usage, and troubleshooting.

The shortest route is:

```bash
# 1. Create and activate a virtual environment.
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

# 2. Install a compatible PyTorch/Torchvision pair for your OS and optional CUDA.
# Use https://pytorch.org/get-started/locally/ for the exact command.

# 3. Install this project.
pip install -e .

# 4. Put images into data/images/<category>/...
# 5. Build the core index.
visual-search index

# 6. Start the application.
streamlit run app.py
```

## Dataset layout

The first directory under `data/images` is used as a category label:

```text
data/images/
├── shoes/
│   ├── sneaker_001.jpg
│   └── sneaker_002.jpg
├── furniture/
│   ├── chair_001.jpg
│   └── table_001.jpg
└── pets/
    ├── cat_001.jpg
    └── dog_001.jpg
```

Images placed directly under `data/images/` receive the `uncategorized` label. Category labels are used for filtering, clustering exports, and the starter retrieval evaluation. They are not required for basic nearest-neighbor search.

## Important limitations and honest interpretation

- The baseline is **feature-based retrieval**, not a trained product-matching system. A pretrained ImageNet model may rank visually or semantically related images, but it is not guaranteed to understand fine-grained product identity.
- Similarity values are ranking scores, **not calibrated probabilities**.
- HSV and HOG features favor color/texture/shape similarity; the deep representation adds semantic structure. Tune the weights against your own judged queries.
- Folder-label evaluation is a weak proxy. If your notion of relevance is “same exact product,” “same breed,” or “same design,” create query-to-relevant-image ground-truth labels.
- The object detector can miss small, occluded, or unusual objects. Object-aware search is optional and requires an index built with object crops.
- Grad-CAM highlights query regions that influence similarity to a selected ResNet50 neighbor. It is an explanatory diagnostic, not a proof of why a human would consider two images similar.
- The local SQLite profile ID is a demo identity mechanism, not production authentication. Do not expose the local database or uploaded images publicly without adding authentication, authorization, rate limits, and privacy controls.
- Pretrained weights and datasets may have separate usage terms. Check their licenses before commercial distribution.

## License

MIT for this project code. Third-party model weights, datasets, and libraries retain their own licenses and terms. See `THIRD_PARTY_NOTICES.md`.
