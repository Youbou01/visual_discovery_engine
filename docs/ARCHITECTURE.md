# Architecture notes

## Package map

```text
app.py                         Streamlit presentation layer
src/visual_search/
├── config.py                  Environment-driven paths and weights
├── cli.py                     Developer and operator commands
├── api.py                     Optional FastAPI adapter
├── features.py                HSV/HOG and weighted vector fusion
├── index_store.py             Versioned NPZ persistence
├── search_index.py            sklearn / FAISS adapter
├── explain.py                 Retrieval-conditioned Grad-CAM
├── models/
│   ├── encoders.py            Torchvision, CLIP, and metric checkpoint encoders
│   └── object_detection.py    Faster R-CNN cropper
├── services/
│   ├── indexing.py            Offline embedding construction
│   ├── search.py              Query-time retrieval and personalization
│   ├── clustering.py          KMeans + PCA exploration
│   └── evaluation.py          Precision/recall/mAP evaluation
├── db/repository.py           Local SQLite search history and feedback
├── training/metric_learning.py Triplet-loss trainer
└── utils/image_io.py          Image safety, quality, crop, and pHash
```

## Retrieval representation

Let the normalized feature vectors for image `i` be:

- `d_i`: deep CNN embedding
- `c_i`: HSV color histogram
- `t_i`: HOG descriptor

Let `w_d`, `w_c`, and `w_t` be non-negative weights normalized to sum to one. The fused vector is:

```text
h_i = normalize([
    sqrt(w_d) * normalize(d_i),
    sqrt(w_c) * normalize(c_i),
    sqrt(w_t) * normalize(t_i),
])
```

Then, before the final normalization drift, the dot product between two fused vectors is the weighted sum of their component cosine similarities:

```text
h_a · h_b ≈ w_d * cos(d_a, d_b)
           + w_c * cos(c_a, c_b)
           + w_t * cos(t_a, t_b)
```

This gives one vector index and one consistent cosine search interface instead of three separate searches with a custom rank-fusion layer.

## Index consistency rules

The NPZ file includes a compact manifest with the encoder identifier, hybrid weights, CLIP model identifier, optional-branch flags, creation timestamp, and indexed-image count. At startup, the search service compares the active encoder and hybrid weights to the manifest and fails early rather than silently mixing incompatible representations.

- All query and corpus images must use the same encoder architecture, weights, transforms, color conventions, descriptor dimensions, and feature weights.
- Changing the CNN or metric-learning checkpoint requires a full re-index.
- Changing hybrid weights also requires a full re-index because the fused vectors are persisted.
- CLIP text search compares a CLIP text vector against CLIP image vectors. Do not compare CLIP vectors to ResNet vectors.
- Object-aware search compares object-crop query vectors to object-crop corpus vectors. The object-crop index is separate from the full-image index.

## Multi-image query fusion

For several reference images, the service encodes each image independently, averages the resulting vectors, and L2-normalizes the mean. This combines example images in the feature space without averaging raw pixels. It is useful for “find more like these examples” workflows; outlier references can still pull the query away from the intended concept, so users should provide coherent examples.

## Personalization

For a user with liked item embeddings `L`, the service calculates a normalized centroid `p = normalize(mean(L))` and blends it with the current image query `q`:

```text
q_personalized = normalize(0.85 * q + 0.15 * p)
```

The profile only applies to the same hybrid embedding space. Text queries and object-aware queries intentionally do not use this profile in the current implementation, preventing accidental mixing of incompatible vector spaces.

Disliked items are excluded for that profile. This is a simple, interpretable baseline, not a full collaborative-filtering recommender.

## Object-aware branch

At index-build time, Faster R-CNN detects candidate objects. The implementation chooses a high-confidence, large bounding box and crops it with a margin. Both full-image and object-crop representations are stored. At query time, the same cropper is applied and the query searches the object index.

The detector's COCO labels are not currently exposed as a semantic filter. This branch is intended to reduce background influence; it can fail when no object is detected or the primary object is not detected correctly.

## Duplicate branch

A DCT perceptual hash is calculated from a grayscale, resized image. Hamming distance counts changed bits between two hashes. Small distances identify likely near-duplicates, but the threshold is heuristic and should be calibrated for the dataset. pHash is used for duplicate screening, not as the primary semantic ranker.

## Failure and fallback strategy

- Invalid images are skipped at index-build time and rejected at query time.
- Optional modules are loaded only when their feature is requested.
- FAISS `auto` mode falls back to Scikit-learn when FAISS is unavailable.
- If object-crop vectors are absent, object-aware mode returns a clear error; the UI disables that option until an object-aware index is built.
- If CLIP vectors are absent, text search returns a clear error instructing the user to rebuild with CLIP.
