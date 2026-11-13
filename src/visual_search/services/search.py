"""Query-time retrieval, hybrid ranking, object-aware search, and personalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from PIL import Image

from visual_search.config import Settings, settings
from visual_search.db.repository import FeedbackRepository
from visual_search.features import color_histogram, compose_hybrid_embedding, texture_hog
from visual_search.index_store import ImageIndexData, load_index
from visual_search.models.encoders import ClipEncoder, MetricLearningEncoder, TorchvisionImageEncoder
from visual_search.models.object_detection import ObjectCropper
from visual_search.search_index import VectorSearchIndex
from visual_search.utils.image_io import hamming_distance, perceptual_hash


@dataclass
class SearchResult:
    image_path: str
    category: str
    similarity: float
    hash_distance: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class SearchService:
    """Load the persisted collection once, then expose image/text retrieval methods."""

    def __init__(
        self,
        config: Settings = settings,
        index_data: ImageIndexData | None = None,
        feedback_repository: FeedbackRepository | None = None,
    ) -> None:
        self.config = config
        self.data = index_data or load_index(config.index_file)
        self._validate_index_manifest()
        self.repository = feedback_repository or FeedbackRepository(config.db_path)
        # Keep model construction lazy: text-only users should not download/load ResNet.
        self._encoder = None
        self.image_index = VectorSearchIndex(config.index_backend).fit(self.data.hybrid)
        self.clip_encoder: ClipEncoder | None = None
        self.clip_index: VectorSearchIndex | None = None
        if self.data.clip is not None:
            self.clip_index = VectorSearchIndex(config.index_backend).fit(self.data.clip)
        self.object_index: VectorSearchIndex | None = None
        if self.data.object_hybrid is not None:
            self.object_index = VectorSearchIndex(config.index_backend).fit(self.data.object_hybrid)
        self.object_cropper: ObjectCropper | None = None

        self._path_to_position = {path: position for position, path in enumerate(self.data.paths)}

    def _validate_index_manifest(self) -> None:
        """Prevent silent retrieval drift when an index and query encoder disagree."""
        manifest = self.data.manifest or {}
        if not manifest:
            return  # Backward compatibility for hand-created or older schema-1 indexes.

        current_encoder = (
            f"metric:{Path(self.config.metric_checkpoint).expanduser().resolve()}"
            if self.config.metric_checkpoint else f"torchvision:{self.config.encoder}"
        )
        if manifest.get("encoder_id") != current_encoder:
            raise ValueError(
                "The configured encoder does not match the saved image index. "
                "Use the same VISUAL_SEARCH_ENCODER / metric checkpoint, or rebuild the index."
            )

        saved_weights = manifest.get("feature_weights", {})
        current_weights = {
            "deep": self.config.deep_weight,
            "color": self.config.color_weight,
            "texture": self.config.texture_weight,
        }
        if saved_weights and any(
            abs(float(saved_weights.get(name, -1)) - value) > 1e-8
            for name, value in current_weights.items()
        ):
            raise ValueError(
                "Hybrid feature weights differ from the saved index. Restore the original "
                "weights or rebuild the index."
            )

    @property
    def encoder(self):
        """Load the configured image encoder on first image query, not at app startup."""
        if self._encoder is None:
            self._encoder = (
                MetricLearningEncoder(self.config.metric_checkpoint, self.config.device)
                if self.config.metric_checkpoint
                else TorchvisionImageEncoder(self.config.encoder, self.config.device)
            )
        return self._encoder

    def _profile_vector(self, user_id: str) -> np.ndarray | None:
        """Average the user's liked image vectors into a small personalization profile."""
        liked = self.repository.liked_paths(user_id)
        positions = [self._path_to_position[path] for path in liked if path in self._path_to_position]
        if not positions:
            return None
        profile = np.mean(self.data.hybrid[positions], axis=0)
        norm = float(np.linalg.norm(profile))
        return profile / max(norm, 1e-12)

    def _rerank_personalized(self, query: np.ndarray, user_id: str | None) -> np.ndarray:
        """Blend a query with a liked-items centroid; no profile means no behavior change."""
        if not user_id:
            return query
        profile = self._profile_vector(user_id)
        if profile is None or profile.shape != query.shape:
            return query
        blended = 0.85 * query + 0.15 * profile
        return blended / max(float(np.linalg.norm(blended)), 1e-12)

    def _run_vector_search(
        self,
        query: np.ndarray,
        top_k: int,
        categories: list[str] | None = None,
        exclude_low_quality: bool = False,
        exclude_near_duplicates: bool = False,
        query_hash: str | None = None,
        user_id: str | None = None,
        vector_index: VectorSearchIndex | None = None,
        category_source: list[str] | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        vector_index = vector_index or self.image_index
        allowed = set(categories or [])
        excluded = set(self.repository.disliked_paths(user_id)) if user_id else set()
        cats = category_source or self.data.categories
        total = len(self.data.paths)

        # Start with a small candidate window for speed. If category/quality/duplicate
        # filters remove too many items, expand geometrically until we have top_k hits
        # or have inspected the entire collection. This preserves nearest-neighbor order.
        candidate_k = min(total, max(top_k, top_k * 4 if (
            allowed or excluded or exclude_low_quality or exclude_near_duplicates
        ) else top_k))
        results: list[SearchResult] = []

        while candidate_k > 0:
            distances, indices = vector_index.search(query, candidate_k)
            results = []
            for distance, index in zip(distances, indices):
                if index < 0:
                    continue
                position = int(index)
                path = self.data.paths[position]
                category = cats[position]
                if allowed and category not in allowed:
                    continue
                if path in excluded:
                    continue
                if exclude_low_quality:
                    width, height, blur_score = self.data.quality[position, :3]
                    if width < 64 or height < 64 or blur_score < 35.0:
                        continue

                hash_distance = None
                if query_hash is not None:
                    hash_distance = hamming_distance(query_hash, self.data.hashes[position])
                    if exclude_near_duplicates and hash_distance <= 4:
                        continue

                results.append(
                    SearchResult(
                        image_path=path,
                        category=category,
                        similarity=float(np.clip(1.0 - distance, -1.0, 1.0)),
                        hash_distance=hash_distance,
                    )
                )
                if len(results) >= top_k:
                    break

            if len(results) >= top_k or candidate_k >= total:
                break
            candidate_k = min(total, max(candidate_k + 1, candidate_k * 2))

        return results

    def _encode_image_query(
        self,
        image: Image.Image,
        object_aware: bool,
        hybrid: bool,
    ) -> tuple[np.ndarray, VectorSearchIndex]:
        """Encode one query through the same representation used during indexing."""
        query_image = image.convert("RGB")
        selected_index = self.image_index
        if object_aware:
            if self.object_index is None:
                raise RuntimeError(
                    "Object-aware vectors are missing. Rebuild the collection with "
                    "visual-search index --object-aware."
                )
            if self.object_cropper is None:
                self.object_cropper = ObjectCropper(self.config.device)
            query_image, _label = self.object_cropper.crop_primary_object(query_image)
            selected_index = self.object_index

        deep = self.encoder.encode(query_image)
        if hybrid:
            vector = compose_hybrid_embedding(
                deep,
                color_histogram(query_image),
                texture_hog(query_image),
                self.config.deep_weight,
                self.config.color_weight,
                self.config.texture_weight,
            )
        else:
            vector = deep
            if object_aware:
                if self.data.object_deep is None:
                    raise RuntimeError("Object-aware deep vectors are missing; rebuild the index.")
                selected_index = VectorSearchIndex(self.config.index_backend).fit(
                    self.data.object_deep
                )
            else:
                selected_index = VectorSearchIndex(self.config.index_backend).fit(self.data.deep)
        return vector, selected_index

    def search_image(
        self,
        image: Image.Image,
        top_k: int = 10,
        categories: list[str] | None = None,
        object_aware: bool = False,
        hybrid: bool = True,
        exclude_low_quality: bool = False,
        exclude_near_duplicates: bool = False,
        personalize_for: str | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        query, selected_index = self._encode_image_query(image, object_aware, hybrid)
        if personalize_for and hybrid and not object_aware:
            query = self._rerank_personalized(query, personalize_for)

        results = self._run_vector_search(
            query=query,
            top_k=top_k,
            categories=categories,
            exclude_low_quality=exclude_low_quality,
            exclude_near_duplicates=exclude_near_duplicates,
            query_hash=perceptual_hash(image),
            user_id=personalize_for,
            vector_index=selected_index,
        )
        self.repository.record_search(
            personalize_for or "anonymous", "image", "uploaded image", len(results)
        )
        return results

    def search_images(
        self,
        images: list[Image.Image],
        top_k: int = 10,
        categories: list[str] | None = None,
        object_aware: bool = False,
        hybrid: bool = True,
        personalize_for: str | None = None,
    ) -> list[SearchResult]:
        """Blend multiple references by averaging their embeddings before retrieval.

        This supports a user who has several examples of a style or product. Averaging
        in embedding space is more stable than averaging raw pixels, which would often
        create an image that does not correspond to any real object.
        """
        if not images:
            raise ValueError("Provide at least one reference image.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        vectors = []
        selected_index = self.image_index
        for image in images:
            vector, selected_index = self._encode_image_query(image, object_aware, hybrid)
            vectors.append(vector)

        # Normalize after averaging so query magnitude does not affect cosine distance.
        query = np.mean(np.stack(vectors, axis=0), axis=0)
        query /= max(float(np.linalg.norm(query)), 1e-12)
        if personalize_for and hybrid and not object_aware:
            query = self._rerank_personalized(query, personalize_for)

        results = self._run_vector_search(
            query=query,
            top_k=top_k,
            categories=categories,
            user_id=personalize_for,
            vector_index=selected_index,
        )
        self.repository.record_search(
            personalize_for or "anonymous", "multi-image", f"{len(images)} reference images", len(results)
        )
        return results

    def search_text(
        self,
        text: str,
        top_k: int = 10,
        categories: list[str] | None = None,
        user_id: str | None = None,
    ) -> list[SearchResult]:
        manifest = self.data.manifest or {}
        if self.data.clip is not None and manifest.get("clip_model_name") not in {
            None, self.config.clip_model_name
        }:
            raise RuntimeError(
                "The configured CLIP model does not match the saved CLIP image vectors. "
                "Restore VISUAL_SEARCH_CLIP_MODEL or rebuild with --with-clip."
            )
        if self.clip_index is None:
            raise RuntimeError(
                "This image index has no CLIP vectors. Rebuild with: "
                "visual-search index --with-clip"
            )
        if self.clip_encoder is None:
            self.clip_encoder = ClipEncoder(self.config.clip_model_name, self.config.device)
        query = self.clip_encoder.encode_text(text)
        results = self._run_vector_search(
            query=query,
            top_k=top_k,
            categories=categories,
            vector_index=self.clip_index,
        )
        self.repository.record_search(user_id or "anonymous", "text", text, len(results))
        return results

    def find_near_duplicates(self, image: Image.Image, max_distance: int = 8) -> list[SearchResult]:
        query_hash = perceptual_hash(image)
        matches: list[SearchResult] = []
        for idx, candidate_hash in enumerate(self.data.hashes):
            distance = hamming_distance(query_hash, candidate_hash)
            if distance <= max_distance:
                matches.append(
                    SearchResult(
                        self.data.paths[idx], self.data.categories[idx],
                        1.0 - distance / max(1, len(query_hash)), distance
                    )
                )
        return sorted(matches, key=lambda item: item.hash_distance or 0)

    def record_feedback(self, user_id: str, result_path: str, action: str) -> None:
        self.repository.record_feedback(user_id, result_path, action)
