"""Streamlit front end for image search, text search, clustering, and feedback."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from visual_search.config import settings
from visual_search.db.repository import FeedbackRepository
from visual_search.explain import SimilarityGradCAM
from visual_search.index_store import load_index, save_index
from visual_search.services.clustering import cluster_collection, export_clusters
from visual_search.services.indexing import IndexBuilder
from visual_search.services.search import SearchService
from visual_search.utils.image_io import load_pil_image

st.set_page_config(
    page_title="Visual Discovery Engine",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading search index…")
def get_search_service() -> SearchService:
    """Cache heavy model and vector-index objects between Streamlit reruns."""
    return SearchService()


@st.cache_resource(show_spinner="Loading Grad-CAM explainer…")
def get_gradcam() -> SimilarityGradCAM:
    return SimilarityGradCAM(settings.device)


def image_preview(path: str, width: int = 260):
    try:
        st.image(path, use_container_width=True)
    except Exception:
        st.warning(f"Could not open image: {Path(path).name}")


def render_results(service: SearchService, results, user_id: str, columns_per_row: int = 4):
    if not results:
        st.info("No matches found. Try a different query or relax the filters.")
        return

    for row_start in range(0, len(results), columns_per_row):
        columns = st.columns(columns_per_row)
        for offset, result in enumerate(results[row_start : row_start + columns_per_row]):
            column = columns[offset]
            with column:
                image_preview(result.image_path)
                st.markdown(f"**{result.category}**")
                st.caption(f"Similarity: {result.similarity:.1%}")
                if result.hash_distance is not None:
                    st.caption(f"Perceptual-hash distance: {result.hash_distance}")
                like_col, dislike_col = st.columns(2)
                key_stub = f"{row_start + offset}_{abs(hash(result.image_path))}"
                with like_col:
                    if st.button("👍 Like", key=f"like_{key_stub}", use_container_width=True):
                        service.record_feedback(user_id, result.image_path, "like")
                        st.toast("Preference saved.")
                with dislike_col:
                    if st.button("👎 Not relevant", key=f"dislike_{key_stub}", use_container_width=True):
                        service.record_feedback(user_id, result.image_path, "dislike")
                        st.toast("Feedback saved; future searches will avoid this image.")


def get_categories(service: SearchService) -> list[str]:
    return sorted(set(service.data.categories))


st.title("🔎 Visual Discovery Engine")
st.markdown(
    "Search an image collection by appearance, describe what you want with text, "
    "explore visual clusters, and improve recommendations with feedback."
)

with st.sidebar:
    st.header("Your workspace")
    user_id = st.text_input("Local profile ID", value="guest", max_chars=100).strip() or "guest"
    st.caption("Profile IDs and feedback are stored in the local SQLite database.")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Search", "Explore collections", "History & feedback", "Manage index"],
        label_visibility="collapsed",
    )

try:
    service = get_search_service()
except (FileNotFoundError, ValueError, RuntimeError) as error:
    service = None
    st.warning("The image index is not ready yet.")
    st.code(str(error))
    st.info("Open **Manage index** after adding images to data/images, then build the index.")

if page == "Search":
    st.subheader("Search the image collection")
    if service is None:
        st.stop()

    image_mode, text_mode = st.tabs(["Image-to-image", "Text-to-image (CLIP)"])
    category_options = get_categories(service)

    with image_mode:
        upload = st.file_uploader(
            "Upload a reference image", type=["jpg", "jpeg", "png", "webp"], key="image_query"
        )
        controls = st.columns([1, 1, 1, 1])
        with controls[0]:
            top_k = st.slider("Results", 5, 30, 10, key="image_top_k")
        with controls[1]:
            hybrid = st.checkbox("Hybrid features", value=True, help="Fuse CNN, color and HOG features.")
        with controls[2]:
            object_aware = st.checkbox(
                "Object-aware", value=False, disabled=service.object_index is None,
                help="Requires rebuilding the index with --object-aware / the matching checkbox in Manage index.",
            )
        if service.object_index is None:
            st.caption("Object-aware search is unavailable until you rebuild the index with object crops enabled.")
        with controls[3]:
            personalized = st.checkbox("Personalize", value=True)

        selected_categories = st.multiselect(
            "Limit to categories (empty means all)", category_options, key="image_categories"
        )
        quality_col, duplicate_col = st.columns(2)
        with quality_col:
            exclude_low_quality = st.checkbox("Hide low-quality images", value=False)
        with duplicate_col:
            exclude_duplicates = st.checkbox("Hide near-duplicates", value=False)

        if upload:
            if len(upload.getvalue()) > settings.max_upload_mb * 1024 * 1024:
                st.error(f"Image exceeds {settings.max_upload_mb} MB.")
            else:
                try:
                    query_image = load_pil_image(upload.getvalue())
                except ValueError as error:
                    st.error(str(error))
                    query_image = None
                if query_image is not None:
                    st.image(query_image, caption="Query image", width=260)
                if query_image is not None and st.button("Find visually similar images", type="primary"):
                    try:
                        results = service.search_image(
                            query_image,
                            top_k=top_k,
                            categories=selected_categories,
                            object_aware=object_aware,
                            hybrid=hybrid,
                            exclude_low_quality=exclude_low_quality,
                            exclude_near_duplicates=exclude_duplicates,
                            personalize_for=user_id if personalized else None,
                        )
                        st.session_state["last_results"] = [item.to_dict() for item in results]
                        st.session_state["last_query_image"] = upload.getvalue()
                        st.session_state["last_query_type"] = "image"
                    except Exception as error:
                        st.error(f"Search failed: {error}")

                if query_image is not None:
                    st.divider()
                    if st.button("Find exact / near duplicates", key="find_duplicates"):
                        duplicates = service.find_near_duplicates(query_image, max_distance=8)
                        st.session_state["duplicate_results"] = [item.to_dict() for item in duplicates]

        with st.expander("Blend multiple reference images"):
            multi_uploads = st.file_uploader(
                "Upload two or more reference images",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key="multi_image_query",
            )
            if multi_uploads:
                total_upload_bytes = sum(len(item.getvalue()) for item in multi_uploads)
                if total_upload_bytes > settings.max_upload_mb * 1024 * 1024:
                    st.error(f"Combined uploads exceed {settings.max_upload_mb} MB.")
                    multi_uploads = []
            if multi_uploads:
                preview_columns = st.columns(min(5, len(multi_uploads)))
                for number, multi_file in enumerate(multi_uploads):
                    with preview_columns[number % len(preview_columns)]:
                        st.image(multi_file, caption=multi_file.name, use_container_width=True)
            if st.button("Search from combined references", key="multi_image_search"):
                if not multi_uploads:
                    st.warning("Upload at least one image.")
                else:
                    try:
                        reference_images = [load_pil_image(item.getvalue()) for item in multi_uploads]
                        results = service.search_images(
                            reference_images,
                            top_k=top_k,
                            categories=selected_categories,
                            object_aware=object_aware,
                            hybrid=hybrid,
                            personalize_for=user_id if personalized else None,
                        )
                        st.session_state["last_results"] = [item.to_dict() for item in results]
                        st.session_state["last_query_type"] = "multi_image"
                        st.session_state.pop("last_query_image", None)
                    except Exception as error:
                        st.error(f"Multi-image search failed: {error}")

        if st.session_state.get("last_query_type") in {"image", "multi_image"}:
            st.subheader("Search results")
            from visual_search.services.search import SearchResult
            last_results = [SearchResult(**item) for item in st.session_state.get("last_results", [])]
            render_results(service, last_results, user_id)

            if (
                st.session_state.get("last_query_type") == "image"
                and last_results and settings.encoder == "resnet50"
                and not settings.metric_checkpoint
            ):
                with st.expander("Explain the top match (retrieval-conditioned Grad-CAM)"):
                    if st.button("Generate visual explanation"):
                        try:
                            query = load_pil_image(st.session_state["last_query_image"])
                            position = service._path_to_position[last_results[0].image_path]
                            explanation = get_gradcam().explain(query, service.data.deep[position])
                            st.image(explanation, caption="Highlighted query regions contributing to deep-feature similarity")
                        except Exception as error:
                            st.error(f"Explanation unavailable: {error}")

        duplicate_results = st.session_state.get("duplicate_results")
        if duplicate_results is not None:
            st.subheader("Near-duplicate matches")
            from visual_search.services.search import SearchResult
            render_results(service, [SearchResult(**item) for item in duplicate_results], user_id)

    with text_mode:
        st.write("Describe the image you want. This mode requires a CLIP-enabled index.")
        prompt = st.text_input(
            "Natural-language description",
            placeholder="e.g. a minimalist wooden chair in a bright room",
        )
        text_top_k = st.slider("Results", 5, 30, 10, key="text_top_k")
        text_categories = st.multiselect(
            "Limit to categories", category_options, key="text_categories"
        )
        if st.button("Search by description", type="primary", key="text_search"):
            if not prompt.strip():
                st.warning("Enter a description first.")
            else:
                try:
                    results = service.search_text(prompt, text_top_k, text_categories, user_id=user_id)
                    st.session_state["text_results"] = [item.to_dict() for item in results]
                    st.session_state["text_query"] = prompt
                except Exception as error:
                    st.error(str(error))
        if "text_results" in st.session_state:
            st.subheader(f"Results for: {st.session_state.get('text_query', '')}")
            from visual_search.services.search import SearchResult
            render_results(
                service,
                [SearchResult(**item) for item in st.session_state["text_results"]],
                user_id,
            )

elif page == "Explore collections":
    st.subheader("Explore visual clusters")
    if service is None:
        st.stop()
    if service.data.size < 2:
        st.info("Add at least two images before clustering the collection.")
        st.stop()
    cluster_max = min(30, service.data.size)
    cluster_count = st.slider(
        "Number of visual groups", 2, cluster_max, min(8, cluster_max)
    )
    if st.button("Generate / refresh clusters", type="primary"):
        try:
            labels, projection = cluster_collection(service.data, cluster_count)
            save_index(service.data, settings.index_file)
            export_clusters(
                service.data, labels, projection, settings.artifact_dir / "clusters.csv"
            )
            st.session_state["cluster_projection"] = projection
            st.session_state["cluster_labels"] = labels
            st.success(f"Grouped {service.data.size} images into {len(set(labels.tolist()))} clusters.")
        except Exception as error:
            st.error(str(error))

    if "cluster_projection" in st.session_state:
        frame = pd.DataFrame(
            {
                "PCA component 1": st.session_state["cluster_projection"][:, 0],
                "PCA component 2": st.session_state["cluster_projection"][:, 1],
                "Cluster": st.session_state["cluster_labels"].astype(str),
                "Category": service.data.categories,
                "Image": [Path(path).name for path in service.data.paths],
            }
        )
        st.scatter_chart(frame, x="PCA component 1", y="PCA component 2", color="Cluster")
        st.dataframe(frame, use_container_width=True, hide_index=True)
    else:
        st.info("Generate clusters to see a 2D PCA map of the indexed images.")

elif page == "History & feedback":
    st.subheader("Your searches and feedback")
    repository = FeedbackRepository(settings.db_path)
    history = repository.recent_searches(user_id, limit=30)
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    else:
        st.info("No searches recorded for this profile yet.")
    liked = repository.liked_paths(user_id)
    st.metric("Liked images", len(liked))
    if liked:
        st.caption("Liked images contribute to the optional personalized ranking profile.")
        grid = st.columns(4)
        for i, path in enumerate(liked[:20]):
            with grid[i % 4]:
                image_preview(path)
                st.caption(Path(path).name)

elif page == "Manage index":
    st.subheader("Dataset and index management")
    st.write(
        "Place images in `data/images`. A first-level subfolder becomes the image category, "
        "for example `data/images/shoes/item-001.jpg`."
    )
    upload_col, category_col = st.columns([2, 1])
    with upload_col:
        new_image = st.file_uploader(
            "Add an image to the local collection", type=["jpg", "jpeg", "png", "webp"], key="catalog_upload"
        )
    with category_col:
        new_category = st.text_input("Category folder", value="uncategorized")
    if new_image and st.button("Save image to collection"):
        safe_category = "".join(ch for ch in new_category if ch.isalnum() or ch in "-_ ").strip().replace(" ", "_")
        safe_name = Path(new_image.name).name
        destination = settings.data_dir / (safe_category or "uncategorized") / safe_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(new_image.getvalue())
        st.success(f"Saved: {destination.relative_to(settings.data_dir)}")

    data_files = [
        path for path in settings.data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ] if settings.data_dir.exists() else []
    st.metric("Images currently on disk", len(data_files))
    st.metric("Indexed images", service.data.size if service else 0)

    st.markdown("#### Index options")
    with_clip = st.checkbox("Enable CLIP text-to-image search", value=False)
    object_aware = st.checkbox("Enable object-aware embeddings", value=False)
    skip_quality = st.checkbox("Skip low-resolution / blurry images", value=False)
    blur_threshold = st.slider("Blur threshold", 5.0, 150.0, 35.0)
    if st.button("Build / rebuild search index", type="primary"):
        try:
            with st.spinner("Computing embeddings. The first run may download pretrained weights…"):
                built = IndexBuilder(settings).build(
                    with_clip=with_clip,
                    object_aware=object_aware,
                    skip_low_quality=skip_quality,
                    blur_threshold=blur_threshold,
                )
            get_search_service.clear()
            st.success(f"Successfully indexed {built.size} images.")
            st.rerun()
        except Exception as error:
            st.error(f"Index build failed: {error}")

    st.caption(
        "Tip: CLIP, object detection, and metric-learning checkpoints are optional and may require "
        "extra dependencies, model downloads, and more memory."
    )

st.divider()
st.caption(
    "Local demo architecture: PyTorch / Torchvision embeddings • OpenCV descriptors • "
    "Scikit-learn or FAISS retrieval • SQLite feedback • Streamlit UI"
)
