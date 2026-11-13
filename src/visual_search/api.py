"""Optional REST API for image and text retrieval."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from visual_search.config import settings
from visual_search.services.search import SearchService
from visual_search.utils.image_io import load_pil_image

app = FastAPI(
    title="Visual Discovery Engine API",
    version="1.0.0",
    description="Image-to-image and CLIP text-to-image retrieval endpoints.",
)


@lru_cache(maxsize=1)
def get_service() -> SearchService:
    """Cache the service so the CNN and vector index are not reloaded per request."""
    return SearchService()


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)


class FeedbackRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    image_path: str = Field(min_length=1, max_length=2000)
    action: str = Field(pattern="^(like|dislike)$")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_exists": settings.index_file.exists(),
        "index_path": str(settings.index_file),
    }


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), top_k: int = 10):
    if top_k < 1 or top_k > 100:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 100")
    # Read at most one byte beyond the limit instead of accepting an unbounded body.
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded image exceeds the configured size limit")
    try:
        image = load_pil_image(content)
        results = get_service().search_image(image, top_k=top_k)
        return {"count": len(results), "results": [item.to_dict() for item in results]}
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/search/text")
def search_text(request: TextSearchRequest):
    try:
        results = get_service().search_text(request.query, top_k=request.top_k)
        return {"count": len(results), "results": [item.to_dict() for item in results]}
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    try:
        get_service().record_feedback(request.user_id, request.image_path, request.action)
        return {"status": "recorded"}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
