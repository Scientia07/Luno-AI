"""
Progress tracking API endpoints.
"""
from fastapi import APIRouter, HTTPException
from typing import Optional

from ..models.progress import (
    ProgressSummary,
    TechnologyProgress,
    ProgressUpdate,
    ProgressUpdateResponse,
    BookmarksResponse,
    BookmarkToggleResponse,
)
from ..services.progress_service import ProgressService
from ..config import settings

router = APIRouter(prefix="/progress", tags=["Progress"])


def get_progress_service(user_id: str = "default") -> ProgressService:
    """Get progress service for user."""
    return ProgressService(user_id)


@router.get("/summary", response_model=ProgressSummary)
async def get_progress_summary(user_id: str = "default"):
    """
    Get overall progress summary.

    Returns progress statistics across all domains,
    learning paths, and recent activity.
    """
    service = get_progress_service(user_id)
    return service.get_progress_summary()


@router.get("/technology/{domain}/{tech}", response_model=TechnologyProgress)
async def get_technology_progress(domain: str, tech: str, user_id: str = "default"):
    """
    Get progress for a specific technology.

    Returns layer-by-layer completion status, notes,
    and bookmark state.
    """
    if domain not in settings.domains:
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found")

    service = get_progress_service(user_id)
    return service.get_technology_progress(domain, tech)


@router.post("/technology/{domain}/{tech}", response_model=ProgressUpdateResponse)
async def update_technology_progress(
    domain: str,
    tech: str,
    update: ProgressUpdate,
    user_id: str = "default"
):
    """
    Update progress for a technology layer.

    Mark layers as completed/incomplete and add notes.
    """
    if domain not in settings.domains:
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found")

    if not 0 <= update.layer <= 4:
        raise HTTPException(
            status_code=400,
            detail="Layer must be between 0 and 4"
        )

    service = get_progress_service(user_id)
    progress = service.update_progress(
        domain=domain,
        tech=tech,
        layer=update.layer,
        completed=update.completed,
        notes=update.notes
    )

    return ProgressUpdateResponse(success=True, progress=progress)


@router.get("/bookmarks", response_model=BookmarksResponse)
async def get_bookmarks(user_id: str = "default"):
    """
    Get all bookmarks for user.

    Returns list of bookmarked technologies and sections.
    """
    service = get_progress_service(user_id)
    bookmarks = service.get_bookmarks()
    return BookmarksResponse(bookmarks=bookmarks)


@router.post("/bookmarks/{domain}/{tech}", response_model=BookmarkToggleResponse)
async def toggle_bookmark(
    domain: str,
    tech: str,
    section: Optional[str] = None,
    title: Optional[str] = None,
    user_id: str = "default"
):
    """
    Toggle bookmark for a technology or section.

    If already bookmarked, removes the bookmark.
    If not bookmarked, adds a new bookmark.
    """
    if domain not in settings.domains:
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found")

    service = get_progress_service(user_id)
    bookmarked = service.toggle_bookmark(
        domain=domain,
        tech=tech,
        section=section,
        title=title
    )

    return BookmarkToggleResponse(bookmarked=bookmarked)


@router.delete("/bookmarks/{domain}/{tech}")
async def remove_bookmark(
    domain: str,
    tech: str,
    section: Optional[str] = None,
    user_id: str = "default"
):
    """
    Remove a specific bookmark.
    """
    if domain not in settings.domains:
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found")

    service = get_progress_service(user_id)
    # Toggle will remove if exists
    bookmarks = service.get_bookmarks()

    # Check if bookmark exists
    exists = any(
        b.domain == domain and b.tech == tech and b.section == section
        for b in bookmarks
    )

    if exists:
        service.toggle_bookmark(domain, tech, section)
        return {"success": True, "message": "Bookmark removed"}
    else:
        return {"success": False, "message": "Bookmark not found"}


@router.get("/activity")
async def get_recent_activity(user_id: str = "default", limit: int = 20):
    """
    Get recent learning activity.

    Returns list of recently completed layers.
    """
    service = get_progress_service(user_id)
    activity = service.get_recent_activity(limit=limit)
    return {"activity": [a.model_dump() for a in activity]}
