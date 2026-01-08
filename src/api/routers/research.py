"""
Research vault API endpoints.
"""
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from typing import Optional
import re

from ..config import settings

router = APIRouter(prefix="/research", tags=["Research"])


def parse_research_readme(readme_path: Path) -> dict:
    """Parse a research session README.md for metadata."""
    if not readme_path.exists():
        return {}

    content = readme_path.read_text(encoding="utf-8")

    # Extract title (first H1)
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    title = title_match.group(1) if title_match else ""

    # Extract description (first paragraph after title)
    desc_match = re.search(r"^#\s+.+\n\n(.+?)(?:\n\n|\Z)", content, re.MULTILINE | re.DOTALL)
    description = desc_match.group(1).strip()[:300] if desc_match else ""

    # Look for tags in content
    tags_match = re.search(r"(?:tags|topics|keywords):\s*(.+)", content, re.IGNORECASE)
    tags = []
    if tags_match:
        tags = [t.strip() for t in tags_match.group(1).split(",")]

    return {
        "title": title,
        "description": description,
        "tags": tags
    }


def get_session_files(session_path: Path) -> list[dict]:
    """Get list of files in a research session."""
    files = []

    for file_path in session_path.iterdir():
        if file_path.is_file() and not file_path.name.startswith("."):
            stat = file_path.stat()
            files.append({
                "name": file_path.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        elif file_path.is_dir() and not file_path.name.startswith("."):
            # List directory contents
            subfiles = list(file_path.iterdir())
            files.append({
                "name": file_path.name,
                "type": "directory",
                "file_count": len(subfiles)
            })

    return files


@router.get("/sessions")
async def list_research_sessions(
    limit: int = 50,
    offset: int = 0
):
    """
    List all research sessions.

    Returns sessions sorted by date (newest first).
    """
    research_path = settings.research_path

    if not research_path.exists():
        return {"sessions": [], "total": 0}

    sessions = []

    for session_dir in sorted(research_path.iterdir(), reverse=True):
        if not session_dir.is_dir():
            continue
        if session_dir.name.startswith(("_", ".", "templates")):
            continue

        # Parse session name for date
        name = session_dir.name
        date_match = re.match(r"(\d{4}-\d{2}-\d{2})_(.+)", name)

        if date_match:
            date_str = date_match.group(1)
            topic = date_match.group(2).replace("-", " ").title()
        else:
            date_str = ""
            topic = name.replace("-", " ").title()

        # Get metadata from README
        readme_path = session_dir / "README.md"
        metadata = parse_research_readme(readme_path)

        sessions.append({
            "id": name,
            "date": date_str,
            "topic": topic,
            "title": metadata.get("title", topic),
            "description": metadata.get("description", ""),
            "tags": metadata.get("tags", []),
            "path": str(session_dir)
        })

    total = len(sessions)
    sessions = sessions[offset:offset + limit]

    return {
        "sessions": sessions,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.get("/sessions/{session_id}")
async def get_research_session(session_id: str):
    """
    Get detailed information about a research session.

    Returns session metadata and list of files.
    """
    session_path = settings.research_path / session_id

    if not session_path.exists() or not session_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    # Parse session name
    date_match = re.match(r"(\d{4}-\d{2}-\d{2})_(.+)", session_id)
    if date_match:
        date_str = date_match.group(1)
        topic = date_match.group(2).replace("-", " ").title()
    else:
        date_str = ""
        topic = session_id.replace("-", " ").title()

    # Get metadata
    readme_path = session_path / "README.md"
    metadata = parse_research_readme(readme_path)

    # Get files
    files = get_session_files(session_path)

    # Read main content files
    content = {}

    for filename in ["README.md", "findings.md", "sources.md"]:
        file_path = session_path / filename
        if file_path.exists():
            content[filename] = file_path.read_text(encoding="utf-8")

    return {
        "id": session_id,
        "date": date_str,
        "topic": topic,
        "title": metadata.get("title", topic),
        "description": metadata.get("description", ""),
        "tags": metadata.get("tags", []),
        "files": files,
        "content": content
    }


@router.get("/sessions/{session_id}/file/{filename:path}")
async def get_session_file(session_id: str, filename: str):
    """
    Get content of a specific file from a research session.
    """
    file_path = settings.research_path / session_id / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    # Security check - ensure path is within research folder
    try:
        file_path.resolve().relative_to(settings.research_path.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    content = file_path.read_text(encoding="utf-8")

    return {
        "filename": filename,
        "content": content,
        "size": file_path.stat().st_size
    }


@router.get("/topics")
async def list_research_topics():
    """
    Get list of all research topics/tags.

    Returns aggregated tags from all research sessions.
    """
    topics_path = settings.research_path / "topics"
    research_path = settings.research_path

    topics = {}

    # Check topics directory
    if topics_path.exists():
        for topic_file in topics_path.glob("*.md"):
            topic_name = topic_file.stem.replace("-", " ").title()
            topics[topic_name] = {
                "name": topic_name,
                "has_index": True,
                "sessions": []
            }

    # Aggregate tags from sessions
    for session_dir in research_path.iterdir():
        if not session_dir.is_dir() or session_dir.name.startswith(("_", ".", "templates", "topics")):
            continue

        readme_path = session_dir / "README.md"
        metadata = parse_research_readme(readme_path)

        for tag in metadata.get("tags", []):
            tag_title = tag.strip().title()
            if tag_title not in topics:
                topics[tag_title] = {
                    "name": tag_title,
                    "has_index": False,
                    "sessions": []
                }
            topics[tag_title]["sessions"].append(session_dir.name)

    return {
        "topics": list(topics.values()),
        "total": len(topics)
    }
