"""
Labs API endpoints.
Interactive notebook management.
"""
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
import json
import re

from ..config import settings
from ..database.sqlite import execute_query, execute_write

router = APIRouter(prefix="/labs", tags=["Labs"])


def parse_notebook_metadata(notebook_path: Path) -> dict:
    """Extract metadata from a Jupyter notebook."""
    if not notebook_path.exists():
        return {}

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)

        cells = notebook.get("cells", [])
        metadata = notebook.get("metadata", {})

        # Get title from first markdown cell or filename
        title = notebook_path.stem.replace("-", " ").replace("_", " ").title()
        description = ""

        for cell in cells:
            if cell.get("cell_type") == "markdown":
                source = "".join(cell.get("source", []))
                # Look for H1 header
                title_match = re.search(r"^#\s+(.+)$", source, re.MULTILINE)
                if title_match:
                    title = title_match.group(1)
                # Get description from first paragraph
                desc_match = re.search(r"^#\s+.+\n\n(.+?)(?:\n\n|\Z)", source, re.DOTALL)
                if desc_match:
                    description = desc_match.group(1).strip()[:200]
                break

        # Count cells by type
        code_cells = sum(1 for c in cells if c.get("cell_type") == "code")
        markdown_cells = sum(1 for c in cells if c.get("cell_type") == "markdown")

        # Get kernel info
        kernel = metadata.get("kernelspec", {}).get("display_name", "Unknown")

        return {
            "title": title,
            "description": description,
            "code_cells": code_cells,
            "markdown_cells": markdown_cells,
            "total_cells": len(cells),
            "kernel": kernel
        }
    except (json.JSONDecodeError, KeyError):
        return {
            "title": notebook_path.stem,
            "description": "",
            "code_cells": 0,
            "markdown_cells": 0,
            "total_cells": 0,
            "kernel": "Unknown"
        }


@router.get("")
async def list_labs(
    domain: str = None,
    limit: int = 50
):
    """
    List available labs.

    Returns Jupyter notebooks organized by domain/category.
    """
    labs_path = settings.labs_path

    if not labs_path.exists():
        return {"labs": [], "total": 0}

    labs = []

    # Walk through labs directory
    for item in labs_path.rglob("*.ipynb"):
        if ".ipynb_checkpoints" in str(item):
            continue

        # Get relative path for categorization
        rel_path = item.relative_to(labs_path)
        parts = rel_path.parts

        # Determine domain/category
        lab_domain = parts[0] if len(parts) > 1 else "general"

        # Filter by domain if specified
        if domain and lab_domain != domain:
            continue

        metadata = parse_notebook_metadata(item)

        labs.append({
            "id": str(rel_path).replace("/", "_").replace(".ipynb", ""),
            "domain": lab_domain,
            "path": str(rel_path),
            "filename": item.name,
            **metadata
        })

    # Sort by domain then title
    labs.sort(key=lambda x: (x["domain"], x["title"]))

    return {
        "labs": labs[:limit],
        "total": len(labs)
    }


@router.get("/domains")
async def list_lab_domains():
    """
    Get list of lab categories/domains.
    """
    labs_path = settings.labs_path

    if not labs_path.exists():
        return {"domains": []}

    domains = set()

    for item in labs_path.iterdir():
        if item.is_dir() and not item.name.startswith((".", "_")):
            # Count notebooks in domain
            notebooks = list(item.rglob("*.ipynb"))
            notebooks = [n for n in notebooks if ".ipynb_checkpoints" not in str(n)]

            if notebooks:
                domains.add(item.name)

    return {
        "domains": sorted(list(domains))
    }


@router.get("/{lab_id}")
async def get_lab(lab_id: str, user_id: str = "default"):
    """
    Get lab details and content.

    Returns notebook metadata and cell contents.
    """
    # Convert ID back to path
    lab_path = lab_id.replace("_", "/") + ".ipynb"
    full_path = settings.labs_path / lab_path

    if not full_path.exists():
        # Try direct match
        for notebook in settings.labs_path.rglob("*.ipynb"):
            if notebook.stem == lab_id or lab_id in str(notebook):
                full_path = notebook
                break
        else:
            raise HTTPException(status_code=404, detail=f"Lab '{lab_id}' not found")

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse notebook")

    metadata = parse_notebook_metadata(full_path)

    # Get user progress for this lab
    progress_rows = execute_query(
        """
        SELECT completed, started_at, completed_at, cell_progress
        FROM lab_runs
        WHERE user_id = ? AND lab_id = ?
        ORDER BY started_at DESC
        LIMIT 1
        """,
        (user_id, lab_id)
    )

    user_progress = None
    if progress_rows:
        row = progress_rows[0]
        user_progress = {
            "completed": bool(row["completed"]),
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "cell_progress": json.loads(row["cell_progress"] or "[]")
        }

    return {
        "id": lab_id,
        "path": str(full_path.relative_to(settings.labs_path)),
        **metadata,
        "cells": notebook.get("cells", []),
        "metadata": notebook.get("metadata", {}),
        "user_progress": user_progress
    }


@router.post("/{lab_id}/start")
async def start_lab(lab_id: str, user_id: str = "default"):
    """
    Start a lab session.

    Creates a progress record for tracking.
    """
    # Verify lab exists
    lab_path = lab_id.replace("_", "/") + ".ipynb"
    full_path = settings.labs_path / lab_path

    if not full_path.exists():
        for notebook in settings.labs_path.rglob("*.ipynb"):
            if notebook.stem == lab_id or lab_id in str(notebook):
                full_path = notebook
                break
        else:
            raise HTTPException(status_code=404, detail=f"Lab '{lab_id}' not found")

    # Create or update lab run record
    execute_write(
        """
        INSERT INTO lab_runs (user_id, lab_id, completed, cell_progress)
        VALUES (?, ?, FALSE, '[]')
        """,
        (user_id, lab_id)
    )

    return {
        "success": True,
        "lab_id": lab_id,
        "started_at": datetime.utcnow().isoformat()
    }


@router.post("/{lab_id}/progress")
async def update_lab_progress(
    lab_id: str,
    cell_index: int,
    completed: bool = True,
    user_id: str = "default"
):
    """
    Update progress for a specific cell in a lab.
    """
    # Get current progress
    rows = execute_query(
        """
        SELECT id, cell_progress FROM lab_runs
        WHERE user_id = ? AND lab_id = ?
        ORDER BY started_at DESC
        LIMIT 1
        """,
        (user_id, lab_id)
    )

    if not rows:
        raise HTTPException(status_code=404, detail="Lab session not found. Start the lab first.")

    row = rows[0]
    cell_progress = json.loads(row["cell_progress"] or "[]")

    # Update cell progress
    cell_entry = {
        "cellIndex": cell_index,
        "completed": completed,
        "executedAt": datetime.utcnow().isoformat() if completed else None
    }

    # Find and update or append
    found = False
    for i, entry in enumerate(cell_progress):
        if entry.get("cellIndex") == cell_index:
            cell_progress[i] = cell_entry
            found = True
            break

    if not found:
        cell_progress.append(cell_entry)

    # Save updated progress
    execute_write(
        """
        UPDATE lab_runs
        SET cell_progress = ?
        WHERE id = ?
        """,
        (json.dumps(cell_progress), row["id"])
    )

    return {
        "success": True,
        "cell_progress": cell_progress
    }


@router.post("/{lab_id}/complete")
async def complete_lab(lab_id: str, user_id: str = "default"):
    """
    Mark a lab as completed.
    """
    rows = execute_query(
        """
        SELECT id FROM lab_runs
        WHERE user_id = ? AND lab_id = ?
        ORDER BY started_at DESC
        LIMIT 1
        """,
        (user_id, lab_id)
    )

    if not rows:
        raise HTTPException(status_code=404, detail="Lab session not found")

    execute_write(
        """
        UPDATE lab_runs
        SET completed = TRUE, completed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (rows[0]["id"],)
    )

    return {
        "success": True,
        "completed_at": datetime.utcnow().isoformat()
    }
