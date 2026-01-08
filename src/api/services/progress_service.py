"""
Progress tracking service.
Manages user progress, bookmarks, and learning activity.
"""
from datetime import datetime
from typing import Optional

from ..database.sqlite import execute_query, execute_write
from ..models.progress import (
    LayerProgress,
    TechnologyProgress,
    DomainProgress,
    PathProgress,
    RecentActivity,
    OverallProgress,
    ProgressSummary,
    Bookmark,
)
from ..config import settings


class ProgressService:
    """Service for managing user progress."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id

    def get_technology_progress(self, domain: str, tech: str) -> TechnologyProgress:
        """Get progress for a specific technology."""
        rows = execute_query(
            """
            SELECT layer, completed, completed_at, notes
            FROM progress
            WHERE user_id = ? AND domain = ? AND technology = ?
            ORDER BY layer
            """,
            (self.user_id, domain, tech)
        )

        # Build layer progress list
        layers = []
        notes = None
        current_layer = 0

        # Create entries for all 5 layers (0-4)
        layer_map = {row["layer"]: row for row in rows}

        for level in range(5):
            if level in layer_map:
                row = layer_map[level]
                completed = bool(row["completed"])
                completed_at = (
                    datetime.fromisoformat(row["completed_at"])
                    if row["completed_at"] else None
                )
                if row["notes"]:
                    notes = row["notes"]
                if completed and level >= current_layer:
                    current_layer = level + 1
            else:
                completed = False
                completed_at = None

            layers.append(LayerProgress(
                level=level,
                completed=completed,
                completed_at=completed_at
            ))

        # Check if bookmarked
        bookmark_rows = execute_query(
            """
            SELECT id FROM bookmarks
            WHERE user_id = ? AND domain = ? AND technology = ?
            LIMIT 1
            """,
            (self.user_id, domain, tech)
        )
        bookmarked = len(bookmark_rows) > 0

        return TechnologyProgress(
            domain=domain,
            tech=tech,
            current_layer=min(current_layer, 4),
            layers=layers,
            notes=notes,
            bookmarked=bookmarked
        )

    def update_progress(
        self,
        domain: str,
        tech: str,
        layer: int,
        completed: bool,
        notes: Optional[str] = None
    ) -> dict:
        """Update progress for a technology layer."""
        completed_at = datetime.utcnow().isoformat() if completed else None

        # Use UPSERT pattern
        execute_write(
            """
            INSERT INTO progress (user_id, domain, technology, layer, completed, completed_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, domain, technology, layer)
            DO UPDATE SET
                completed = excluded.completed,
                completed_at = CASE WHEN excluded.completed THEN excluded.completed_at ELSE completed_at END,
                notes = COALESCE(excluded.notes, notes),
                updated_at = CURRENT_TIMESTAMP
            """,
            (self.user_id, domain, tech, layer, completed, completed_at, notes)
        )

        return self.get_technology_progress(domain, tech).model_dump()

    def get_domain_progress(self, domain: str) -> DomainProgress:
        """Get progress summary for a domain."""
        # Count total technologies in domain
        from ..services.markdown_parser import markdown_parser
        technologies = markdown_parser.list_technologies(domain)
        total_tech = len(technologies)

        # Count completed (all 5 layers done)
        rows = execute_query(
            """
            SELECT technology, COUNT(*) as completed_layers
            FROM progress
            WHERE user_id = ? AND domain = ? AND completed = 1
            GROUP BY technology
            HAVING completed_layers = 5
            """,
            (self.user_id, domain)
        )

        completed = len(rows)
        total = total_tech * 5  # 5 layers per tech
        percentage = (completed * 5 / total * 100) if total > 0 else 0

        return DomainProgress(
            domain=domain,
            completed=completed,
            total=total_tech,
            percentage=round(percentage, 1)
        )

    def get_overall_progress(self) -> OverallProgress:
        """Get overall progress across all domains."""
        total_layers = 0
        completed_layers = 0

        for domain_id in settings.domains:
            from ..services.markdown_parser import markdown_parser
            technologies = markdown_parser.list_technologies(domain_id)
            total_layers += len(technologies) * 5

        rows = execute_query(
            """
            SELECT COUNT(*) as count FROM progress
            WHERE user_id = ? AND completed = 1
            """,
            (self.user_id,)
        )

        completed_layers = rows[0]["count"] if rows else 0
        percentage = (completed_layers / total_layers * 100) if total_layers > 0 else 0

        return OverallProgress(
            completed=completed_layers,
            total=total_layers,
            percentage=round(percentage, 1)
        )

    def get_recent_activity(self, limit: int = 10) -> list[RecentActivity]:
        """Get recent learning activity."""
        rows = execute_query(
            """
            SELECT domain, technology, layer, completed_at
            FROM progress
            WHERE user_id = ? AND completed = 1 AND completed_at IS NOT NULL
            ORDER BY completed_at DESC
            LIMIT ?
            """,
            (self.user_id, limit)
        )

        return [
            RecentActivity(
                domain=row["domain"],
                tech=row["technology"],
                layer=row["layer"],
                completed_at=datetime.fromisoformat(row["completed_at"])
            )
            for row in rows
        ]

    def get_progress_summary(self) -> ProgressSummary:
        """Get full progress summary."""
        overall = self.get_overall_progress()

        by_domain = [
            self.get_domain_progress(domain_id)
            for domain_id in settings.domains
        ]

        # TODO: Implement learning paths
        by_path: list[PathProgress] = []

        recent_activity = self.get_recent_activity()

        return ProgressSummary(
            overall=overall,
            by_domain=by_domain,
            by_path=by_path,
            recent_activity=recent_activity
        )

    def get_bookmarks(self) -> list[Bookmark]:
        """Get all bookmarks for user."""
        rows = execute_query(
            """
            SELECT domain, technology, section, title, created_at
            FROM bookmarks
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (self.user_id,)
        )

        return [
            Bookmark(
                domain=row["domain"],
                tech=row["technology"],
                section=row["section"],
                title=row["title"],
                created_at=datetime.fromisoformat(row["created_at"])
            )
            for row in rows
        ]

    def toggle_bookmark(
        self,
        domain: str,
        tech: str,
        section: Optional[str] = None,
        title: Optional[str] = None
    ) -> bool:
        """Toggle bookmark status. Returns new bookmark state."""
        # Check if bookmark exists
        rows = execute_query(
            """
            SELECT id FROM bookmarks
            WHERE user_id = ? AND domain = ? AND technology = ?
            AND (section = ? OR (section IS NULL AND ? IS NULL))
            """,
            (self.user_id, domain, tech, section, section)
        )

        if rows:
            # Remove bookmark
            execute_write(
                """
                DELETE FROM bookmarks
                WHERE user_id = ? AND domain = ? AND technology = ?
                AND (section = ? OR (section IS NULL AND ? IS NULL))
                """,
                (self.user_id, domain, tech, section, section)
            )
            return False
        else:
            # Add bookmark
            execute_write(
                """
                INSERT INTO bookmarks (user_id, domain, technology, section, title)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self.user_id, domain, tech, section, title)
            )
            return True


# Default service instance
progress_service = ProgressService()
