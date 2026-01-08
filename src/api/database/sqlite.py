"""
SQLite database setup and connection management.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from ..config import settings


def get_db_path() -> Path:
    """Get the path to the SQLite database file."""
    # Extract path from sqlite:/// URL
    db_url = settings.database_url
    if db_url.startswith("sqlite:///"):
        db_path = db_url[10:]  # Remove 'sqlite:///'
    else:
        db_path = "data/luno.db"

    # Make it relative to project root
    full_path = settings.project_root / db_path

    # Ensure parent directory exists
    full_path.parent.mkdir(parents=True, exist_ok=True)

    return full_path


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """Initialize the database schema."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY DEFAULT 'default',
                name TEXT,
                email TEXT UNIQUE,
                avatar_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                settings JSON DEFAULT '{}'
            )
        """)

        # Insert default user
        cursor.execute("""
            INSERT OR IGNORE INTO users (id, name) VALUES ('default', 'Local User')
        """)

        # Create progress table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                domain TEXT NOT NULL,
                technology TEXT NOT NULL,
                layer INTEGER NOT NULL DEFAULT 0,
                completed BOOLEAN NOT NULL DEFAULT FALSE,
                completed_at TIMESTAMP,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, domain, technology, layer)
            )
        """)

        # Create bookmarks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bookmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                domain TEXT NOT NULL,
                technology TEXT NOT NULL,
                section TEXT,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                UNIQUE(user_id, domain, technology, section)
            )
        """)

        # Create research_views table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_views (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                session_path TEXT NOT NULL,
                viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # Create lab_runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lab_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                lab_id TEXT NOT NULL,
                completed BOOLEAN DEFAULT FALSE,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                cell_progress JSON DEFAULT '[]',
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # Create search_history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                query TEXT NOT NULL,
                results_count INTEGER,
                searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_progress_user ON progress(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_progress_domain ON progress(domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_progress_tech ON progress(domain, technology)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bookmarks_user ON bookmarks(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_research_user ON research_views(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_research_session ON research_views(session_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labs_user ON lab_runs(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labs_lab ON lab_runs(lab_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_user ON search_history(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_time ON search_history(searched_at)")

        conn.commit()


def execute_query(query: str, params: tuple = ()) -> list[dict]:
    """Execute a SELECT query and return results as list of dicts."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def execute_write(query: str, params: tuple = ()) -> int:
    """Execute an INSERT/UPDATE/DELETE and return last row id or affected rows."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.lastrowid if cursor.lastrowid else cursor.rowcount
