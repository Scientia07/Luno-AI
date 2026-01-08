# Luno-AI Database Schema

> **Version**: 1.0.0
> **Database**: SQLite (local) + ChromaDB (vectors)

---

## Overview

Luno-AI uses a hybrid data storage approach:

| Store | Technology | Purpose |
|-------|------------|---------|
| **Progress DB** | SQLite | User progress, bookmarks, notes |
| **Vector DB** | ChromaDB | Semantic search, embeddings |
| **File System** | Markdown | PRDs, research, labs |

---

## SQLite Schema

### Entity Relationship Diagram

```
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│     users     │       │   progress    │       │   bookmarks   │
├───────────────┤       ├───────────────┤       ├───────────────┤
│ id (PK)       │───────│ user_id (FK)  │───────│ user_id (FK)  │
│ name          │       │ domain        │       │ domain        │
│ email         │       │ technology    │       │ technology    │
│ created_at    │       │ layer         │       │ section       │
│ settings      │       │ completed     │       │ created_at    │
└───────────────┘       │ completed_at  │       └───────────────┘
                        │ notes         │
                        └───────────────┘
                               │
                               │
                        ┌──────┴──────┐
                        │             │
┌───────────────┐       │       ┌───────────────┐
│ research_views│───────┘       │   lab_runs    │
├───────────────┤               ├───────────────┤
│ user_id (FK)  │               │ user_id (FK)  │
│ session_path  │               │ lab_id        │
│ viewed_at     │               │ completed     │
└───────────────┘               │ started_at    │
                                │ completed_at  │
                                └───────────────┘
```

### Table Definitions

#### users

```sql
CREATE TABLE users (
    id TEXT PRIMARY KEY DEFAULT 'default',
    name TEXT,
    email TEXT UNIQUE,
    avatar_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settings JSON DEFAULT '{}'
);

-- Default user for single-user mode
INSERT INTO users (id, name) VALUES ('default', 'Local User');
```

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `id` | TEXT | User identifier (UUID or 'default') |
| `name` | TEXT | Display name |
| `email` | TEXT | Email (optional, for multi-user) |
| `avatar_url` | TEXT | Profile image URL |
| `created_at` | TIMESTAMP | Account creation time |
| `settings` | JSON | User preferences |

**Settings JSON Schema**:
```json
{
  "theme": "dark",
  "defaultDomain": "agents",
  "showCompletedLayers": true,
  "notificationsEnabled": false
}
```

---

#### progress

```sql
CREATE TABLE progress (
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
);

-- Indexes for common queries
CREATE INDEX idx_progress_user ON progress(user_id);
CREATE INDEX idx_progress_domain ON progress(domain);
CREATE INDEX idx_progress_tech ON progress(domain, technology);

-- Trigger to update updated_at
CREATE TRIGGER update_progress_timestamp
AFTER UPDATE ON progress
BEGIN
    UPDATE progress SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
```

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `user_id` | TEXT | Foreign key to users |
| `domain` | TEXT | Domain identifier (e.g., 'agents') |
| `technology` | TEXT | Technology identifier (e.g., 'crewai') |
| `layer` | INTEGER | Learning layer (0-4) |
| `completed` | BOOLEAN | Whether layer is completed |
| `completed_at` | TIMESTAMP | Completion timestamp |
| `notes` | TEXT | User notes for this tech |
| `created_at` | TIMESTAMP | Record creation time |
| `updated_at` | TIMESTAMP | Last update time |

---

#### bookmarks

```sql
CREATE TABLE bookmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',
    domain TEXT NOT NULL,
    technology TEXT NOT NULL,
    section TEXT,  -- Optional: specific section/anchor
    title TEXT,    -- Display title for bookmark
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(user_id, domain, technology, section)
);

CREATE INDEX idx_bookmarks_user ON bookmarks(user_id);
```

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `user_id` | TEXT | Foreign key to users |
| `domain` | TEXT | Domain identifier |
| `technology` | TEXT | Technology identifier |
| `section` | TEXT | Section anchor (e.g., 'quick-start') |
| `title` | TEXT | Bookmark display title |
| `created_at` | TIMESTAMP | Bookmark creation time |

---

#### research_views

```sql
CREATE TABLE research_views (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',
    session_path TEXT NOT NULL,
    viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_research_user ON research_views(user_id);
CREATE INDEX idx_research_session ON research_views(session_path);
```

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `user_id` | TEXT | Foreign key to users |
| `session_path` | TEXT | Research session folder path |
| `viewed_at` | TIMESTAMP | View timestamp |

---

#### lab_runs

```sql
CREATE TABLE lab_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',
    lab_id TEXT NOT NULL,
    completed BOOLEAN DEFAULT FALSE,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    cell_progress JSON DEFAULT '[]',

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_labs_user ON lab_runs(user_id);
CREATE INDEX idx_labs_lab ON lab_runs(lab_id);
```

**Fields**:
| Field | Type | Description |
|-------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `user_id` | TEXT | Foreign key to users |
| `lab_id` | TEXT | Lab identifier |
| `completed` | BOOLEAN | Whether lab is completed |
| `started_at` | TIMESTAMP | Start timestamp |
| `completed_at` | TIMESTAMP | Completion timestamp |
| `cell_progress` | JSON | Progress through cells |

**cell_progress JSON Schema**:
```json
[
  {"cellId": "cell-1", "completed": true, "executedAt": "2026-01-05T10:00:00Z"},
  {"cellId": "cell-2", "completed": true, "executedAt": "2026-01-05T10:05:00Z"},
  {"cellId": "cell-3", "completed": false, "executedAt": null}
]
```

---

#### search_history (optional)

```sql
CREATE TABLE search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',
    query TEXT NOT NULL,
    results_count INTEGER,
    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX idx_search_user ON search_history(user_id);
CREATE INDEX idx_search_time ON search_history(searched_at);
```

---

## ChromaDB Collections

### Collection: integrations

Stores embeddings for all 48 integration PRDs.

```python
from chromadb import Client
from chromadb.config import Settings

client = Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./data/chroma"
))

integrations = client.get_or_create_collection(
    name="integrations",
    metadata={
        "description": "PRD content for all integrations",
        "embedding_model": "text-embedding-3-small"
    }
)
```

**Document Schema**:
```python
{
    "id": "agents_crewai_overview",      # Unique ID
    "document": "CrewAI is a...",        # Text content
    "metadata": {
        "domain": "agents",
        "technology": "crewai",
        "section": "overview",
        "layer": 0,
        "path": "integrations/agents/crewai.md",
        "title": "CrewAI Integration"
    }
}
```

**Indexing Strategy**:
```python
def index_prd(prd_path: str):
    """Index a PRD file into ChromaDB."""
    content = read_markdown(prd_path)
    sections = split_by_sections(content)

    documents = []
    ids = []
    metadatas = []

    for section in sections:
        doc_id = f"{domain}_{tech}_{section.slug}"
        documents.append(section.content)
        ids.append(doc_id)
        metadatas.append({
            "domain": domain,
            "technology": tech,
            "section": section.name,
            "layer": section.layer,
            "path": prd_path
        })

    integrations.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
```

---

### Collection: research

Stores embeddings for research session content.

```python
research = client.get_or_create_collection(
    name="research",
    metadata={
        "description": "Research session content",
        "embedding_model": "text-embedding-3-small"
    }
)
```

**Document Schema**:
```python
{
    "id": "2026-01-05_agentic-framework-comparison_readme",
    "document": "This research compares...",
    "metadata": {
        "session": "2026-01-05_agentic-framework-comparison",
        "file": "README.md",
        "date": "2026-01-05",
        "tags": ["crewai", "langgraph", "autogen"]
    }
}
```

---

### Collection: code_examples

Stores code snippets for semantic code search.

```python
code_examples = client.get_or_create_collection(
    name="code_examples",
    metadata={
        "description": "Code snippets from PRDs",
        "embedding_model": "text-embedding-3-small"
    }
)
```

**Document Schema**:
```python
{
    "id": "agents_crewai_quickstart_1",
    "document": "from crewai import Agent, Task, Crew...",
    "metadata": {
        "domain": "agents",
        "technology": "crewai",
        "language": "python",
        "title": "Quick Start",
        "description": "Basic agent creation"
    }
}
```

---

## Query Examples

### SQLite Queries

#### Get User Progress Summary

```sql
SELECT
    domain,
    COUNT(DISTINCT technology) as total_tech,
    SUM(CASE WHEN completed = 1 THEN 1 ELSE 0 END) as completed_layers,
    ROUND(100.0 * SUM(CASE WHEN completed = 1 THEN 1 ELSE 0 END) /
          (COUNT(DISTINCT technology) * 5), 1) as percentage
FROM progress
WHERE user_id = 'default'
GROUP BY domain;
```

#### Get Recent Activity

```sql
SELECT
    p.domain,
    p.technology,
    p.layer,
    p.completed_at
FROM progress p
WHERE p.user_id = 'default' AND p.completed = 1
ORDER BY p.completed_at DESC
LIMIT 10;
```

#### Get All Bookmarks with Details

```sql
SELECT
    b.domain,
    b.technology,
    b.section,
    b.title,
    b.created_at,
    MAX(p.layer) as current_layer
FROM bookmarks b
LEFT JOIN progress p ON b.domain = p.domain
    AND b.technology = p.technology
    AND b.user_id = p.user_id
WHERE b.user_id = 'default'
GROUP BY b.id
ORDER BY b.created_at DESC;
```

---

### ChromaDB Queries

#### Semantic Search

```python
results = integrations.query(
    query_texts=["how to create multi-agent systems"],
    n_results=5,
    where={"domain": {"$in": ["agents", "llms"]}}
)

# Returns:
# {
#     "ids": [["agents_crewai_overview", "agents_multi-agent_patterns", ...]],
#     "documents": [["CrewAI is a...", "Multi-agent patterns...", ...]],
#     "distances": [[0.23, 0.31, ...]],
#     "metadatas": [[{...}, {...}, ...]]
# }
```

#### Code Search

```python
results = code_examples.query(
    query_texts=["create agent with tools"],
    n_results=3,
    where={"language": "python"}
)
```

---

## Migrations

### Initial Migration (v1.0.0)

```sql
-- migrations/001_initial.sql

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY DEFAULT 'default',
    name TEXT,
    email TEXT UNIQUE,
    avatar_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settings JSON DEFAULT '{}'
);

-- Insert default user
INSERT OR IGNORE INTO users (id, name) VALUES ('default', 'Local User');

-- Create progress table
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
);

-- Create bookmarks table
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
);

-- Create research_views table
CREATE TABLE IF NOT EXISTS research_views (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',
    session_path TEXT NOT NULL,
    viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create lab_runs table
CREATE TABLE IF NOT EXISTS lab_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'default',
    lab_id TEXT NOT NULL,
    completed BOOLEAN DEFAULT FALSE,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    cell_progress JSON DEFAULT '[]',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_progress_user ON progress(user_id);
CREATE INDEX IF NOT EXISTS idx_progress_domain ON progress(domain);
CREATE INDEX IF NOT EXISTS idx_progress_tech ON progress(domain, technology);
CREATE INDEX IF NOT EXISTS idx_bookmarks_user ON bookmarks(user_id);
CREATE INDEX IF NOT EXISTS idx_research_user ON research_views(user_id);
CREATE INDEX IF NOT EXISTS idx_labs_user ON lab_runs(user_id);
```

---

## Data Directory Structure

```
data/
├── luno.db              # SQLite database
├── chroma/              # ChromaDB persistence
│   ├── integrations/
│   ├── research/
│   └── code_examples/
└── backups/             # Automated backups
    └── luno_2026-01-05.db
```

---

## Backup & Restore

```bash
# Backup SQLite
cp data/luno.db data/backups/luno_$(date +%Y-%m-%d).db

# Backup ChromaDB
tar -czvf data/backups/chroma_$(date +%Y-%m-%d).tar.gz data/chroma/

# Restore
cp data/backups/luno_2026-01-05.db data/luno.db
```

---

*Database Schema for Luno-AI v1.0*
