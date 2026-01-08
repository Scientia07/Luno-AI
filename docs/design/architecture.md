# Luno-AI Platform Architecture

> **Version**: 1.0.0
> **Date**: 2026-01-05
> **Status**: Design Specification

---

## Executive Summary

Luno-AI is an educational AI technology exploration platform with:
- **48 integration PRDs** across 9 domains
- **Layered learning** (L0-L4 depth system)
- **Research vault** for persistent context
- **Interactive labs** for hands-on experiments

This document defines the architecture for the web dashboard and supporting services.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LUNO-AI PLATFORM                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │   WEB DASHBOARD  │  │   API GATEWAY    │  │   FILE SYSTEM    │       │
│  │   (Next.js 14)   │──│   (FastAPI)      │──│   (Markdown)     │       │
│  │                  │  │                  │  │                  │       │
│  │  - Tech Explorer │  │  - REST API      │  │  - PRDs (48)     │       │
│  │  - Learning Path │  │  - WebSocket     │  │  - Research (11) │       │
│  │  - Lab Runner    │  │  - Auth          │  │  - Labs          │       │
│  │  - Research View │  │                  │  │                  │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
│           │                    │                      │                  │
│           └────────────────────┼──────────────────────┘                  │
│                                │                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │   SQLITE DB      │  │   VECTOR DB      │  │   JUPYTER        │       │
│  │   (Progress)     │  │   (ChromaDB)     │  │   (Labs)         │       │
│  │                  │  │                  │  │                  │       │
│  │  - User progress │  │  - Doc search    │  │  - Notebooks     │       │
│  │  - Bookmarks     │  │  - RAG queries   │  │  - Code exec     │       │
│  │  - Notes         │  │  - Embeddings    │  │  - Visualize     │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Web Dashboard (Frontend)

**Technology**: Next.js 14 + TypeScript + Tailwind CSS

```
src/dashboard/
├── app/                      # Next.js 14 App Router
│   ├── layout.tsx           # Root layout with sidebar
│   ├── page.tsx             # Home/Dashboard
│   ├── explore/             # Technology explorer
│   │   ├── page.tsx         # Domain overview
│   │   └── [domain]/        # Dynamic domain routes
│   │       ├── page.tsx     # Domain detail
│   │       └── [tech]/      # Technology detail
│   │           └── page.tsx
│   ├── paths/               # Learning paths
│   │   ├── page.tsx         # All paths
│   │   └── [path]/          # Path detail
│   │       └── page.tsx
│   ├── labs/                # Interactive labs
│   │   ├── page.tsx         # Lab gallery
│   │   └── [lab]/           # Lab runner
│   │       └── page.tsx
│   ├── research/            # Research vault
│   │   ├── page.tsx         # Session list
│   │   └── [session]/       # Session detail
│   │       └── page.tsx
│   └── api/                 # API routes (BFF)
│       ├── explore/
│       ├── progress/
│       └── search/
├── components/
│   ├── ui/                  # Shadcn/UI components
│   ├── layout/              # Layout components
│   │   ├── Sidebar.tsx
│   │   ├── Header.tsx
│   │   └── Breadcrumb.tsx
│   ├── explore/             # Explorer components
│   │   ├── DomainCard.tsx
│   │   ├── TechCard.tsx
│   │   ├── LayerNav.tsx
│   │   └── CodeBlock.tsx
│   ├── labs/                # Lab components
│   │   ├── NotebookViewer.tsx
│   │   └── CodeEditor.tsx
│   └── search/              # Search components
│       ├── SearchBar.tsx
│       └── SearchResults.tsx
├── lib/
│   ├── api.ts               # API client
│   ├── markdown.ts          # MD parsing
│   └── utils.ts             # Utilities
├── hooks/
│   ├── useProgress.ts
│   ├── useSearch.ts
│   └── useLabs.ts
└── types/
    └── index.ts             # TypeScript types
```

### 2. API Gateway (Backend)

**Technology**: FastAPI + Python 3.11+

```
src/api/
├── main.py                  # FastAPI app
├── routers/
│   ├── explore.py           # Technology explorer API
│   ├── progress.py          # Progress tracking API
│   ├── search.py            # Search API
│   ├── labs.py              # Lab management API
│   └── research.py          # Research vault API
├── services/
│   ├── markdown_parser.py   # Parse PRD markdown
│   ├── search_service.py    # Vector search
│   ├── progress_service.py  # Progress tracking
│   └── lab_service.py       # Jupyter integration
├── models/
│   ├── domain.py            # Domain/Tech models
│   ├── progress.py          # Progress models
│   └── search.py            # Search models
├── database/
│   ├── sqlite.py            # SQLite connection
│   └── chroma.py            # ChromaDB connection
└── config.py                # Configuration
```

### 3. Data Layer

**SQLite Schema** (Progress Tracking):

```sql
-- Users (optional, for multi-user)
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Learning progress
CREATE TABLE progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT DEFAULT 'default',
    domain TEXT NOT NULL,
    technology TEXT NOT NULL,
    layer INTEGER DEFAULT 0,  -- 0-4
    completed_at TIMESTAMP,
    notes TEXT,
    UNIQUE(user_id, domain, technology)
);

-- Bookmarks
CREATE TABLE bookmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT DEFAULT 'default',
    domain TEXT NOT NULL,
    technology TEXT NOT NULL,
    section TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Research sessions viewed
CREATE TABLE research_views (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT DEFAULT 'default',
    session_path TEXT NOT NULL,
    viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**ChromaDB Collections**:

```python
# Collections for semantic search
collections = {
    "integrations": {
        "description": "PRD content for all 48 integrations",
        "embedding_model": "text-embedding-3-small"
    },
    "research": {
        "description": "Research session content",
        "embedding_model": "text-embedding-3-small"
    },
    "code_examples": {
        "description": "Code snippets from PRDs",
        "embedding_model": "text-embedding-3-small"
    }
}
```

---

## Data Flow

### 1. Technology Explorer Flow

```
User Request: GET /explore/agents/crewai

┌─────────┐    ┌─────────────┐    ┌─────────────────┐    ┌──────────────┐
│ Browser │───▶│ Next.js API │───▶│ FastAPI Backend │───▶│ File System  │
│         │    │  (BFF)      │    │                 │    │              │
└─────────┘    └─────────────┘    └─────────────────┘    └──────────────┘
     │                                    │                      │
     │         Response JSON              │   Parse Markdown     │
     │◀───────────────────────────────────│◀─────────────────────│
     │                                    │                      │
     │  {                                 │  integrations/       │
     │    domain: "agents",               │  agents/             │
     │    tech: "crewai",                 │  crewai.md           │
     │    title: "...",                   │                      │
     │    layers: [...],                  │                      │
     │    codeExamples: [...],            │                      │
     │    relatedTech: [...]              │                      │
     │  }                                 │                      │
```

### 2. Search Flow

```
User Search: "how to create agents"

┌─────────┐    ┌─────────────┐    ┌─────────────────┐    ┌──────────────┐
│ Browser │───▶│  Search API │───▶│ Search Service  │───▶│   ChromaDB   │
└─────────┘    └─────────────┘    └─────────────────┘    └──────────────┘
     │                                    │                      │
     │                                    │  1. Generate embedding
     │                                    │  2. Vector similarity
     │                                    │  3. Rank results
     │◀───────────────────────────────────│◀─────────────────────│
     │                                    │                      │
     │  [                                 │                      │
     │    { path: "agents/crewai.md",     │                      │
     │      section: "Quick Start",       │                      │
     │      score: 0.92 },                │                      │
     │    { path: "agents/langgraph.md",  │                      │
     │      section: "Basic Agent",       │                      │
     │      score: 0.87 }                 │                      │
     │  ]                                 │                      │
```

### 3. Progress Tracking Flow

```
User Action: Complete Layer 1 of CrewAI

┌─────────┐    ┌─────────────┐    ┌─────────────────┐    ┌──────────────┐
│ Browser │───▶│ Progress API│───▶│ Progress Service│───▶│   SQLite     │
└─────────┘    └─────────────┘    └─────────────────┘    └──────────────┘
     │                                    │                      │
     │  POST /api/progress               │  INSERT/UPDATE       │
     │  {                                │  progress SET        │
     │    domain: "agents",              │  layer = 1,          │
     │    tech: "crewai",                │  completed_at = NOW  │
     │    layer: 1                       │                      │
     │  }                                │                      │
     │                                    │                      │
     │◀───────────────────────────────────│◀─────────────────────│
     │  { success: true, progress: 25% } │                      │
```

---

## UI Components

### Dashboard Home

```
┌─────────────────────────────────────────────────────────────────┐
│  [Logo] Luno-AI                           [Search] [Settings]   │
├─────────┬───────────────────────────────────────────────────────┤
│         │                                                        │
│ EXPLORE │   Welcome to Luno-AI                                   │
│ ├ Visual│                                                        │
│ ├ Gen AI│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │
│ ├ Audio │   │ Your        │ │ Quick Start │ │ Recently    │    │
│ ├ LLMs  │   │ Progress    │ │             │ │ Viewed      │    │
│ ├ Agents│   │             │ │ • YOLO      │ │             │    │
│ ├ ML    │   │ ████░░ 42%  │ │ • Ollama    │ │ • CrewAI    │    │
│ ├ Deploy│   │             │ │ • CrewAI    │ │ • LangGraph │    │
│ ├ Robot │   └─────────────┘ └─────────────┘ └─────────────┘    │
│ └ Special                                                        │
│         │   Learning Paths                                       │
│ PATHS   │   ┌─────────────────────────────────────────────┐     │
│ • Beginn│   │ AI Beginner      ████████░░░░░░░░  50%      │     │
│ • ML Eng│   │ LLM Developer    ██░░░░░░░░░░░░░░  15%      │     │
│ • LLM   │   │ CV Specialist    ░░░░░░░░░░░░░░░░   0%      │     │
│ • CV    │   └─────────────────────────────────────────────┘     │
│ • GenAI │                                                        │
│ • Audio │   Recent Research                                      │
│         │   ┌─────────────────────────────────────────────┐     │
│ LABS    │   │ 📄 Agentic Framework Comparison (Today)     │     │
│         │   │ 📄 MCP Implementation Patterns (Today)      │     │
│ RESEARCH│   │ 📄 Edge AI Deployment (Jan 2)               │     │
│         │   └─────────────────────────────────────────────┘     │
└─────────┴───────────────────────────────────────────────────────┘
```

### Technology Explorer

```
┌─────────────────────────────────────────────────────────────────┐
│  [←] Agents / CrewAI                      [Bookmark] [Progress] │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  # CrewAI Integration                                           │
│  > Role-based multi-agent collaboration                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ [L0 Overview] [L1 Concepts] [L2 Deep Dive] [L3 Code]      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ## Overview                                                     │
│  | Aspect | Details |                                           │
│  |--------|---------|                                           │
│  | What   | Role-based multi-agent framework |                  │
│  | Why    | Collaborative AI workflows |                        │
│  | Tools  | crewai, crewai-tools |                             │
│                                                                  │
│  ## Quick Start (15 min)                                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ pip install crewai crewai-tools                     [Copy] │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ## Related Technologies                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
│  │ LangGraph│ │ AutoGen  │ │   RAG    │                        │
│  └──────────┘ └──────────┘ └──────────┘                        │
│                                                                  │
│  [Mark L0 Complete ✓]                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Lab Runner

```
┌─────────────────────────────────────────────────────────────────┐
│  [←] Labs / Agents / CrewAI Basics                [Run All]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Cell 1: Setup                                         [▶]  │ │
│  │ ────────────────────────────────────────────────────────── │ │
│  │ from crewai import Agent, Task, Crew                       │ │
│  │                                                            │ │
│  │ # Create a simple agent                                    │ │
│  │ researcher = Agent(                                        │ │
│  │     role='Researcher',                                     │ │
│  │     goal='Find information about AI',                      │ │
│  │     backstory='Expert researcher'                          │ │
│  │ )                                                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Output                                                     │ │
│  │ ────────────────────────────────────────────────────────── │ │
│  │ ✓ Agent created: Researcher                                │ │
│  │ ✓ Goal: Find information about AI                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Cell 2: Create Task                                   [▶]  │ │
│  │ ...                                                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack Summary

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Frontend | Next.js 14 | Server components, file routing, API routes |
| Styling | Tailwind + Shadcn/UI | Rapid development, consistent design |
| State | Zustand | Simple, TypeScript-friendly |
| Backend | FastAPI | Fast, async, Python ecosystem |
| Database | SQLite | Simple, embedded, no setup |
| Vector DB | ChromaDB | Python-native, local-first |
| Markdown | remark + rehype | Extensible, React-friendly |
| Labs | Jupyter | Industry standard for notebooks |

---

## Deployment Options

### Option 1: Local Development (Default)

```bash
# Start all services
docker-compose up

# Or individual services
cd src/dashboard && npm run dev      # Frontend: localhost:3000
cd src/api && uvicorn main:app      # Backend: localhost:8000
jupyter lab --notebook-dir=labs     # Labs: localhost:8888
```

### Option 2: Self-Hosted Production

```yaml
# docker-compose.prod.yml
services:
  dashboard:
    image: luno-ai/dashboard
    ports: ["3000:3000"]

  api:
    image: luno-ai/api
    ports: ["8000:8000"]
    volumes:
      - ./integrations:/app/integrations
      - ./research:/app/research

  jupyter:
    image: luno-ai/labs
    ports: ["8888:8888"]
```

### Option 3: Cloud Deployment

- **Vercel**: Frontend (Next.js native)
- **Railway/Render**: Backend (FastAPI)
- **Persistent Volume**: SQLite + ChromaDB data

---

## Next Steps

1. **Phase 1**: Implement API endpoints for explore/search
2. **Phase 2**: Build frontend components
3. **Phase 3**: Integrate progress tracking
4. **Phase 4**: Add lab runner functionality
5. **Phase 5**: Deploy and iterate

---

*Architecture designed for Luno-AI v1.0*
