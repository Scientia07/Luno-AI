# Luno-AI Design Documentation

> **Version**: 1.0.0
> **Date**: 2026-01-05
> **Status**: Design Complete - Ready for Implementation

---

## Overview

This directory contains the complete system design for the Luno-AI platform - an educational AI technology exploration dashboard with:

- **48 Integration PRDs** across 9 AI domains
- **11 Research Sessions** in the knowledge vault
- **6 Learning Paths** from beginner to advanced
- **Interactive Labs** for hands-on experimentation

---

## Design Documents

| Document | Description |
|----------|-------------|
| [architecture.md](./architecture.md) | System architecture, component design, data flow |
| [api-specification.md](./api-specification.md) | REST API endpoints, request/response schemas |
| [database-schema.md](./database-schema.md) | SQLite + ChromaDB schemas, migrations |

---

## Quick Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    LUNO-AI PLATFORM                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│   │  Next.js 14 │───▶│   FastAPI   │───▶│  Markdown   │    │
│   │  Dashboard  │    │   Backend   │    │   PRDs (48) │    │
│   └─────────────┘    └─────────────┘    └─────────────┘    │
│          │                  │                               │
│          │                  ▼                               │
│          │           ┌─────────────┐    ┌─────────────┐    │
│          │           │   SQLite    │    │  ChromaDB   │    │
│          └──────────▶│  Progress   │    │   Search    │    │
│                      └─────────────┘    └─────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Choices

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Frontend** | Next.js 14 | Server components, file routing, great DX |
| **Styling** | Tailwind + Shadcn/UI | Rapid development, consistent design |
| **Backend** | FastAPI | Fast async Python, OpenAPI docs |
| **Database** | SQLite | Simple, embedded, no setup required |
| **Vector DB** | ChromaDB | Python-native, local-first |
| **Labs** | Jupyter | Industry standard notebooks |

---

## Key Features

### 1. Technology Explorer
Navigate all 48 integrations by domain, with layered depth (L0-L4).

### 2. Semantic Search
Find content across PRDs and research using vector similarity.

### 3. Progress Tracking
Track completion of learning layers, bookmarks, and notes.

### 4. Research Vault
Browse 11 research sessions with full context preservation.

### 5. Interactive Labs
Run Jupyter notebooks directly in the browser.

---

## API Endpoints Summary

| Group | Endpoints | Purpose |
|-------|-----------|---------|
| `/explore` | 4 | Technology browser |
| `/search` | 2 | Semantic search |
| `/progress` | 5 | Progress tracking |
| `/research` | 2 | Research vault |
| `/labs` | 3 | Lab management |

See [api-specification.md](./api-specification.md) for full details.

---

## Database Summary

### SQLite Tables
- `users` - User accounts
- `progress` - Learning progress (domain/tech/layer)
- `bookmarks` - Saved content
- `research_views` - Research session history
- `lab_runs` - Lab completion tracking

### ChromaDB Collections
- `integrations` - PRD content embeddings
- `research` - Research session embeddings
- `code_examples` - Code snippet embeddings

See [database-schema.md](./database-schema.md) for full schemas.

---

## Implementation Phases

### Phase 1: Backend API (1-2 days)
- [ ] FastAPI project setup
- [ ] Markdown parser service
- [ ] Explore endpoints
- [ ] SQLite setup

### Phase 2: Search Service (1 day)
- [ ] ChromaDB integration
- [ ] Indexing pipeline
- [ ] Search endpoints

### Phase 3: Frontend Shell (2-3 days)
- [ ] Next.js project setup
- [ ] Layout components
- [ ] Domain explorer pages
- [ ] Technology detail pages

### Phase 4: Progress System (1 day)
- [ ] Progress API
- [ ] Frontend progress UI
- [ ] Bookmarks feature

### Phase 5: Labs Integration (1-2 days)
- [ ] Jupyter bridge
- [ ] Lab runner component
- [ ] Cell execution

### Phase 6: Polish (1-2 days)
- [ ] Search UI
- [ ] Dark mode
- [ ] Mobile responsive
- [ ] Performance optimization

**Estimated Total**: 7-11 days

---

## File Structure (Target)

```
src/
├── api/                     # FastAPI backend
│   ├── main.py
│   ├── routers/
│   │   ├── explore.py
│   │   ├── search.py
│   │   ├── progress.py
│   │   ├── research.py
│   │   └── labs.py
│   ├── services/
│   │   ├── markdown_parser.py
│   │   ├── search_service.py
│   │   └── progress_service.py
│   ├── models/
│   └── database/
│
├── dashboard/               # Next.js frontend
│   ├── app/
│   │   ├── page.tsx
│   │   ├── explore/
│   │   ├── paths/
│   │   ├── labs/
│   │   └── research/
│   ├── components/
│   └── lib/
│
└── core/                    # Shared utilities
    └── config.py
```

---

## Getting Started (Implementation)

```bash
# 1. Create backend structure
mkdir -p src/api/{routers,services,models,database}

# 2. Create frontend structure
cd src/dashboard
npx create-next-app@latest . --typescript --tailwind --app

# 3. Install dependencies
pip install fastapi uvicorn chromadb
npm install @shadcn/ui

# 4. Start development
uvicorn src.api.main:app --reload  # Backend
npm run dev                         # Frontend
```

---

## Related Documents

- [CONCEPT.md](../../CONCEPT.md) - Project vision
- [integrations/_index.md](../../integrations/_index.md) - PRD catalog
- [research/_index.md](../../research/_index.md) - Research sessions

---

*Design documentation for Luno-AI v1.0*
