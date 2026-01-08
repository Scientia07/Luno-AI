# Luno-AI API Specification

> **Version**: 1.0.0
> **Base URL**: `http://localhost:8000/api/v1`
> **Format**: REST + JSON

---

## Overview

| Endpoint Group | Description |
|----------------|-------------|
| `/explore` | Technology explorer - domains, techs, content |
| `/search` | Semantic search across all content |
| `/progress` | User progress tracking |
| `/research` | Research vault access |
| `/labs` | Lab management and execution |

---

## Authentication

For local development: No auth required (single user).

For multi-user deployment:
```http
Authorization: Bearer <jwt_token>
```

---

## API Endpoints

### 1. Explore API

#### List All Domains

```http
GET /explore/domains
```

**Response** `200 OK`:
```json
{
  "domains": [
    {
      "id": "visual-ai",
      "name": "Visual AI",
      "description": "Object detection, segmentation, classification",
      "icon": "eye",
      "techCount": 6,
      "color": "#3B82F6"
    },
    {
      "id": "agents",
      "name": "Agentic AI",
      "description": "Autonomous agents and workflows",
      "icon": "bot",
      "techCount": 7,
      "color": "#8B5CF6"
    }
    // ... 9 domains total
  ]
}
```

#### Get Domain Details

```http
GET /explore/domains/{domain_id}
```

**Parameters**:
- `domain_id` (path): Domain identifier (e.g., `agents`, `visual-ai`)

**Response** `200 OK`:
```json
{
  "id": "agents",
  "name": "Agentic AI",
  "description": "Build autonomous AI agents and multi-agent systems",
  "technologies": [
    {
      "id": "crewai",
      "name": "CrewAI",
      "tagline": "Role-based multi-agent collaboration",
      "difficulty": "intermediate",
      "status": "ready",
      "quickStartTime": "15 min"
    },
    {
      "id": "langgraph",
      "name": "LangGraph",
      "tagline": "Graph-based agent orchestration",
      "difficulty": "advanced",
      "status": "ready",
      "quickStartTime": "20 min"
    }
    // ... more techs
  ],
  "learningPath": {
    "id": "llm-developer",
    "name": "LLM Application Developer"
  }
}
```

#### Get Technology Details

```http
GET /explore/domains/{domain_id}/tech/{tech_id}
```

**Parameters**:
- `domain_id` (path): Domain identifier
- `tech_id` (path): Technology identifier (e.g., `crewai`)
- `layer` (query, optional): Specific layer (0-4), default returns all

**Response** `200 OK`:
```json
{
  "id": "crewai",
  "domain": "agents",
  "name": "CrewAI",
  "tagline": "Role-based multi-agent collaboration",
  "overview": {
    "what": "Multi-agent orchestration framework",
    "why": "Collaborative AI workflows",
    "tools": ["crewai", "crewai-tools"],
    "bestFor": "Role-based task delegation"
  },
  "prerequisites": [
    {"name": "Python", "details": "3.10+"},
    {"name": "OpenAI API", "details": "Or compatible LLM"}
  ],
  "quickStart": {
    "time": "15 min",
    "install": "pip install crewai crewai-tools",
    "code": "from crewai import Agent, Task, Crew..."
  },
  "layers": [
    {
      "level": 0,
      "name": "Overview",
      "content": "# CrewAI Integration...",
      "checklistItems": [
        "Understand agents and roles",
        "Create first crew"
      ]
    },
    {
      "level": 1,
      "name": "Concepts",
      "content": "## Core Concepts...",
      "checklistItems": [
        "Agent configuration",
        "Task definition",
        "Process types"
      ]
    }
    // ... layers 2-4
  ],
  "codeExamples": [
    {
      "title": "Basic Agent",
      "language": "python",
      "code": "researcher = Agent(...)"
    }
  ],
  "relatedTech": ["langgraph", "autogen", "rag"],
  "resources": [
    {"title": "Official Docs", "url": "https://docs.crewai.com"}
  ]
}
```

#### Get Technology Content (Raw)

```http
GET /explore/domains/{domain_id}/tech/{tech_id}/raw
```

**Response** `200 OK`:
```json
{
  "markdown": "# CrewAI Integration\n\n> **Role-based multi-agent...",
  "frontmatter": {
    "title": "CrewAI Integration",
    "domain": "agents",
    "difficulty": "intermediate"
  }
}
```

---

### 2. Search API

#### Semantic Search

```http
POST /search
```

**Request Body**:
```json
{
  "query": "how to create a multi-agent system",
  "filters": {
    "domains": ["agents", "llms"],  // optional
    "types": ["integration", "research"]  // optional
  },
  "limit": 10
}
```

**Response** `200 OK`:
```json
{
  "results": [
    {
      "id": "agents/crewai",
      "type": "integration",
      "title": "CrewAI Integration",
      "section": "Quick Start",
      "snippet": "...create multi-agent systems with role-based delegation...",
      "score": 0.94,
      "path": "/explore/agents/crewai"
    },
    {
      "id": "agents/multi-agent",
      "type": "integration",
      "title": "Multi-Agent Systems",
      "section": "Architecture Patterns",
      "snippet": "...hierarchical, peer-to-peer, and swarm patterns...",
      "score": 0.89,
      "path": "/explore/agents/multi-agent"
    },
    {
      "id": "research/2026-01-05_agentic-framework-comparison",
      "type": "research",
      "title": "Agentic Framework Comparison",
      "section": "Multi-Agent Patterns",
      "snippet": "...CrewAI vs LangGraph vs AutoGen...",
      "score": 0.85,
      "path": "/research/2026-01-05_agentic-framework-comparison"
    }
  ],
  "totalResults": 15,
  "queryTime": 0.045
}
```

#### Quick Search (Autocomplete)

```http
GET /search/quick?q={query}
```

**Parameters**:
- `q` (query): Search query (min 2 characters)

**Response** `200 OK`:
```json
{
  "suggestions": [
    {"text": "CrewAI", "path": "/explore/agents/crewai", "type": "tech"},
    {"text": "Creating Agents", "path": "/explore/agents/crewai#quick-start", "type": "section"},
    {"text": "crew management", "path": "/explore/agents/crewai#crews", "type": "topic"}
  ]
}
```

---

### 3. Progress API

#### Get User Progress

```http
GET /progress
```

**Response** `200 OK`:
```json
{
  "overall": {
    "completed": 12,
    "total": 48,
    "percentage": 25
  },
  "byDomain": [
    {
      "domain": "agents",
      "completed": 3,
      "total": 7,
      "percentage": 43
    },
    {
      "domain": "visual-ai",
      "completed": 2,
      "total": 6,
      "percentage": 33
    }
  ],
  "byPath": [
    {
      "pathId": "ai-beginner",
      "name": "AI Beginner",
      "completed": 3,
      "total": 6,
      "percentage": 50
    }
  ],
  "recentActivity": [
    {
      "domain": "agents",
      "tech": "crewai",
      "layer": 1,
      "completedAt": "2026-01-05T12:30:00Z"
    }
  ]
}
```

#### Get Technology Progress

```http
GET /progress/{domain}/{tech}
```

**Response** `200 OK`:
```json
{
  "domain": "agents",
  "tech": "crewai",
  "currentLayer": 1,
  "layers": [
    {"level": 0, "completed": true, "completedAt": "2026-01-04T10:00:00Z"},
    {"level": 1, "completed": true, "completedAt": "2026-01-05T12:30:00Z"},
    {"level": 2, "completed": false, "completedAt": null},
    {"level": 3, "completed": false, "completedAt": null},
    {"level": 4, "completed": false, "completedAt": null}
  ],
  "notes": "Good framework for role-based workflows",
  "bookmarked": true
}
```

#### Update Progress

```http
POST /progress/{domain}/{tech}
```

**Request Body**:
```json
{
  "layer": 2,
  "completed": true,
  "notes": "Completed deep dive section"
}
```

**Response** `200 OK`:
```json
{
  "success": true,
  "progress": {
    "currentLayer": 2,
    "overallProgress": 40
  }
}
```

#### Toggle Bookmark

```http
POST /progress/{domain}/{tech}/bookmark
```

**Response** `200 OK`:
```json
{
  "bookmarked": true
}
```

#### Get Bookmarks

```http
GET /progress/bookmarks
```

**Response** `200 OK`:
```json
{
  "bookmarks": [
    {
      "domain": "agents",
      "tech": "crewai",
      "section": "Quick Start",
      "createdAt": "2026-01-05T10:00:00Z"
    }
  ]
}
```

---

### 4. Research API

#### List Research Sessions

```http
GET /research
```

**Parameters**:
- `domain` (query, optional): Filter by domain
- `limit` (query, optional): Max results (default 20)

**Response** `200 OK`:
```json
{
  "sessions": [
    {
      "id": "2026-01-05_agentic-framework-comparison",
      "title": "Agentic Framework Comparison",
      "date": "2026-01-05",
      "domains": ["agents", "multi-agent"],
      "status": "complete",
      "summary": "Deep comparison of CrewAI vs LangGraph vs AutoGen"
    },
    {
      "id": "2026-01-05_mcp-implementation-patterns",
      "title": "MCP Implementation Patterns",
      "date": "2026-01-05",
      "domains": ["agents", "tools"],
      "status": "complete",
      "summary": "Model Context Protocol specification and patterns"
    }
  ],
  "total": 11
}
```

#### Get Research Session

```http
GET /research/{session_id}
```

**Response** `200 OK`:
```json
{
  "id": "2026-01-05_agentic-framework-comparison",
  "title": "Agentic Framework Comparison",
  "date": "2026-01-05",
  "status": "complete",
  "files": {
    "readme": "# Agentic AI Framework Comparison...",
    "sources": "# Sources...",
    "findings": "# Detailed Findings..."
  },
  "artifacts": [
    {"name": "code-examples.py", "type": "python"},
    {"name": "decision-flowchart.md", "type": "markdown"}
  ],
  "tags": ["multi-agent", "crewai", "langgraph", "autogen"],
  "relatedTech": ["agents/crewai", "agents/langgraph"]
}
```

---

### 5. Labs API

#### List Available Labs

```http
GET /labs
```

**Response** `200 OK`:
```json
{
  "labs": [
    {
      "id": "agents-crewai-basics",
      "name": "CrewAI Basics",
      "domain": "agents",
      "tech": "crewai",
      "description": "Create your first multi-agent crew",
      "difficulty": "beginner",
      "estimatedTime": "30 min",
      "status": "available"
    }
  ]
}
```

#### Get Lab Details

```http
GET /labs/{lab_id}
```

**Response** `200 OK`:
```json
{
  "id": "agents-crewai-basics",
  "name": "CrewAI Basics",
  "domain": "agents",
  "tech": "crewai",
  "description": "Create your first multi-agent crew",
  "prerequisites": [
    "Python 3.10+",
    "OpenAI API key"
  ],
  "cells": [
    {
      "id": "cell-1",
      "type": "markdown",
      "content": "# CrewAI Basics Lab\n\nIn this lab..."
    },
    {
      "id": "cell-2",
      "type": "code",
      "language": "python",
      "content": "from crewai import Agent, Task, Crew",
      "output": null
    }
  ],
  "jupyterUrl": "/jupyter/lab?path=agents/crewai-basics.ipynb"
}
```

#### Execute Lab Cell (via Jupyter)

```http
POST /labs/{lab_id}/execute
```

**Request Body**:
```json
{
  "cellId": "cell-2",
  "code": "from crewai import Agent, Task, Crew\nprint('Hello')"
}
```

**Response** `200 OK`:
```json
{
  "success": true,
  "output": "Hello\n",
  "executionTime": 0.5
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Technology 'xyz' not found in domain 'agents'",
    "details": {
      "domain": "agents",
      "tech": "xyz"
    }
  }
}
```

**Error Codes**:

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `SEARCH_ERROR` | 500 | Search service error |
| `LAB_ERROR` | 500 | Lab execution error |
| `AUTH_ERROR` | 401 | Authentication required |

---

## WebSocket API

### Real-time Lab Execution

```
WS /ws/labs/{lab_id}
```

**Client → Server**:
```json
{"type": "execute", "cellId": "cell-2", "code": "..."}
```

**Server → Client** (streaming):
```json
{"type": "output", "cellId": "cell-2", "content": "Processing..."}
{"type": "output", "cellId": "cell-2", "content": "Done!"}
{"type": "complete", "cellId": "cell-2", "success": true}
```

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/search` | 60/min |
| `/labs/execute` | 30/min |
| All others | 120/min |

---

## OpenAPI Schema

Full OpenAPI 3.0 schema available at:
```
GET /openapi.json
```

Interactive documentation:
```
GET /docs      # Swagger UI
GET /redoc     # ReDoc
```

---

*API Specification for Luno-AI v1.0*
