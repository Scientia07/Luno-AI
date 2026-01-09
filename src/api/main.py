"""
Luno-AI FastAPI Application
Educational AI technology exploration platform.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database.sqlite import init_database
from .routers import explore, progress, search, research, labs


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Integrations path: {settings.integrations_path}")
    print(f"Research path: {settings.research_path}")

    # Initialize database
    init_database()
    print("Database initialized")

    yield

    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title=settings.app_name,
    description="Educational AI technology exploration platform with layered learning (L0-L4).",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(explore.router, prefix=settings.api_prefix)
app.include_router(progress.router, prefix=settings.api_prefix)
app.include_router(search.router, prefix=settings.api_prefix)
app.include_router(research.router, prefix=settings.api_prefix)
app.include_router(labs.router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Educational AI technology exploration platform",
        "docs": "/docs",
        "api_prefix": settings.api_prefix,
        "endpoints": {
            "explore": f"{settings.api_prefix}/explore",
            "progress": f"{settings.api_prefix}/progress",
            "search": f"{settings.api_prefix}/search",
            "research": f"{settings.api_prefix}/research",
            "labs": f"{settings.api_prefix}/labs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from pathlib import Path

    # Check critical paths
    integrations_ok = settings.integrations_path.exists()
    research_ok = settings.research_path.exists()

    # Check database
    try:
        from .database.sqlite import get_db_path
        db_ok = get_db_path().parent.exists()
    except Exception:
        db_ok = False

    status = "healthy" if all([integrations_ok, research_ok, db_ok]) else "degraded"

    return {
        "status": status,
        "checks": {
            "integrations_path": integrations_ok,
            "research_path": research_ok,
            "database": db_ok
        }
    }


@app.get(f"{settings.api_prefix}/stats")
async def platform_stats():
    """Get platform-wide statistics."""
    from .services.markdown_parser import markdown_parser

    # Count technologies per domain
    domain_stats = {}
    total_tech = 0

    for domain_id, domain_config in settings.domains.items():
        technologies = markdown_parser.list_technologies(domain_id)
        count = len(technologies)
        total_tech += count
        domain_stats[domain_id] = {
            "name": domain_config["name"],
            "tech_count": count
        }

    # Count research sessions
    research_count = 0
    if settings.research_path.exists():
        research_count = sum(
            1 for d in settings.research_path.iterdir()
            if d.is_dir() and not d.name.startswith(("_", ".", "templates"))
        )

    # Count labs
    labs_count = 0
    if settings.labs_path.exists():
        labs_count = sum(
            1 for _ in settings.labs_path.rglob("*.ipynb")
            if ".ipynb_checkpoints" not in str(_)
        )

    return {
        "domains": len(settings.domains),
        "technologies": total_tech,
        "research_sessions": research_count,
        "labs": labs_count,
        "by_domain": domain_stats
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
