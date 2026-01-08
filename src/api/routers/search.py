"""
Search API endpoints.
"""
from fastapi import APIRouter, Query
from typing import Optional

from ..services.search_service import search_service

router = APIRouter(prefix="/search", tags=["Search"])


@router.get("")
async def search(
    q: str = Query(..., description="Search query"),
    domains: Optional[str] = Query(None, description="Comma-separated domain IDs to filter"),
    type: str = Query("all", description="Search type: all, integrations, research"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results")
):
    """
    Search across integrations and research.

    Performs semantic search when ChromaDB is available,
    falls back to keyword search otherwise.
    """
    # Parse domains filter
    domain_list = None
    if domains:
        domain_list = [d.strip() for d in domains.split(",")]

    results = {
        "query": q,
        "type": type,
        "integrations": [],
        "research": []
    }

    if type in ("all", "integrations"):
        integration_results = search_service.semantic_search(
            query=q,
            domains=domain_list,
            limit=limit
        )
        results["integrations"] = integration_results

    if type in ("all", "research"):
        research_results = search_service.search_research(
            query=q,
            limit=limit
        )
        results["research"] = research_results

    results["total"] = len(results["integrations"]) + len(results["research"])

    return results


@router.get("/suggest")
async def search_suggestions(
    q: str = Query(..., min_length=2, description="Partial query for suggestions"),
    limit: int = Query(5, ge=1, le=20)
):
    """
    Get search suggestions based on partial query.

    Returns technology names and domains that match.
    """
    from ..config import settings
    from ..services.markdown_parser import markdown_parser

    q_lower = q.lower()
    suggestions = []

    # Search through technology names
    for domain_id in settings.domains:
        domain_name = settings.domains[domain_id]["name"]

        # Check domain name
        if q_lower in domain_name.lower():
            suggestions.append({
                "type": "domain",
                "id": domain_id,
                "name": domain_name,
                "match": "name"
            })

        # Check technologies
        technologies = markdown_parser.list_technologies(domain_id)
        for tech in technologies:
            if q_lower in tech.name.lower() or q_lower in tech.id.lower():
                suggestions.append({
                    "type": "technology",
                    "id": tech.id,
                    "domain": domain_id,
                    "name": tech.name,
                    "match": "name"
                })
            elif q_lower in tech.tagline.lower():
                suggestions.append({
                    "type": "technology",
                    "id": tech.id,
                    "domain": domain_id,
                    "name": tech.name,
                    "match": "tagline"
                })

    # Sort by match quality and limit
    suggestions.sort(key=lambda x: (x["match"] == "tagline", x["name"]))
    return {"suggestions": suggestions[:limit]}


@router.post("/index")
async def rebuild_index():
    """
    Rebuild the search index.

    Indexes all PRD files into ChromaDB for semantic search.
    """
    result = search_service.index_all_prds()
    return {
        "success": True,
        "message": "Index rebuilt",
        **result
    }


@router.get("/stats")
async def search_stats():
    """
    Get search index statistics.
    """
    search_service._init_chromadb()

    if not search_service._initialized:
        return {
            "status": "fallback",
            "message": "ChromaDB not available, using keyword search",
            "integrations_indexed": 0,
            "research_indexed": 0
        }

    integrations_count = search_service.integrations_collection.count()
    research_count = search_service.research_collection.count()

    return {
        "status": "active",
        "integrations_indexed": integrations_count,
        "research_indexed": research_count
    }
