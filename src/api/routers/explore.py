"""
Explore API endpoints.
Technology browser for navigating domains and PRDs.
"""
from fastapi import APIRouter, HTTPException

from ..config import settings
from ..models.domain import (
    DomainSummary,
    DomainDetail,
    DomainsResponse,
    TechnologyDetail,
    TechnologyRawResponse,
)
from ..services.markdown_parser import markdown_parser

router = APIRouter(prefix="/explore", tags=["Explore"])


@router.get("/domains", response_model=DomainsResponse)
async def list_domains():
    """
    List all available technology domains.

    Returns summary information for each domain including
    name, description, icon, and technology count.
    """
    domains = []

    for domain_id, domain_config in settings.domains.items():
        # Count technologies in this domain
        technologies = markdown_parser.list_technologies(domain_id)

        domains.append(DomainSummary(
            id=domain_id,
            name=domain_config["name"],
            description=domain_config["description"],
            icon=domain_config["icon"],
            tech_count=len(technologies),
            color=domain_config["color"]
        ))

    # Sort by name
    domains.sort(key=lambda d: d.name)

    return DomainsResponse(domains=domains)


@router.get("/domains/{domain_id}", response_model=DomainDetail)
async def get_domain(domain_id: str):
    """
    Get detailed information about a specific domain.

    Includes list of all technologies in the domain with
    their summary information.
    """
    if domain_id not in settings.domains:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")

    domain_config = settings.domains[domain_id]
    technologies = markdown_parser.list_technologies(domain_id)

    return DomainDetail(
        id=domain_id,
        name=domain_config["name"],
        description=domain_config["description"],
        icon=domain_config["icon"],
        color=domain_config["color"],
        technologies=technologies,
        learning_path=None  # TODO: Add learning path lookup
    )


@router.get("/domains/{domain_id}/{tech_id}", response_model=TechnologyDetail)
async def get_technology(domain_id: str, tech_id: str):
    """
    Get full details for a specific technology.

    Returns complete PRD content including overview, prerequisites,
    quick start guide, learning layers (L0-L4), code examples,
    and external resources.
    """
    if domain_id not in settings.domains:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")

    prd_path = settings.integrations_path / domain_id / f"{tech_id}.md"

    if not prd_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Technology '{tech_id}' not found in domain '{domain_id}'"
        )

    technology = markdown_parser.parse_prd(prd_path)

    if not technology:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse PRD for '{tech_id}'"
        )

    return technology


@router.get("/domains/{domain_id}/{tech_id}/raw", response_model=TechnologyRawResponse)
async def get_technology_raw(domain_id: str, tech_id: str):
    """
    Get raw markdown content for a technology.

    Returns the original markdown and parsed frontmatter
    for custom rendering or processing.
    """
    if domain_id not in settings.domains:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")

    prd_path = settings.integrations_path / domain_id / f"{tech_id}.md"

    if not prd_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Technology '{tech_id}' not found in domain '{domain_id}'"
        )

    content = prd_path.read_text(encoding="utf-8")
    frontmatter, _ = markdown_parser.parse_frontmatter(content)

    return TechnologyRawResponse(
        markdown=content,
        frontmatter=frontmatter
    )


@router.get("/technologies")
async def list_all_technologies():
    """
    List all technologies across all domains.

    Returns a flat list of all technologies with their
    domain information for search and filtering.
    """
    all_technologies = []

    for domain_id in settings.domains:
        technologies = markdown_parser.list_technologies(domain_id)
        for tech in technologies:
            all_technologies.append({
                "domain": domain_id,
                "domain_name": settings.domains[domain_id]["name"],
                **tech.model_dump()
            })

    return {"technologies": all_technologies, "total": len(all_technologies)}


@router.get("/stats")
async def get_stats():
    """
    Get platform statistics.

    Returns counts of domains, technologies, and other metrics.
    """
    total_tech = 0
    domain_stats = {}

    for domain_id in settings.domains:
        technologies = markdown_parser.list_technologies(domain_id)
        count = len(technologies)
        total_tech += count
        domain_stats[domain_id] = {
            "name": settings.domains[domain_id]["name"],
            "count": count
        }

    return {
        "total_domains": len(settings.domains),
        "total_technologies": total_tech,
        "by_domain": domain_stats
    }
