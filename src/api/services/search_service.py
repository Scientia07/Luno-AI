"""
Search service with ChromaDB for semantic search.
"""
from pathlib import Path
from typing import Optional
import re

from ..config import settings


class SearchService:
    """Service for semantic and keyword search across PRDs and research."""

    def __init__(self):
        self.chroma_client = None
        self.integrations_collection = None
        self.research_collection = None
        self._initialized = False

    def _init_chromadb(self):
        """Lazily initialize ChromaDB client and collections."""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            persist_dir = settings.project_root / settings.chroma_persist_dir
            persist_dir.mkdir(parents=True, exist_ok=True)

            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            # Get or create collections
            self.integrations_collection = self.chroma_client.get_or_create_collection(
                name="integrations",
                metadata={"description": "PRD content embeddings"}
            )

            self.research_collection = self.chroma_client.get_or_create_collection(
                name="research",
                metadata={"description": "Research session embeddings"}
            )

            self._initialized = True
        except ImportError:
            # ChromaDB not installed, use fallback
            self._initialized = False

    def index_prd(self, prd_path: Path) -> int:
        """Index a PRD file for semantic search."""
        self._init_chromadb()

        if not self._initialized or not prd_path.exists():
            return 0

        content = prd_path.read_text(encoding="utf-8")
        domain = prd_path.parent.name
        tech_id = prd_path.stem

        # Split content into chunks
        chunks = self._split_content(content)
        indexed = 0

        for i, chunk in enumerate(chunks):
            doc_id = f"{domain}_{tech_id}_{i}"
            self.integrations_collection.upsert(
                ids=[doc_id],
                documents=[chunk["text"]],
                metadatas=[{
                    "domain": domain,
                    "technology": tech_id,
                    "section": chunk.get("section", ""),
                    "path": str(prd_path)
                }]
            )
            indexed += 1

        return indexed

    def index_all_prds(self) -> dict:
        """Index all PRD files."""
        self._init_chromadb()

        if not self._initialized:
            return {"error": "ChromaDB not available", "indexed": 0}

        total_indexed = 0
        domain_counts = {}

        for domain_id in settings.domains:
            domain_path = settings.integrations_path / domain_id
            if not domain_path.exists():
                continue

            domain_indexed = 0
            for prd_file in domain_path.glob("*.md"):
                if prd_file.name.startswith("_"):
                    continue
                count = self.index_prd(prd_file)
                domain_indexed += count
                total_indexed += count

            domain_counts[domain_id] = domain_indexed

        return {
            "total_indexed": total_indexed,
            "by_domain": domain_counts
        }

    def semantic_search(
        self,
        query: str,
        domains: Optional[list[str]] = None,
        limit: int = 10
    ) -> list[dict]:
        """Perform semantic search across PRDs."""
        self._init_chromadb()

        if not self._initialized:
            # Fallback to keyword search
            return self.keyword_search(query, domains, limit)

        # Build where clause
        where = None
        if domains:
            where = {"domain": {"$in": domains}}

        results = self.integrations_collection.query(
            query_texts=[query],
            n_results=limit,
            where=where
        )

        # Format results
        formatted = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0
                })

        return formatted

    def keyword_search(
        self,
        query: str,
        domains: Optional[list[str]] = None,
        limit: int = 10
    ) -> list[dict]:
        """Fallback keyword search across PRD files."""
        results = []
        search_domains = domains or list(settings.domains.keys())

        query_lower = query.lower()
        query_words = query_lower.split()

        for domain_id in search_domains:
            domain_path = settings.integrations_path / domain_id
            if not domain_path.exists():
                continue

            for prd_file in domain_path.glob("*.md"):
                if prd_file.name.startswith("_"):
                    continue

                content = prd_file.read_text(encoding="utf-8")
                content_lower = content.lower()

                # Score based on word matches
                score = sum(
                    content_lower.count(word) for word in query_words
                )

                if score > 0:
                    # Extract matching snippet
                    snippet = self._extract_snippet(content, query)

                    results.append({
                        "id": f"{domain_id}_{prd_file.stem}",
                        "domain": domain_id,
                        "technology": prd_file.stem,
                        "snippet": snippet,
                        "score": score,
                        "path": str(prd_file)
                    })

        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def search_research(
        self,
        query: str,
        limit: int = 10
    ) -> list[dict]:
        """Search across research sessions."""
        results = []
        research_path = settings.research_path

        if not research_path.exists():
            return results

        query_lower = query.lower()
        query_words = query_lower.split()

        for session_dir in research_path.iterdir():
            if not session_dir.is_dir() or session_dir.name.startswith(("_", ".")):
                continue

            # Search in README.md and findings.md
            for filename in ["README.md", "findings.md", "sources.md"]:
                file_path = session_dir / filename
                if not file_path.exists():
                    continue

                content = file_path.read_text(encoding="utf-8")
                content_lower = content.lower()

                score = sum(
                    content_lower.count(word) for word in query_words
                )

                if score > 0:
                    snippet = self._extract_snippet(content, query)
                    results.append({
                        "session": session_dir.name,
                        "file": filename,
                        "snippet": snippet,
                        "score": score,
                        "path": str(file_path)
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _split_content(self, content: str) -> list[dict]:
        """Split markdown content into chunks for indexing."""
        chunks = []

        # Split by headers
        sections = re.split(r"^(#{1,3}\s+.+)$", content, flags=re.MULTILINE)

        current_section = ""
        current_text = ""

        for part in sections:
            if re.match(r"^#{1,3}\s+", part):
                # This is a header
                if current_text.strip():
                    chunks.append({
                        "section": current_section,
                        "text": current_text.strip()[:2000]  # Limit chunk size
                    })
                current_section = part.strip("# \n")
                current_text = ""
            else:
                current_text += part

        # Add final chunk
        if current_text.strip():
            chunks.append({
                "section": current_section,
                "text": current_text.strip()[:2000]
            })

        return chunks

    def _extract_snippet(self, content: str, query: str, context_chars: int = 200) -> str:
        """Extract a snippet around the first query match."""
        query_lower = query.lower()
        content_lower = content.lower()

        # Find first match
        pos = content_lower.find(query_lower)

        if pos == -1:
            # Try individual words
            for word in query_lower.split():
                pos = content_lower.find(word)
                if pos != -1:
                    break

        if pos == -1:
            # Return beginning of content
            return content[:context_chars * 2] + "..."

        # Extract context around match
        start = max(0, pos - context_chars)
        end = min(len(content), pos + len(query) + context_chars)

        snippet = content[start:end]

        # Clean up
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet


# Singleton instance
search_service = SearchService()
