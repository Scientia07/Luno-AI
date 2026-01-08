"""
Markdown parser service for PRD files.
Extracts structured content from integration PRDs.
"""
import re
from pathlib import Path
from typing import Optional
import yaml

from ..config import settings
from ..models.domain import (
    TechnologyDetail,
    TechnologySummary,
    Overview,
    Prerequisite,
    QuickStart,
    Layer,
    CodeExample,
    Resource,
)


class MarkdownParser:
    """Parser for PRD markdown files."""

    def __init__(self):
        self.integrations_path = settings.integrations_path

    def parse_frontmatter(self, content: str) -> tuple[dict, str]:
        """Extract YAML frontmatter and body from markdown."""
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1])
                    body = parts[2].strip()
                    return frontmatter or {}, body
                except yaml.YAMLError:
                    pass
        return {}, content

    def extract_section(self, content: str, header: str) -> Optional[str]:
        """Extract content under a specific header."""
        pattern = rf"^##\s+{re.escape(header)}\s*\n(.*?)(?=^##\s|\Z)"
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def extract_subsection(self, content: str, header: str) -> Optional[str]:
        """Extract content under a ### header."""
        pattern = rf"^###\s+{re.escape(header)}\s*\n(.*?)(?=^###\s|^##\s|\Z)"
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def extract_code_blocks(self, content: str) -> list[tuple[str, str, str]]:
        """Extract all code blocks with language and optional title."""
        pattern = r"```(\w+)?\s*(?:#\s*(.+?))?\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)
        return [(lang or "text", title or "", code.strip()) for lang, title, code in matches]

    def extract_list_items(self, content: str) -> list[str]:
        """Extract bullet points from content."""
        pattern = r"^[-*]\s+(.+)$"
        matches = re.findall(pattern, content, re.MULTILINE)
        return matches

    def extract_numbered_items(self, content: str) -> list[str]:
        """Extract numbered list items."""
        pattern = r"^\d+\.\s+(.+)$"
        matches = re.findall(pattern, content, re.MULTILINE)
        return matches

    def parse_overview(self, content: str) -> Overview:
        """Parse the Overview section."""
        overview_section = self.extract_section(content, "Overview") or ""

        # Try to extract What/Why/Tools/Best For
        what = self.extract_subsection(overview_section, "What is it") or ""
        if not what:
            # Fall back to first paragraph
            paragraphs = overview_section.split("\n\n")
            what = paragraphs[0] if paragraphs else ""

        why = self.extract_subsection(overview_section, "Why use it") or ""

        # Extract tools list
        tools_section = self.extract_subsection(overview_section, "Key Tools") or ""
        tools = self.extract_list_items(tools_section) if tools_section else []

        best_for = self.extract_subsection(overview_section, "Best For") or ""

        return Overview(
            what=what,
            why=why,
            tools=tools,
            best_for=best_for
        )

    def parse_prerequisites(self, content: str) -> list[Prerequisite]:
        """Parse prerequisites section."""
        prereq_section = self.extract_section(content, "Prerequisites") or ""
        items = self.extract_list_items(prereq_section)

        prerequisites = []
        for item in items:
            # Try to split on colon or dash
            if ":" in item:
                name, details = item.split(":", 1)
            elif " - " in item:
                name, details = item.split(" - ", 1)
            else:
                name = item
                details = ""
            prerequisites.append(Prerequisite(name=name.strip(), details=details.strip()))

        return prerequisites

    def parse_quick_start(self, content: str) -> QuickStart:
        """Parse quick start section."""
        quick_start = self.extract_section(content, "Quick Start") or ""

        # Extract time estimate
        time_pattern = r"(\d+\s*(?:minutes?|mins?|hours?|hrs?))"
        time_match = re.search(time_pattern, quick_start, re.IGNORECASE)
        time_estimate = time_match.group(1) if time_match else "5 minutes"

        # Extract install commands
        code_blocks = self.extract_code_blocks(quick_start)
        install_code = ""
        example_code = ""

        for lang, title, code in code_blocks:
            if lang in ("bash", "shell", "sh") or "install" in title.lower():
                install_code = code
            elif lang in ("python", "javascript", "typescript"):
                example_code = code

        return QuickStart(
            time=time_estimate,
            install=install_code,
            code=example_code
        )

    def parse_layers(self, content: str) -> list[Layer]:
        """Parse learning layers (L0-L4)."""
        layers = []

        layer_patterns = [
            (0, r"L0|Layer\s*0|Overview|Introduction"),
            (1, r"L1|Layer\s*1|Getting Started|Basics"),
            (2, r"L2|Layer\s*2|Intermediate|Core Concepts"),
            (3, r"L3|Layer\s*3|Advanced|Deep Dive"),
            (4, r"L4|Layer\s*4|Expert|Production|Optimization"),
        ]

        layer_names = [
            "Overview",
            "Getting Started",
            "Core Concepts",
            "Advanced Topics",
            "Production & Optimization"
        ]

        for level, pattern in layer_patterns:
            # Look for layer sections
            section_pattern = rf"^##\s+.*?({pattern}).*?\n(.*?)(?=^##\s|\Z)"
            match = re.search(section_pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE)

            if match:
                layer_content = match.group(2).strip()
                checklist = self.extract_list_items(layer_content)

                layers.append(Layer(
                    level=level,
                    name=layer_names[level],
                    content=layer_content,
                    checklist_items=checklist[:10]  # Limit checklist items
                ))

        # If no layers found, create a default L0 from overview
        if not layers:
            overview = self.extract_section(content, "Overview") or content[:500]
            layers.append(Layer(
                level=0,
                name="Overview",
                content=overview,
                checklist_items=[]
            ))

        return layers

    def parse_code_examples(self, content: str) -> list[CodeExample]:
        """Extract code examples from the PRD."""
        examples = []
        code_blocks = self.extract_code_blocks(content)

        for i, (lang, title, code) in enumerate(code_blocks):
            if len(code) > 20:  # Skip very short snippets
                examples.append(CodeExample(
                    title=title or f"Example {i + 1}",
                    language=lang,
                    code=code
                ))

        return examples[:10]  # Limit to 10 examples

    def parse_resources(self, content: str) -> list[Resource]:
        """Extract resource links."""
        resources = []

        # Look for links in Resources/References sections
        resources_section = (
            self.extract_section(content, "Resources") or
            self.extract_section(content, "References") or
            self.extract_section(content, "Further Reading") or
            ""
        )

        # Extract markdown links
        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        matches = re.findall(link_pattern, resources_section)

        for title, url in matches:
            if url.startswith("http"):
                resources.append(Resource(title=title, url=url))

        return resources[:20]  # Limit resources

    def parse_related_tech(self, content: str, frontmatter: dict) -> list[str]:
        """Extract related technologies."""
        # Try frontmatter first
        if "related" in frontmatter:
            return frontmatter["related"]

        # Look for Related section
        related_section = self.extract_section(content, "Related") or ""
        return self.extract_list_items(related_section)

    def parse_prd(self, prd_path: Path) -> Optional[TechnologyDetail]:
        """Parse a complete PRD file into structured data."""
        if not prd_path.exists():
            return None

        content = prd_path.read_text(encoding="utf-8")
        frontmatter, body = self.parse_frontmatter(content)

        # Extract domain from path
        domain = prd_path.parent.name
        tech_id = prd_path.stem

        # Get name from frontmatter or first H1
        name = frontmatter.get("title", "")
        if not name:
            h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
            name = h1_match.group(1) if h1_match else tech_id.replace("-", " ").title()

        # Get tagline
        tagline = frontmatter.get("tagline", "")
        if not tagline:
            # Use first paragraph after H1
            first_para = re.search(r"^#\s+.+\n\n(.+?)(?:\n\n|\Z)", body, re.MULTILINE)
            tagline = first_para.group(1)[:200] if first_para else ""

        return TechnologyDetail(
            id=tech_id,
            domain=domain,
            name=name,
            tagline=tagline,
            overview=self.parse_overview(body),
            prerequisites=self.parse_prerequisites(body),
            quick_start=self.parse_quick_start(body),
            layers=self.parse_layers(body),
            code_examples=self.parse_code_examples(body),
            related_tech=self.parse_related_tech(body, frontmatter),
            resources=self.parse_resources(body),
            raw_content=content
        )

    def get_technology_summary(self, prd_path: Path) -> Optional[TechnologySummary]:
        """Get a lightweight summary of a technology."""
        if not prd_path.exists():
            return None

        content = prd_path.read_text(encoding="utf-8")
        frontmatter, body = self.parse_frontmatter(content)

        tech_id = prd_path.stem

        # Get name
        name = frontmatter.get("title", "")
        if not name:
            h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
            name = h1_match.group(1) if h1_match else tech_id.replace("-", " ").title()

        # Get tagline
        tagline = frontmatter.get("tagline", "")
        if not tagline:
            first_para = re.search(r"^#\s+.+\n\n(.+?)(?:\n\n|\Z)", body, re.MULTILINE)
            tagline = first_para.group(1)[:200] if first_para else ""

        # Get difficulty and status from frontmatter
        difficulty = frontmatter.get("difficulty", "intermediate")
        status = frontmatter.get("status", "ready")
        quick_start_time = frontmatter.get("quick_start_time")

        return TechnologySummary(
            id=tech_id,
            name=name,
            tagline=tagline,
            difficulty=difficulty,
            status=status,
            quick_start_time=quick_start_time
        )

    def list_technologies(self, domain: str) -> list[TechnologySummary]:
        """List all technologies in a domain."""
        domain_path = self.integrations_path / domain
        if not domain_path.exists():
            return []

        technologies = []
        for prd_file in domain_path.glob("*.md"):
            if prd_file.name.startswith("_"):
                continue  # Skip index files
            summary = self.get_technology_summary(prd_file)
            if summary:
                technologies.append(summary)

        return technologies

    def list_domains(self) -> list[str]:
        """List all available domains."""
        domains = []
        for item in self.integrations_path.iterdir():
            if item.is_dir() and not item.name.startswith(("_", ".")):
                domains.append(item.name)
        return sorted(domains)


# Singleton instance
markdown_parser = MarkdownParser()
