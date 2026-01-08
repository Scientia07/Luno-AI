# Model Context Protocol (MCP) Integration

> **Universal tool integration standard for AI agents**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Open protocol for connecting AI applications to external tools and data |
| **Why** | Solves N×M integration problem → N+M (10 apps × 20 tools = 200 → 30) |
| **Created** | Anthropic (Nov 2024), now Linux Foundation governed |
| **Adoption** | 8M+ downloads, 5,800+ servers, used by Claude, ChatGPT, Gemini, Copilot |

### Key Capabilities
- **Tools**: Functions that perform actions (POST-like)
- **Resources**: Data endpoints (GET-like)
- **Prompts**: Reusable interaction templates
- **Transport**: stdio (local) or Streamable HTTP (remote)

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10+ |
| **Node.js** | 18+ (for TypeScript) |
| **Hardware** | Any (protocol is lightweight) |
| **Knowledge** | Basic async Python or TypeScript |

---

## Quick Start (15 min)

### Python Server

```bash
# Install
pip install mcp

# Or with uv (recommended)
uv add "mcp[cli]"
```

```python
# hello_server.py
from mcp import FastMCP

mcp = FastMCP("Hello Server")

@mcp.tool()
async def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

@mcp.resource("info://version")
async def get_version() -> str:
    """Get server version."""
    return "1.0.0"

if __name__ == "__main__":
    mcp.run()
```

```bash
# Test with MCP Inspector
mcp dev hello_server.py

# Install for Claude Desktop
mcp install hello_server.py
```

### TypeScript Server

```bash
npm install @modelcontextprotocol/sdk zod
```

```typescript
// hello_server.ts
import { McpServer, StdioServerTransport } from "@modelcontextprotocol/sdk";
import { z } from "zod";

const server = new McpServer({
  name: "hello-server",
  version: "1.0.0",
});

server.tool(
  "greet",
  "Greet someone by name",
  { name: z.string() },
  async ({ name }) => ({
    content: [{ type: "text", text: `Hello, ${name}!` }]
  })
);

const transport = new StdioServerTransport();
server.connect(transport);
```

---

## Full Setup

### Claude Desktop Configuration

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "hello": {
      "command": "python",
      "args": ["/path/to/hello_server.py"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
    }
  }
}
```

### Popular Pre-built Servers

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/docs"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "ghp_xxx" }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": { "DATABASE_URL": "postgresql://..." }
    },
    "slack": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-slack"],
      "env": { "SLACK_BOT_TOKEN": "xoxb-xxx" }
    }
  }
}
```

---

## Learning Path

### L0: Basic Usage (1 hour)
- [ ] Install MCP Python SDK
- [ ] Run `mcp dev` with sample server
- [ ] Test tools in MCP Inspector
- [ ] Configure Claude Desktop

### L1: Build Your First Server (2-3 hours)
- [ ] Create server with 2-3 tools
- [ ] Add resources for data retrieval
- [ ] Add prompt templates
- [ ] Test end-to-end with Claude

### L2: Production Patterns (4-6 hours)
- [ ] Implement error handling
- [ ] Add input validation
- [ ] Configure logging
- [ ] Handle authentication
- [ ] Deploy HTTP transport

### L3: Advanced Integration (1-2 days)
- [ ] Multi-server architecture
- [ ] LangGraph/LangChain integration
- [ ] OAuth 2.1 implementation
- [ ] Build custom MCP client

---

## Code Examples

### Database Query Server

```python
from mcp import FastMCP
import sqlite3

mcp = FastMCP("Database Server")

@mcp.tool()
async def query_database(sql: str) -> list[dict]:
    """
    Execute a read-only SQL query.

    Args:
        sql: SELECT query to execute (no modifications allowed)
    """
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries allowed")

    conn = sqlite3.connect("data.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(sql)
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results

@mcp.resource("schema://tables")
async def list_tables() -> str:
    """List all tables in the database."""
    conn = sqlite3.connect("data.db")
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return "\n".join(tables)
```

### Web Scraper Server

```python
from mcp import FastMCP
import httpx
from bs4 import BeautifulSoup

mcp = FastMCP("Web Scraper")

@mcp.tool()
async def fetch_webpage(url: str) -> dict:
    """
    Fetch and parse a webpage.

    Args:
        url: The URL to fetch (must be https)
    """
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS URLs allowed")

    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
        soup = BeautifulSoup(response.text, "html.parser")

        return {
            "title": soup.title.string if soup.title else None,
            "text": soup.get_text()[:5000],  # Limit text length
            "links": [a.get("href") for a in soup.find_all("a", href=True)][:20]
        }
```

### LangGraph Integration

```python
from langchain_mcp_adapters import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Connect to MCP servers
async with MultiServerMCPClient({
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/docs"]
    },
    "database": {
        "command": "python",
        "args": ["db_server.py"]
    }
}) as client:
    # Get tools from all servers
    tools = client.get_tools()

    # Create LangGraph agent with MCP tools
    llm = ChatOpenAI(model="gpt-4o")
    agent = create_react_agent(llm, tools)

    # Run agent
    result = await agent.ainvoke({
        "messages": [("user", "Find all PDF files and summarize them")]
    })
```

---

## Integration Points

### Transport Options

| Transport | Use Case | Setup |
|-----------|----------|-------|
| **stdio** | Local processes | Default, simplest |
| **Streamable HTTP** | Remote servers | Requires HTTP server |

### Connecting to Frameworks

| Framework | Integration |
|-----------|-------------|
| **LangChain** | `langchain-mcp-adapters` package |
| **LangGraph** | Via LangChain tools |
| **Claude Desktop** | Native support via config |
| **ChatGPT** | Native MCP support |
| **Custom** | Use MCP Client SDK |

---

## Security Considerations

### Critical Security Rules

```python
# 1. VALIDATE ALL INPUTS
@mcp.tool()
async def read_file(path: str) -> str:
    # Prevent path traversal
    if ".." in path or path.startswith("/"):
        raise ValueError("Invalid path")

    allowed_dir = Path("/safe/directory")
    full_path = (allowed_dir / path).resolve()

    if not full_path.is_relative_to(allowed_dir):
        raise ValueError("Path outside allowed directory")

    return full_path.read_text()

# 2. LIMIT CAPABILITIES
@mcp.tool()
async def execute_query(sql: str) -> list:
    # Only allow SELECT, never DELETE/UPDATE/DROP
    if not sql.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries allowed")
    # ...

# 3. REQUIRE HUMAN APPROVAL FOR SENSITIVE ACTIONS
# (Implement in your MCP client, not server)
```

### Security Checklist

- [ ] Validate all user inputs
- [ ] Use allowlists, not blocklists
- [ ] Limit file system access to specific directories
- [ ] Restrict database queries to read-only
- [ ] Implement rate limiting
- [ ] Log all tool invocations
- [ ] Require human approval for sensitive actions
- [ ] Use OAuth 2.1 for remote servers

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| Server not appearing in Claude | Check config JSON syntax, restart Claude |
| "Connection refused" | Ensure server is running, check transport |
| Tools not showing | Verify `@mcp.tool()` decorator, check logs |
| Permission denied | Check file paths, environment variables |

### Debugging

```bash
# Enable verbose logging
export MCP_DEBUG=1
python server.py

# Use MCP Inspector
mcp dev server.py

# Check Claude Desktop logs (macOS)
tail -f ~/Library/Logs/Claude/mcp*.log
```

---

## Resources

### Official
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [Official Servers](https://github.com/modelcontextprotocol/servers)

### Community
- [MCP Server Registry](https://github.com/punkpeye/awesome-mcp-servers)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)

### Tutorials
- [Building Your First MCP Server](https://modelcontextprotocol.io/quickstart)
- [Claude Desktop MCP Setup](https://modelcontextprotocol.io/quickstart/user)

---

*Part of [Luno-AI](../../README.md) | Agentic AI Track*
