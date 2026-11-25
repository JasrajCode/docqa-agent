import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

from mcp import StdioServerParameters
from mcp.client import Client
from mcp.server.mcpserver import MCPServer
from mcp_types import TextContent

from docqa_agent.models import ChunkHit

# langchain-mcp-adapters requires mcp<2 and this server is built on mcp 2, so the
# agent calls the server through mcp's own client.

_SERVER_ENV_VARS = ("DATABASE_URL", "OPENAI_API_KEY", "DOCQA_SCOPE_USER_IDS")


class SearchError(RuntimeError):
    pass


class Searcher(Protocol):
    async def search(self, query: str, k: int) -> list[ChunkHit]: ...


class MCPSearcher:
    """Searches through the docqa MCP server's search_documents tool."""

    def __init__(self, client: Client) -> None:
        self._client = client

    async def search(self, query: str, k: int) -> list[ChunkHit]:
        result = await self._client.call_tool("search_documents", {"query": query, "k": k})
        if result.is_error:
            message = " ".join(b.text for b in result.content if isinstance(b, TextContent))
            raise SearchError(message or "search_documents failed")
        return [ChunkHit.model_validate(hit) for hit in result.structured_content["result"]]


def local_server_params() -> StdioServerParameters:
    """The docqa MCP server as a subprocess, configured from .env or the environment."""
    env = {name: os.environ[name] for name in _SERVER_ENV_VARS if name in os.environ}
    return StdioServerParameters(command=sys.executable, args=["-m", "docqa_agent"], env=env)


@asynccontextmanager
async def connect(server: StdioServerParameters | MCPServer[Any]) -> AsyncIterator[MCPSearcher]:
    """Opens one MCP session for the whole agent run instead of one per tool call."""
    async with Client(server) as client:
        yield MCPSearcher(client)
