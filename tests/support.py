import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from mcp.server.mcpserver import MCPServer

from docqa_agent.repository import Repository
from docqa_agent.server import AppContext, create_server

DIMS = 1536

# Users the server is configured to read. "user_other" exists in the seed data
# but is out of scope, so nothing of theirs should ever come back.
IN_SCOPE = ["user_demo", "user_eval"]


def basis(i: int) -> list[float]:
    v = [0.0] * DIMS
    v[i] = 1.0
    return v


def blend(i: int, j: int, weight_i: float) -> list[float]:
    """Unit vector whose cosine similarity with basis(i) is exactly weight_i."""
    v = [0.0] * DIMS
    v[i] = weight_i
    v[j] = math.sqrt(1 - weight_i**2)
    return v


# Seeded so a query of basis(0) ranks them: rag_0 (1.0), eval_0 (0.9), rag_1 (0.6), then 0.0s.
QUERY = "what is retrieval-augmented generation"


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    async def embed(self, text: str) -> list[float]:
        return self._vectors[text]


def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder({QUERY: basis(0)})


def make_test_server(pool: asyncpg.Pool) -> MCPServer[AppContext]:
    """The real MCP server, backed by the test database and a fake embedder."""

    @asynccontextmanager
    async def deps() -> AsyncIterator[AppContext]:
        yield AppContext(Repository(pool, IN_SCOPE), fake_embedder())

    return create_server(deps)
