from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from openai import AsyncOpenAI
from pydantic import Field

from docqa_agent.config import Settings
from docqa_agent.db import create_pool
from docqa_agent.embeddings import Embedder, OpenAIEmbedder
from docqa_agent.models import Chunk, ChunkHit, DocumentSummary
from docqa_agent.repository import Repository


@dataclass(frozen=True)
class AppContext:
    repo: Repository
    embedder: Embedder


DepsFactory = Callable[[], AbstractAsyncContextManager[AppContext]]


@asynccontextmanager
async def deps_from_settings() -> AsyncIterator[AppContext]:
    settings = Settings()  # pyright: ignore[reportCallIssue]  (fields are read from the environment)
    pool = await create_pool(
        settings.database_url.get_secret_value(), max_size=settings.db_pool_max_size
    )
    try:
        embedder = OpenAIEmbedder(
            AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value()),
            settings.embedding_model,
            settings.embedding_dimensions,
        )
        yield AppContext(Repository(pool, settings.docqa_scope_user_ids), embedder)
    finally:
        await pool.close()


READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)

INSTRUCTIONS = (
    "Read-only access to a corpus of PDF documents that were chunked and embedded by DocQA. "
    "Call list_documents to see what is available, search_documents to find passages relevant "
    "to a question, and get_chunk to reread a specific passage. When you use a passage in an "
    "answer, cite its chunk_id."
)


def create_server(deps: DepsFactory = deps_from_settings) -> MCPServer[AppContext]:
    @asynccontextmanager
    async def lifespan(_server: MCPServer[AppContext]) -> AsyncIterator[AppContext]:
        async with deps() as app:
            yield app

    server = MCPServer("docqa", instructions=INSTRUCTIONS, lifespan=lifespan)

    @server.tool(annotations=READ_ONLY)
    async def search_documents(
        query: str,
        ctx: Context[AppContext],
        k: Annotated[int, Field(ge=1, le=20)] = 5,
        document_id: str | None = None,
    ) -> list[ChunkHit]:
        """Semantic search over the document corpus.

        Returns up to k passages ranked by cosine similarity to the query (1.0 is an exact
        match), best first. Pass a document_id from list_documents to search one document.
        """
        if not query.strip():
            raise ToolError("query must not be empty")
        app = ctx.request_context.lifespan_context
        embedding = await app.embedder.embed(query)
        return await app.repo.search(embedding, k, document_id)

    @server.tool(annotations=READ_ONLY)
    async def get_chunk(chunk_id: str, ctx: Context[AppContext]) -> Chunk:
        """Fetch one passage by its chunk_id, for example to quote it exactly."""
        chunk = await ctx.request_context.lifespan_context.repo.get_chunk(chunk_id)
        if chunk is None:
            raise ToolError(f"Chunk not found: {chunk_id}")
        return chunk

    @server.tool(annotations=READ_ONLY)
    async def list_documents(ctx: Context[AppContext]) -> list[DocumentSummary]:
        """List the documents in the corpus with their status and number of chunks.

        Only documents with status READY are searchable.
        """
        return await ctx.request_context.lifespan_context.repo.list_documents()

    return server


def main() -> None:
    create_server().run()
