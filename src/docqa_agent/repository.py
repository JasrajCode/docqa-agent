from collections.abc import Sequence

import asyncpg

from docqa_agent.models import Chunk, ChunkHit, DocumentSummary

# Table and column names come from DocQA's Prisma schema, hence the quoted camelCase.
# Same cosine ranking as DocQA's lib/retrieval.ts, plus a READY filter so the
# server never serves a half-processed document.
_SEARCH_SQL = """
SELECT c.id AS chunk_id,
       c."documentId" AS document_id,
       d.title AS document_title,
       c."chunkIndex" AS chunk_index,
       1 - (c.embedding <=> $1::text::vector) AS similarity,
       c.content
FROM "Chunk" c
JOIN "Document" d ON d.id = c."documentId"
WHERE d."userId" = ANY($2::text[])
  AND d.status = 'READY'
  AND c.embedding IS NOT NULL
  AND ($3::text IS NULL OR c."documentId" = $3::text)
ORDER BY c.embedding <=> $1::text::vector
LIMIT $4
"""

_GET_CHUNK_SQL = """
SELECT c.id AS chunk_id,
       c."documentId" AS document_id,
       d.title AS document_title,
       c."chunkIndex" AS chunk_index,
       c.content
FROM "Chunk" c
JOIN "Document" d ON d.id = c."documentId"
WHERE c.id = $1 AND d."userId" = ANY($2::text[])
"""

_LIST_DOCUMENTS_SQL = """
SELECT d.id AS document_id,
       d.title,
       d.status::text AS status,
       COUNT(c.id)::int AS chunk_count,
       -- Prisma writes UTC into a column without a time zone. Marking it as UTC makes the
       -- output a valid JSON Schema date-time, which strict MCP clients check.
       d."createdAt" AT TIME ZONE 'UTC' AS created_at
FROM "Document" d
LEFT JOIN "Chunk" c ON c."documentId" = d.id
WHERE d."userId" = ANY($1::text[])
GROUP BY d.id
ORDER BY d."createdAt" DESC
"""


def to_vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


class Repository:
    """Read-only queries over DocQA's tables, limited to a fixed set of users."""

    def __init__(self, pool: asyncpg.Pool, scope_user_ids: Sequence[str]) -> None:
        if not scope_user_ids:
            raise ValueError("scope_user_ids must not be empty")
        self._pool = pool
        self._scope = list(scope_user_ids)

    async def search(
        self, embedding: Sequence[float], k: int, document_id: str | None = None
    ) -> list[ChunkHit]:
        rows = await self._pool.fetch(
            _SEARCH_SQL, to_vector_literal(embedding), self._scope, document_id, k
        )
        return [ChunkHit(**dict(row)) for row in rows]

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        # Out-of-scope and nonexistent chunks both come back as None,
        # so callers can't probe for other users' chunk ids.
        row = await self._pool.fetchrow(_GET_CHUNK_SQL, chunk_id, self._scope)
        return Chunk(**dict(row)) if row else None

    async def list_documents(self) -> list[DocumentSummary]:
        rows = await self._pool.fetch(_LIST_DOCUMENTS_SQL, self._scope)
        return [DocumentSummary(**dict(row)) for row in rows]
