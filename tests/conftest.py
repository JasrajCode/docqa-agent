import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from pathlib import Path

import asyncpg
import pytest
from testcontainers.community.postgres import PostgresContainer

from docqa_agent.db import create_pool
from docqa_agent.repository import Repository, to_vector_literal
from tests.support import IN_SCOPE, basis, blend

SCHEMA = Path(__file__).parent / "schema.sql"

USERS = [
    ("user_demo", "demo@example.com"),
    ("user_eval", "eval@example.com"),
    ("user_other", "other@example.com"),
]

DOCUMENTS = [
    ("doc_rag", "user_demo", "Retrieval-Augmented Generation", "READY", datetime(2026, 1, 2)),
    ("doc_eval", "user_eval", "Evaluation Notes", "READY", datetime(2026, 1, 3)),
    ("doc_draft", "user_demo", "Draft Upload", "PROCESSING", datetime(2026, 1, 1)),
    ("doc_private", "user_other", "Someone Else's Notes", "READY", datetime(2026, 1, 4)),
]

CHUNKS = [
    ("rag_0", "doc_rag", 0, "RAG pairs a retriever with a generator.", basis(0)),
    ("rag_1", "doc_rag", 1, "The retriever is a dense passage index.", blend(0, 1, 0.6)),
    ("rag_2", "doc_rag", 2, "Training uses a joint objective.", basis(1)),
    ("rag_3", "doc_rag", 3, "A chunk that was never embedded.", None),
    ("eval_0", "doc_eval", 0, "Evaluation notes on RAG benchmarks.", blend(0, 2, 0.9)),
    ("eval_1", "doc_eval", 1, "Latency measurements.", basis(2)),
    ("draft_0", "doc_draft", 0, "Still processing.", basis(0)),
    ("private_0", "doc_private", 0, "Private notes about RAG.", basis(0)),
]


async def _seed(url: str) -> None:
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(SCHEMA.read_text())
        await conn.executemany('INSERT INTO "User" (id, email) VALUES ($1, $2)', USERS)
        await conn.executemany(
            'INSERT INTO "Document" (id, "userId", title, "s3Key", status, "createdAt") '
            "VALUES ($1, $2, $3, 'test-key', $4, $5)",
            DOCUMENTS,
        )
        await conn.executemany(
            'INSERT INTO "Chunk" (id, "documentId", "chunkIndex", content, embedding) '
            "VALUES ($1, $2, $3, $4, $5::text::vector)",
            [
                (cid, doc, idx, text, to_vector_literal(vec) if vec else None)
                for cid, doc, idx, text, vec in CHUNKS
            ],
        )
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    with PostgresContainer("pgvector/pgvector:pg16", driver=None) as pg:
        url = pg.get_connection_url()
        asyncio.run(_seed(url))
        yield url


@pytest.fixture
async def pool(pg_url: str) -> AsyncIterator[asyncpg.Pool]:
    pool = await create_pool(pg_url)
    yield pool
    await pool.close()


@pytest.fixture
def repo(pool: asyncpg.Pool) -> Repository:
    return Repository(pool, IN_SCOPE)
