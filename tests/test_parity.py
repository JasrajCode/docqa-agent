"""Checks that this server's search returns what DocQA's own retrieval returns.

Needs a fixture captured from DocQA (see README) plus real credentials in .env,
so it is skipped in CI and on machines without them.
"""

import json
from pathlib import Path

import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError

from docqa_agent.config import Settings
from docqa_agent.db import create_pool
from docqa_agent.embeddings import OpenAIEmbedder
from docqa_agent.repository import Repository

FIXTURE = Path(__file__).parent / "fixtures" / "parity.json"

pytestmark = pytest.mark.integration


def _settings() -> Settings | None:
    try:
        return Settings()  # pyright: ignore[reportCallIssue]
    except ValidationError:
        return None


async def test_search_matches_docqa_retrieval() -> None:
    settings = _settings()
    if not FIXTURE.exists() or settings is None:
        pytest.skip("needs tests/fixtures/parity.json and real credentials in .env")

    cases = json.loads(FIXTURE.read_text())
    pool = await create_pool(settings.database_url.get_secret_value())
    embedder = OpenAIEmbedder(
        AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value()),
        settings.embedding_model,
        settings.embedding_dimensions,
    )
    try:
        for case in cases:
            repo = Repository(pool, [case["user_id"]])
            embedding = await embedder.embed(case["query"])
            hits = await repo.search(embedding, case["top_k"], case["document_id"])

            ours = {h.chunk_id: h.similarity for h in hits}
            docqa = {r["id"]: r["similarity"] for r in case["results"]}
            assert set(ours) == set(docqa), case["query"]
            for chunk_id, similarity in docqa.items():
                # Embeddings can differ in the last few decimals between API calls.
                assert ours[chunk_id] == pytest.approx(similarity, abs=1e-3), case["query"]
    finally:
        await pool.close()
