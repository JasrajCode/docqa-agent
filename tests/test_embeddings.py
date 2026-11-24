from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI

from docqa_agent.embeddings import OpenAIEmbedder


class FakeEmbeddingsAPI:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=self.vector)])


def _embedder(vector: list[float], dimensions: int) -> tuple[OpenAIEmbedder, FakeEmbeddingsAPI]:
    api = FakeEmbeddingsAPI(vector)
    client = cast(AsyncOpenAI, SimpleNamespace(embeddings=api))
    return OpenAIEmbedder(client, "text-embedding-3-small", dimensions), api


async def test_matches_docqa_model_dimensions_and_preprocessing() -> None:
    embedder, api = _embedder([0.1, 0.2, 0.3], dimensions=3)
    assert await embedder.embed("line one\nline two\n") == [0.1, 0.2, 0.3]
    assert api.calls == [
        {"model": "text-embedding-3-small", "input": "line one line two ", "dimensions": 3}
    ]


async def test_rejects_an_embedding_with_the_wrong_size() -> None:
    embedder, _ = _embedder([0.1, 0.2], dimensions=1536)
    with pytest.raises(ValueError, match="1536-dim"):
        await embedder.embed("hello")
