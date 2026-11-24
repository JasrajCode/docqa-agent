from typing import Protocol

from openai import AsyncOpenAI


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


def prepare_for_embedding(text: str) -> str:
    # Mirrors DocQA's lib/embeddings.ts, which embeds text.replace(/\n/g, " ").
    return text.replace("\n", " ")


class OpenAIEmbedder:
    def __init__(self, client: AsyncOpenAI, model: str, dimensions: int) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=prepare_for_embedding(text),
            dimensions=self._dimensions,
        )
        vector = response.data[0].embedding
        if len(vector) != self._dimensions:
            raise ValueError(f"expected a {self._dimensions}-dim embedding, got {len(vector)}")
        return vector
