from datetime import UTC, datetime

import asyncpg
import pytest

from docqa_agent.repository import Repository, to_vector_literal
from tests.support import IN_SCOPE, basis


async def test_search_ranks_by_cosine_similarity(repo: Repository) -> None:
    hits = await repo.search(basis(0), k=3)
    assert [h.chunk_id for h in hits] == ["rag_0", "eval_0", "rag_1"]
    assert [h.similarity for h in hits] == pytest.approx([1.0, 0.9, 0.6])
    assert hits[0].document_title == "Retrieval-Augmented Generation"


async def test_search_never_returns_chunks_from_users_out_of_scope(repo: Repository) -> None:
    hits = await repo.search(basis(0), k=20)
    assert "private_0" not in {h.chunk_id for h in hits}


async def test_search_skips_unready_documents_and_unembedded_chunks(repo: Repository) -> None:
    hits = await repo.search(basis(0), k=20)
    assert {h.chunk_id for h in hits} == {"rag_0", "rag_1", "rag_2", "eval_0", "eval_1"}


async def test_search_can_be_limited_to_one_document(repo: Repository) -> None:
    hits = await repo.search(basis(0), k=20, document_id="doc_eval")
    assert [h.chunk_id for h in hits] == ["eval_0", "eval_1"]


async def test_search_in_an_out_of_scope_document_returns_nothing(repo: Repository) -> None:
    assert await repo.search(basis(0), k=20, document_id="doc_private") == []


async def test_get_chunk_returns_an_in_scope_chunk(repo: Repository) -> None:
    chunk = await repo.get_chunk("rag_1")
    assert chunk is not None
    assert (chunk.document_id, chunk.chunk_index) == ("doc_rag", 1)
    assert chunk.content == "The retriever is a dense passage index."


async def test_get_chunk_treats_out_of_scope_and_missing_ids_the_same(repo: Repository) -> None:
    assert await repo.get_chunk("private_0") is None
    assert await repo.get_chunk("does-not-exist") is None


async def test_list_documents_only_shows_in_scope_documents_newest_first(repo: Repository) -> None:
    docs = await repo.list_documents()
    assert [d.document_id for d in docs] == ["doc_eval", "doc_rag", "doc_draft"]
    assert {d.document_id: d.chunk_count for d in docs} == {
        "doc_eval": 2,
        "doc_rag": 4,
        "doc_draft": 1,
    }
    assert {d.document_id: d.status for d in docs}["doc_draft"] == "PROCESSING"
    assert docs[0].created_at == datetime(2026, 1, 3, tzinfo=UTC)


async def test_repository_refuses_an_empty_scope(pool: asyncpg.Pool) -> None:
    with pytest.raises(ValueError):
        Repository(pool, [])
    assert Repository(pool, IN_SCOPE)


def test_vector_literal_matches_pgvector_text_format() -> None:
    assert to_vector_literal([1, 0.5, -2]) == "[1.0,0.5,-2.0]"
