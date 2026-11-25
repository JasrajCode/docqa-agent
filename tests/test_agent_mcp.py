import asyncpg
import pytest

from docqa_agent.agent.graph import answer_question, build_graph
from docqa_agent.agent.search import SearchError, connect
from docqa_agent.agent.state import Claim, Draft, Plan, Review
from tests.support import QUERY, make_test_server
from tests.test_agent_graph import ScriptedReasoner


async def test_mcp_searcher_parses_results_from_the_real_server(pool: asyncpg.Pool) -> None:
    async with connect(make_test_server(pool)) as searcher:
        hits = await searcher.search(QUERY, 2)

    assert [h.chunk_id for h in hits] == ["rag_0", "eval_0"]
    assert hits[0].document_title == "Retrieval-Augmented Generation"


async def test_mcp_searcher_raises_on_a_tool_error(pool: asyncpg.Pool) -> None:
    async with connect(make_test_server(pool)) as searcher:
        with pytest.raises(SearchError, match="must not be empty"):
            await searcher.search("   ", 2)


async def test_agent_answers_through_the_real_mcp_server(pool: asyncpg.Pool) -> None:
    answer = Draft(
        answer="RAG pairs a retriever with a generator [rag_0].",
        claims=[Claim(text="RAG pairs a retriever with a generator", chunk_ids=["rag_0"])],
    )
    reasoner = ScriptedReasoner(Plan(), [answer], [Review(approved=True)])

    async with connect(make_test_server(pool)) as searcher:
        result = await answer_question(build_graph(reasoner, searcher, k=3), QUERY)

    assert reasoner.evidence_seen == [{"rag_0", "eval_0", "rag_1"}]
    assert result.approved
    assert [(c.chunk_id, c.document_title) for c in result.citations] == [
        ("rag_0", "Retrieval-Augmented Generation")
    ]
