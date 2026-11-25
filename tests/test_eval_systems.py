from docqa_agent.agent.state import Claim, Draft, Plan
from docqa_agent.evals.systems import answer_single_pass
from tests.test_agent_graph import FakeSearcher, ScriptedReasoner, hit

QUESTION = "What is RAG?"


async def test_single_pass_searches_once_writes_once_and_skips_review() -> None:
    draft = Draft(answer="RAG is ... [c1]", claims=[Claim(text="RAG is ...", chunk_ids=["c1"])])
    reasoner = ScriptedReasoner(Plan(), [draft], [])
    searcher = FakeSearcher({QUESTION: [hit("c1"), hit("c2")]})

    result = await answer_single_pass(reasoner, searcher, QUESTION, k=8)

    assert searcher.queries == [QUESTION]
    assert reasoner.review_calls == 0
    assert result.approved is None
    assert [c.chunk_id for c in result.citations] == ["c1"]
    assert {h.chunk_id for h in result.evidence} == {"c1", "c2"}


async def test_single_pass_reports_citations_of_unretrieved_chunks() -> None:
    draft = Draft(answer="x", claims=[Claim(text="x", chunk_ids=["c1", "invented"])])
    reasoner = ScriptedReasoner(Plan(), [draft], [])
    searcher = FakeSearcher({QUESTION: [hit("c1")]})

    result = await answer_single_pass(reasoner, searcher, QUESTION)

    assert result.invalid_citations == ["invented"]
    assert [c.chunk_id for c in result.citations] == ["c1"]
