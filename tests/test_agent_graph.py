from collections.abc import Sequence

from docqa_agent.agent.graph import answer_question, build_graph
from docqa_agent.agent.state import Claim, Draft, Plan, Review
from docqa_agent.models import ChunkHit

QUESTION = "What does RAG retrieve, and how is it evaluated?"


def hit(chunk_id: str, title: str = "RAG paper", index: int = 0) -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id,
        document_id="doc",
        document_title=title,
        chunk_index=index,
        similarity=0.9,
        content=f"text of {chunk_id}",
    )


def draft(*chunk_ids: str) -> Draft:
    citations = " ".join(f"[{cid}]" for cid in chunk_ids)
    return Draft(
        answer=f"An answer {citations}.",
        claims=[Claim(text=f"claim from {cid}", chunk_ids=[cid]) for cid in chunk_ids],
    )


class ScriptedReasoner:
    def __init__(self, plan: Plan, drafts: list[Draft], reviews: list[Review]) -> None:
        self._plan = plan
        self._drafts = list(drafts)
        self._reviews = list(reviews)
        self.feedback_seen: list[Review | None] = []
        self.evidence_seen: list[set[str]] = []
        self.review_calls = 0

    async def plan(self, question: str) -> Plan:
        return self._plan

    async def write(
        self, question: str, evidence: Sequence[ChunkHit], feedback: Review | None
    ) -> Draft:
        self.feedback_seen.append(feedback)
        self.evidence_seen.append({h.chunk_id for h in evidence})
        return self._drafts.pop(0)

    async def review(self, question: str, draft: Draft, evidence: Sequence[ChunkHit]) -> Review:
        self.review_calls += 1
        return self._reviews.pop(0)


class FakeSearcher:
    def __init__(self, results: dict[str, list[ChunkHit]]) -> None:
        self._results = results
        self.queries: list[str] = []

    async def search(self, query: str, k: int) -> list[ChunkHit]:
        self.queries.append(query)
        return self._results.get(query, [])[:k]


async def test_simple_question_is_searched_as_written_and_approved() -> None:
    reasoner = ScriptedReasoner(Plan(), [draft("c1")], [Review(approved=True)])
    searcher = FakeSearcher({QUESTION: [hit("c1", index=3)]})

    result = await answer_question(build_graph(reasoner, searcher), QUESTION)

    assert searcher.queries == [QUESTION]
    assert result.approved and result.revisions == 0
    assert [(c.chunk_id, c.document_title, c.chunk_index) for c in result.citations] == [
        ("c1", "RAG paper", 3)
    ]


async def test_planner_sub_questions_are_each_searched_and_evidence_is_merged() -> None:
    plan = Plan(sub_questions=["what RAG retrieves", "how RAG is evaluated", "what RAG retrieves"])
    reasoner = ScriptedReasoner(plan, [draft("c1", "c2")], [Review(approved=True)])
    searcher = FakeSearcher(
        {"what RAG retrieves": [hit("c1")], "how RAG is evaluated": [hit("c2"), hit("c1")]}
    )

    result = await answer_question(build_graph(reasoner, searcher), QUESTION)

    assert searcher.queries == ["what RAG retrieves", "how RAG is evaluated"]
    assert reasoner.evidence_seen == [{"c1", "c2"}]
    assert [c.chunk_id for c in result.citations] == ["c1", "c2"]


async def test_rejected_draft_triggers_follow_up_search_and_rewrite() -> None:
    rejection = Review(
        approved=False,
        unsupported_claims=["RAG was evaluated on FEVER"],
        follow_up_queries=["RAG FEVER evaluation"],
    )
    reasoner = ScriptedReasoner(
        Plan(), [draft("c1"), draft("c1", "c2")], [rejection, Review(approved=True)]
    )
    searcher = FakeSearcher({QUESTION: [hit("c1")], "RAG FEVER evaluation": [hit("c2")]})

    result = await answer_question(build_graph(reasoner, searcher), QUESTION)

    assert searcher.queries == [QUESTION, "RAG FEVER evaluation"]
    assert reasoner.feedback_seen == [None, rejection]
    assert reasoner.evidence_seen[1] == {"c1", "c2"}
    assert result.approved and result.revisions == 1


async def test_rejection_without_follow_ups_rewrites_without_searching_again() -> None:
    rejection = Review(approved=False, unsupported_claims=["an overreach"])
    reasoner = ScriptedReasoner(
        Plan(), [draft("c1"), draft("c1")], [rejection, Review(approved=True)]
    )
    searcher = FakeSearcher({QUESTION: [hit("c1")]})

    result = await answer_question(build_graph(reasoner, searcher), QUESTION)

    assert searcher.queries == [QUESTION]
    assert result.approved and result.revisions == 1


async def test_review_loop_stops_after_max_revisions() -> None:
    rejection = Review(approved=False, unsupported_claims=["still unsupported"])
    reasoner = ScriptedReasoner(Plan(), [draft("c1")] * 3, [rejection] * 3)
    searcher = FakeSearcher({QUESTION: [hit("c1")]})

    result = await answer_question(build_graph(reasoner, searcher, max_revisions=2), QUESTION)

    assert reasoner.review_calls == 3
    assert not result.approved and result.revisions == 2
    assert result.unsupported_claims == ["still unsupported"]


async def test_citing_an_unretrieved_chunk_forces_a_rewrite_without_an_llm_review() -> None:
    reasoner = ScriptedReasoner(
        Plan(), [draft("c1", "made-up"), draft("c1")], [Review(approved=True)]
    )
    searcher = FakeSearcher({QUESTION: [hit("c1")]})

    result = await answer_question(build_graph(reasoner, searcher), QUESTION)

    assert reasoner.review_calls == 1
    feedback = reasoner.feedback_seen[1]
    assert feedback is not None and "made-up" in feedback.unsupported_claims[0]
    assert [c.chunk_id for c in result.citations] == ["c1"]


async def test_inline_text_in_brackets_that_is_not_a_chunk_id_is_not_cited() -> None:
    answer = Draft(answer="See the table [1] and [c1].", claims=[Claim(text="x", chunk_ids=["c1"])])
    reasoner = ScriptedReasoner(Plan(), [answer], [Review(approved=True)])
    searcher = FakeSearcher({QUESTION: [hit("c1")]})

    result = await answer_question(build_graph(reasoner, searcher), QUESTION)

    assert [c.chunk_id for c in result.citations] == ["c1"]
