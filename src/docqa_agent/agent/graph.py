import asyncio
import re
from collections.abc import Iterable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from docqa_agent.agent.reasoner import Reasoner
from docqa_agent.agent.search import Searcher
from docqa_agent.agent.state import AgentResult, AgentState, Citation, Draft, NextStep, Review

INLINE_CITATION = re.compile(r"\[([^\[\]\s]+)\]")


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in items if item.strip()))


def _claimed_ids(draft: Draft) -> list[str]:
    return _unique(chunk_id for claim in draft.claims for chunk_id in claim.chunk_ids)


def _need[T](value: T | None, key: str) -> T:
    if value is None:
        raise RuntimeError(f"agent state has no {key!r}; a node ran before the one that sets it")
    return value


def build_graph(
    reasoner: Reasoner,
    searcher: Searcher,
    *,
    k: int = 4,
    max_revisions: int = 2,
    max_sub_questions: int = 4,
    max_follow_ups: int = 3,
) -> CompiledStateGraph:
    """Planner -> retriever -> writer -> reviewer, looping back at most max_revisions times."""

    async def planner(state: AgentState) -> dict[str, Any]:
        plan = await reasoner.plan(state["question"])
        queries = _unique(plan.sub_questions)[:max_sub_questions]
        return {"queries": queries or [state["question"]]}

    async def retriever(state: AgentState) -> dict[str, Any]:
        searched = state["searched"]
        queries = [q for q in _unique(state.get("queries", [])) if q not in searched]
        results = await asyncio.gather(*(searcher.search(q, k) for q in queries))
        evidence = {hit.chunk_id: hit for hits in results for hit in hits}
        return {"evidence": evidence, "searched": [*searched, *queries], "queries": []}

    async def writer(state: AgentState) -> dict[str, Any]:
        evidence = list(state["evidence"].values())
        draft = await reasoner.write(state["question"], evidence, state.get("review"))
        return {"draft": draft}

    async def reviewer(state: AgentState) -> dict[str, Any]:
        draft, evidence = _need(state.get("draft"), "draft"), state["evidence"]
        # Citations of passages that were never retrieved fail without an LLM call.
        unknown = [chunk_id for chunk_id in _claimed_ids(draft) if chunk_id not in evidence]
        if unknown:
            review = Review(
                approved=False,
                unsupported_claims=[
                    f"Cites {cid}, which is not a retrieved passage" for cid in unknown
                ],
            )
        else:
            review = await reasoner.review(state["question"], draft, list(evidence.values()))

        revisions = state["revisions"]
        if review.approved or revisions >= max_revisions:
            return {"review": review, "next_step": "finalize"}
        follow_ups = _unique(review.follow_up_queries)[:max_follow_ups]
        return {
            "review": review,
            "revisions": revisions + 1,
            "queries": follow_ups,
            "next_step": "retriever" if follow_ups else "writer",
        }

    def finalize(state: AgentState) -> dict[str, Any]:
        draft = _need(state.get("draft"), "draft")
        review = _need(state.get("review"), "review")
        evidence = state["evidence"]
        cited = [
            chunk_id
            for chunk_id in _unique([*INLINE_CITATION.findall(draft.answer), *_claimed_ids(draft)])
            if chunk_id in evidence
        ]
        result = AgentResult(
            question=state["question"],
            answer=draft.answer,
            citations=[
                Citation(
                    chunk_id=cid,
                    document_title=evidence[cid].document_title,
                    chunk_index=evidence[cid].chunk_index,
                )
                for cid in cited
            ],
            approved=review.approved,
            revisions=state["revisions"],
            searches=state["searched"],
            unsupported_claims=[] if review.approved else review.unsupported_claims,
        )
        return {"result": result}

    def next_step(state: AgentState) -> NextStep:
        return _need(state.get("next_step"), "next_step")

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("retriever", retriever)
    graph.add_node("writer", writer)
    graph.add_node("reviewer", reviewer)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges("reviewer", next_step, ["retriever", "writer", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile()


async def answer_question(
    graph: CompiledStateGraph, question: str, *, metadata: dict[str, Any] | None = None
) -> AgentResult:
    initial: AgentState = {"question": question, "evidence": {}, "searched": [], "revisions": 0}
    final = await graph.ainvoke(
        initial, config={"run_name": "docqa-agent", "metadata": metadata or {}}
    )
    return final["result"]
