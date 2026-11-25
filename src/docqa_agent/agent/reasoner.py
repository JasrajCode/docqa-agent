from collections.abc import Sequence
from typing import Protocol

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, SecretStr

from docqa_agent.agent.state import Draft, Plan, Review
from docqa_agent.models import ChunkHit

PLANNER_PROMPT = """You plan searches over a collection of documents.
Split the user's question into at most 4 standalone search queries, one for each distinct \
part of the question. If the question asks about a single thing, return an empty list and \
it will be searched as written."""

WRITER_PROMPT = """You answer questions using only the passages provided.
Cite every factual statement inline with the chunk_id of a passage that supports it, \
like [chunk_id]. If the passages do not answer the question, say so rather than guessing.
In `claims`, list each factual statement from the answer with the chunk_ids that support it."""

REVIEWER_PROMPT = """You check a draft answer against the passages it cites.
Check every claim, in order. For each one, first name the cited passage that supports it or \
explain why none does, then decide. A claim is supported if its cited passages state it \
directly or it follows plainly from them. A statement that the passages do not mention \
something counts as supported unless a passage contradicts it.
If any claim is unsupported, suggest up to 3 search queries that could find supporting \
passages, or none if the documents are unlikely to contain them."""


class ClaimCheck(BaseModel):
    # Field order matters: the model writes its reasoning before the verdict.
    claim: str = Field(description="The claim being checked, as written in the draft.")
    reasoning: str = Field(
        description="The cited passage that states the claim, or why none does. One sentence."
    )
    supported: bool = Field(description="Whether the cited passages support the claim.")


class ReviewerOutput(BaseModel):
    checks: list[ClaimCheck] = Field(description="One check per claim, in the order given.")
    follow_up_queries: list[str] = Field(
        default_factory=list,
        description="Up to 3 searches likely to find support for unsupported claims.",
    )


def to_review(output: ReviewerOutput) -> Review:
    """Derives the verdict from the per-claim checks so it can't contradict them."""
    unsupported = [check.claim for check in output.checks if not check.supported]
    return Review(
        approved=not unsupported,
        unsupported_claims=unsupported,
        follow_up_queries=output.follow_up_queries if unsupported else [],
    )


class Reasoner(Protocol):
    async def plan(self, question: str) -> Plan: ...

    async def write(
        self, question: str, evidence: Sequence[ChunkHit], feedback: Review | None
    ) -> Draft: ...

    async def review(self, question: str, draft: Draft, evidence: Sequence[ChunkHit]) -> Review: ...


def format_evidence(evidence: Sequence[ChunkHit]) -> str:
    if not evidence:
        return "(no passages found)"
    return "\n\n".join(
        f"[{hit.chunk_id}] ({hit.document_title}, chunk {hit.chunk_index})\n{hit.content}"
        for hit in evidence
    )


class AnthropicReasoner:
    def __init__(self, model: str, api_key: SecretStr, max_tokens: int = 2048) -> None:
        # temperature is left unset: current Claude models reject the parameter.
        self._llm = ChatAnthropic(
            model_name=model,
            api_key=api_key,
            max_tokens_to_sample=max_tokens,
            timeout=None,
            stop=None,
        )

    async def _ask[T: BaseModel](self, schema: type[T], system: str, user: str) -> T:
        result = await self._llm.with_structured_output(schema).ainvoke(
            [SystemMessage(system), HumanMessage(user)]
        )
        if not isinstance(result, schema):
            raise TypeError(f"expected {schema.__name__}, got {type(result).__name__}")
        return result

    async def plan(self, question: str) -> Plan:
        return await self._ask(Plan, PLANNER_PROMPT, question)

    async def write(
        self, question: str, evidence: Sequence[ChunkHit], feedback: Review | None
    ) -> Draft:
        parts = [f"Question: {question}", f"Passages:\n{format_evidence(evidence)}"]
        if feedback and feedback.unsupported_claims:
            flagged = "\n".join(f"- {claim}" for claim in feedback.unsupported_claims)
            parts.append(
                "A reviewer found these statements unsupported by the passages. "
                f"Remove them or support them with a passage:\n{flagged}"
            )
        return await self._ask(Draft, WRITER_PROMPT, "\n\n".join(parts))

    async def review(self, question: str, draft: Draft, evidence: Sequence[ChunkHit]) -> Review:
        claims = "\n".join(
            f"- {claim.text} (cites: {', '.join(claim.chunk_ids) or 'nothing'})"
            for claim in draft.claims
        )
        user = (
            f"Question: {question}\n\nDraft answer:\n{draft.answer}\n\n"
            f"Claims:\n{claims or '(none)'}\n\nPassages:\n{format_evidence(evidence)}"
        )
        return to_review(await self._ask(ReviewerOutput, REVIEWER_PROMPT, user))
