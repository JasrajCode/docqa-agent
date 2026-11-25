from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field

from docqa_agent.models import ChunkHit

# Field descriptions are part of the structured-output schemas sent to the model.


class Plan(BaseModel):
    sub_questions: list[str] = Field(
        default_factory=list,
        description=(
            "Standalone search queries, one per distinct part of the question (at most 4). "
            "Empty when the question can be searched as written."
        ),
    )


class Claim(BaseModel):
    text: str = Field(description="One factual statement made in the answer.")
    chunk_ids: list[str] = Field(description="chunk_ids of the passages supporting the statement.")


class Draft(BaseModel):
    answer: str = Field(description="Markdown answer citing passages inline as [chunk_id].")
    claims: list[Claim] = Field(
        description="Every factual statement in the answer, with the chunk_ids supporting it."
    )


class Review(BaseModel):
    approved: bool = Field(description="True only if every claim is supported by its passages.")
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims not supported by the passages they cite."
    )
    follow_up_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 3 searches likely to find support for the unsupported claims. "
            "Empty if the documents are unlikely to contain it."
        ),
    )


class Citation(BaseModel):
    chunk_id: str
    document_title: str
    chunk_index: int


class AgentResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    approved: bool
    revisions: int
    searches: list[str]
    unsupported_claims: list[str]


def merge_evidence(current: dict[str, ChunkHit], new: dict[str, ChunkHit]) -> dict[str, ChunkHit]:
    return {**current, **new}


NextStep = Literal["retriever", "writer", "finalize"]


class AgentState(TypedDict):
    # Set when a run starts
    question: str
    evidence: Annotated[dict[str, ChunkHit], merge_evidence]
    searched: list[str]
    revisions: int
    # Produced by nodes as the run progresses
    queries: NotRequired[list[str]]
    draft: NotRequired[Draft]
    review: NotRequired[Review]
    next_step: NotRequired[NextStep]
    result: NotRequired[AgentResult]
