import re
import statistics
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel

from docqa_agent.evals.dataset import EvalCase

Score = Literal["correct", "partially_correct", "incorrect"]
SCORE_VALUES: dict[Score, float] = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_gold(case: EvalCase, texts: Iterable[str]) -> bool | None:
    """Whether the texts contain the case's gold quotes. None for unanswerable cases."""
    if not case.answerable:
        return None
    haystack = [normalize(t) for t in texts]
    found = [any(normalize(quote) in text for text in haystack) for quote in case.gold_quotes]
    return all(found) if case.match == "all" else any(found)


_REVIEW_TALK = re.compile(
    r"\b(reviewer|flagged statement|as requested|earlier draft|previous draft)\b", re.IGNORECASE
)


def mentions_review(answer: str) -> bool:
    """Whether a final answer talks about the review process instead of the question."""
    return bool(_REVIEW_TALK.search(answer))


class CaseResult(BaseModel):
    case_id: str
    system: str
    answerable: bool
    answer: str = ""
    retrieval_hit: bool | None = None
    citation_hit: bool | None = None
    citations: int = 0
    invalid_citations: int = 0
    review_leak: bool = False
    score: Score | None = None
    judge_reasoning: str = ""
    approved: bool | None = None
    revisions: int = 0
    searches: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None


class SystemSummary(BaseModel):
    system: str
    cases: int
    errors: int
    retrieval_hit_rate: float | None
    citation_hit_rate: float | None
    correctness: float | None
    fully_correct_rate: float | None
    abstention_accuracy: float | None
    fabricated_citation_rate: float | None
    review_leak_rate: float | None
    approval_rate: float | None
    mean_revisions: float | None
    median_latency_s: float | None
    mean_llm_calls: float | None
    mean_input_tokens: float | None
    mean_output_tokens: float | None


def _rate(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(system: str, rows: Sequence[CaseResult]) -> SystemSummary:
    ok = [r for r in rows if r.error is None]
    answerable = [r for r in ok if r.answerable]
    unanswerable = [r for r in ok if not r.answerable]
    scored = [r for r in ok if r.score is not None]
    reviewed = [r for r in ok if r.approved is not None]
    return SystemSummary(
        system=system,
        cases=len(rows),
        errors=len(rows) - len(ok),
        retrieval_hit_rate=_rate([bool(r.retrieval_hit) for r in answerable]),
        citation_hit_rate=_rate([bool(r.citation_hit) for r in answerable]),
        correctness=_mean([SCORE_VALUES[r.score] for r in scored if r.score]),
        fully_correct_rate=_rate([r.score == "correct" for r in scored]),
        abstention_accuracy=_rate([r.score == "correct" for r in unanswerable]),
        fabricated_citation_rate=_rate([r.invalid_citations > 0 for r in ok]),
        review_leak_rate=_rate([r.review_leak for r in ok]),
        approval_rate=_rate([bool(r.approved) for r in reviewed]),
        mean_revisions=_mean([r.revisions for r in reviewed]),
        median_latency_s=statistics.median([r.latency_s for r in ok]) if ok else None,
        mean_llm_calls=_mean([r.llm_calls for r in ok]),
        mean_input_tokens=_mean([r.input_tokens for r in ok]),
        mean_output_tokens=_mean([r.output_tokens for r in ok]),
    )
