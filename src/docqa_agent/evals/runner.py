import asyncio
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, SecretStr

from docqa_agent.agent.graph import answer_question, build_graph
from docqa_agent.agent.reasoner import AnthropicReasoner
from docqa_agent.agent.search import Searcher
from docqa_agent.agent.state import AgentResult
from docqa_agent.evals.dataset import EvalCase
from docqa_agent.evals.judge import Judge
from docqa_agent.evals.metrics import (
    CaseResult,
    SystemSummary,
    contains_gold,
    mentions_review,
    summarize,
)
from docqa_agent.evals.systems import answer_single_pass

SYSTEMS = ("agent", "single_pass")


class RunConfig(BaseModel):
    run_id: str
    model: str
    judge_model: str
    search_k: int
    max_revisions: int
    single_pass_k: int


class EvalRun(BaseModel):
    config: RunConfig
    started_at: datetime
    finished_at: datetime
    summaries: list[SystemSummary]
    results: list[CaseResult]


async def _answer(
    system: str, case: EvalCase, reasoner: AnthropicReasoner, searcher: Searcher, config: RunConfig
) -> AgentResult:
    if system == "single_pass":
        return await answer_single_pass(reasoner, searcher, case.question, k=config.single_pass_k)
    graph = build_graph(reasoner, searcher, k=config.search_k, max_revisions=config.max_revisions)
    metadata = {"eval_run": config.run_id, "case_id": case.id, "system": system}
    return await answer_question(graph, case.question, metadata=metadata)


async def run_case(
    system: str,
    case: EvalCase,
    searcher: Searcher,
    judge: Judge,
    config: RunConfig,
    api_key: SecretStr,
) -> CaseResult:
    reasoner = AnthropicReasoner(config.model, api_key)
    row = CaseResult(case_id=case.id, system=system, answerable=case.answerable)
    started = time.perf_counter()
    try:
        result = await _answer(system, case, reasoner, searcher, config)
    except Exception as exc:  # one failed case shouldn't abort the whole run
        row.error = f"{type(exc).__name__}: {exc}"
        return row
    row.latency_s = time.perf_counter() - started

    evidence = {hit.chunk_id: hit.content for hit in result.evidence}
    row.answer = result.answer
    row.retrieval_hit = contains_gold(case, evidence.values())
    row.citation_hit = contains_gold(case, (evidence[c.chunk_id] for c in result.citations))
    row.citations = len(result.citations)
    row.invalid_citations = len(result.invalid_citations)
    row.review_leak = mentions_review(result.answer)
    row.approved = result.approved
    row.revisions = result.revisions
    row.searches = len(result.searches)
    row.llm_calls = reasoner.usage.calls
    row.input_tokens = reasoner.usage.input_tokens
    row.output_tokens = reasoner.usage.output_tokens

    try:
        verdict = await judge.grade(case, result.answer)
        row.score, row.judge_reasoning = verdict.score, verdict.reasoning
    except Exception as exc:
        row.error = f"judge failed: {type(exc).__name__}: {exc}"
    return row


async def run_eval(
    cases: Sequence[EvalCase],
    systems: Sequence[str],
    searcher: Searcher,
    judge: Judge,
    config: RunConfig,
    api_key: SecretStr,
    *,
    concurrency: int = 4,
    on_result: Callable[[CaseResult], None] | None = None,
) -> EvalRun:
    started_at = datetime.now(UTC)
    limit = asyncio.Semaphore(concurrency)

    async def bounded(system: str, case: EvalCase) -> CaseResult:
        async with limit:
            row = await run_case(system, case, searcher, judge, config, api_key)
        if on_result:
            on_result(row)
        return row

    results = await asyncio.gather(*(bounded(system, case) for system in systems for case in cases))
    summaries = [summarize(s, [r for r in results if r.system == s]) for s in systems]
    return EvalRun(
        config=config,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        summaries=summaries,
        results=list(results),
    )


def _fmt(value: float | None, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "pct":
        return f"{value:.0%}"
    if kind == "s":
        return f"{value:.1f}s"
    if kind == "int":
        return f"{value:,.0f}"
    return f"{value:.2f}"


METRICS: list[tuple[str, str, str]] = [
    ("Retrieval hit rate", "retrieval_hit_rate", "pct"),
    ("Citation hit rate", "citation_hit_rate", "pct"),
    ("Correctness (0-1)", "correctness", "num"),
    ("Fully correct", "fully_correct_rate", "pct"),
    ("Abstention accuracy", "abstention_accuracy", "pct"),
    ("Fabricated citations", "fabricated_citation_rate", "pct"),
    ("Review text leaked", "review_leak_rate", "pct"),
    ("Reviewer approval", "approval_rate", "pct"),
    ("Mean revisions", "mean_revisions", "num"),
    ("Median latency", "median_latency_s", "s"),
    ("Mean LLM calls", "mean_llm_calls", "num"),
    ("Mean input tokens", "mean_input_tokens", "int"),
    ("Mean output tokens", "mean_output_tokens", "int"),
]


def format_summary(run: EvalRun) -> str:
    header = "| Metric | " + " | ".join(s.system for s in run.summaries) + " |"
    divider = "|---|" + "---|" * len(run.summaries)
    lines = [header, divider]
    for label, field, kind in METRICS:
        cells = [_fmt(getattr(s, field), kind) for s in run.summaries]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    errors = sum(s.errors for s in run.summaries)
    cases = run.summaries[0].cases if run.summaries else 0
    lines.append("")
    lines.append(f"{cases} cases per system, {errors} errors. Model: {run.config.model}.")
    return "\n".join(lines)
