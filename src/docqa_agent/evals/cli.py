import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from docqa_agent.agent.search import connect, local_server_params
from docqa_agent.config import AgentSettings, Settings
from docqa_agent.db import create_pool
from docqa_agent.evals.dataset import EvalCase, load_dataset
from docqa_agent.evals.judge import Judge
from docqa_agent.evals.metrics import CaseResult, normalize
from docqa_agent.evals.runner import SYSTEMS, EvalRun, RunConfig, format_summary, run_eval

_ALL_CHUNKS_SQL = """
SELECT c.content FROM "Chunk" c JOIN "Document" d ON d.id = c."documentId"
WHERE d."userId" = ANY($1::text[])
"""


async def check_dataset(cases: list[EvalCase]) -> list[str]:
    """Returns the gold quotes that no chunk in scope contains."""
    settings = Settings()  # pyright: ignore[reportCallIssue]
    pool = await create_pool(settings.database_url.get_secret_value())
    try:
        rows = await pool.fetch(_ALL_CHUNKS_SQL, settings.docqa_scope_user_ids)
    finally:
        await pool.close()
    corpus = [normalize(row["content"]) for row in rows]
    return [
        f"{case.id}: {quote!r}"
        for case in cases
        for quote in case.gold_quotes
        if not any(normalize(quote) in chunk for chunk in corpus)
    ]


async def run(cases: list[EvalCase], systems: list[str], concurrency: int, out: Path) -> EvalRun:
    settings = AgentSettings()  # pyright: ignore[reportCallIssue]
    judge_model = settings.eval_judge_model or settings.agent_model
    config = RunConfig(
        run_id=datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
        model=settings.agent_model,
        judge_model=judge_model,
        search_k=settings.agent_search_k,
        max_revisions=settings.agent_max_revisions,
        single_pass_k=settings.eval_single_pass_k,
    )
    judge = Judge(judge_model, settings.anthropic_api_key)

    # Rows are appended as they finish, so an interrupted run keeps its completed cases.
    partial = out / f"{config.run_id}.partial.jsonl"
    total = len(cases) * len(systems)
    done = 0

    def record(row: CaseResult) -> None:
        nonlocal done
        done += 1
        with partial.open("a", encoding="utf-8") as f:
            f.write(row.model_dump_json() + "\n")
        status = row.error or f"{row.score}, {row.latency_s:.1f}s"
        print(f"[{done}/{total}] {row.system:<11} {row.case_id}: {status}", flush=True)

    async with connect(local_server_params()) as searcher:
        result = await run_eval(
            cases,
            systems,
            searcher,
            judge,
            config,
            settings.anthropic_api_key,
            concurrency=concurrency,
            on_result=record,
        )
    partial.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="docqa-eval", description="Evaluate the agent against a single-pass RAG baseline."
    )
    parser.add_argument("--dataset", type=Path, default=Path("evals/dataset.json"))
    parser.add_argument("--out", type=Path, default=Path("evals/results"))
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--cases", nargs="+", help="only run these case ids")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--check-dataset",
        action="store_true",
        help="verify every gold quote appears in the corpus, then exit",
    )
    args = parser.parse_args()

    load_dotenv()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

    cases = load_dataset(args.dataset)
    if args.cases:
        cases = [case for case in cases if case.id in set(args.cases)]

    if args.check_dataset:
        missing = asyncio.run(check_dataset(cases))
        for line in missing:
            print(f"missing: {line}")
        print(f"{len(cases)} cases checked, {len(missing)} gold quotes not found")
        sys.exit(1 if missing else 0)

    args.out.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(run(cases, args.systems, args.concurrency, args.out))
    path = args.out / f"{result.config.run_id}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(format_summary(result))
    print(f"\nFull results: {path}")
