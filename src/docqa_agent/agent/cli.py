import argparse
import asyncio
import sys

from dotenv import load_dotenv

from docqa_agent.agent.graph import answer_question, build_graph
from docqa_agent.agent.reasoner import AnthropicReasoner
from docqa_agent.agent.search import connect, local_server_params
from docqa_agent.agent.state import AgentResult
from docqa_agent.config import AgentSettings


async def run(question: str) -> AgentResult:
    settings = AgentSettings()  # pyright: ignore[reportCallIssue]  (fields are read from the environment)
    reasoner = AnthropicReasoner(settings.agent_model, settings.anthropic_api_key)
    async with connect(local_server_params()) as searcher:
        graph = build_graph(
            reasoner,
            searcher,
            k=settings.agent_search_k,
            max_revisions=settings.agent_max_revisions,
        )
        return await answer_question(graph, question, metadata={"model": settings.agent_model})


def format_result(result: AgentResult) -> str:
    lines = [result.answer.strip(), "", "Sources:"]
    lines += [
        f"  [{c.chunk_id}] {c.document_title}, chunk {c.chunk_index}" for c in result.citations
    ] or ["  (none)"]
    revisions = f"{result.revisions} revision{'s' if result.revisions != 1 else ''}"
    if result.approved is None:
        lines += ["", "Reviewer: not run"]
    elif result.approved:
        lines += ["", f"Reviewer: approved after {revisions}"]
    else:
        lines += ["", f"Reviewer: not approved after {revisions}. Unsupported claims:"]
        lines += [f"  - {claim}" for claim in result.unsupported_claims]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="docqa-ask", description="Answer a question from the DocQA documents, with citations."
    )
    parser.add_argument("question")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()

    # LangSmith reads its settings from the process environment, not from pydantic settings.
    load_dotenv()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]

    result = asyncio.run(run(args.question))
    print(result.model_dump_json(indent=2) if args.json else format_result(result))
