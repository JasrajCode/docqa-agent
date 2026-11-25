"""Runs the full agent with real Claude calls against the real corpus.

Needs ANTHROPIC_API_KEY plus the server settings in .env; skipped otherwise.
"""

import pytest
from dotenv import load_dotenv
from pydantic import ValidationError

from docqa_agent.agent.cli import run
from docqa_agent.config import AgentSettings, Settings

pytestmark = pytest.mark.integration


async def test_agent_answers_a_real_question_with_valid_citations() -> None:
    load_dotenv()
    try:
        AgentSettings()  # pyright: ignore[reportCallIssue]
        Settings()  # pyright: ignore[reportCallIssue]
    except ValidationError:
        pytest.skip("needs ANTHROPIC_API_KEY and server settings in .env")

    result = await run("Which datasets were used to evaluate RAG?")

    assert result.answer.strip()
    assert result.citations, "expected at least one cited passage"
    assert result.searches
    print(f"\napproved={result.approved} revisions={result.revisions} searches={result.searches}")
    print(result.answer)
