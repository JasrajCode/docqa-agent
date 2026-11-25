from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from docqa_agent.agent.llm import StructuredClaude, TruncatedOutputError
from docqa_agent.agent.reasoner import ClaimCheck, ReviewerOutput, to_review


def check(claim: str, supported: bool) -> ClaimCheck:
    return ClaimCheck(claim=claim, reasoning="...", supported=supported)


def test_verdict_is_approved_only_when_every_check_passes() -> None:
    review = to_review(ReviewerOutput(checks=[check("a", True), check("b", True)]))
    assert review.approved
    assert review.unsupported_claims == []


def test_unsupported_checks_become_the_unsupported_claims() -> None:
    output = ReviewerOutput(
        checks=[check("a", True), check("b", False)], follow_up_queries=["find b"]
    )
    review = to_review(output)
    assert not review.approved
    assert review.unsupported_claims == ["b"]
    assert review.follow_up_queries == ["find b"]


def test_follow_up_searches_are_dropped_when_nothing_is_unsupported() -> None:
    review = to_review(ReviewerOutput(checks=[check("a", True)], follow_up_queries=["pointless"]))
    assert review.approved
    assert review.follow_up_queries == []


class _CannedStructured:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    async def ainvoke(self, messages: object) -> dict[str, Any]:
        return self.response


class _CannedLLM:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response

    def with_structured_output(self, *args: object, **kwargs: object) -> _CannedStructured:
        return _CannedStructured(self.response)


async def test_output_cut_off_at_the_token_limit_raises_a_clear_error() -> None:
    claude = StructuredClaude("claude-haiku-5-5", SecretStr("unused"), max_tokens=100)
    raw = AIMessage(
        content='{"checks": [{"claim": "a", "reas', response_metadata={"stop_reason": "max_tokens"}
    )
    claude._llm = _CannedLLM(  # pyright: ignore[reportAttributeAccessIssue]
        {"raw": raw, "parsed": None, "parsing_error": ValueError("missing field")}
    )
    with pytest.raises(TruncatedOutputError, match="ReviewerOutput output was cut off at the 100"):
        await claude.ask(ReviewerOutput, "system", "user")
    assert claude.usage.calls == 1
