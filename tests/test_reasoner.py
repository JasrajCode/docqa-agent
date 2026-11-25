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
