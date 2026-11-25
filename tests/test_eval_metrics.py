import pytest

from docqa_agent.evals.dataset import EvalCase
from docqa_agent.evals.metrics import (
    CaseResult,
    contains_gold,
    mentions_review,
    normalize,
    summarize,
)


def test_review_talk_in_an_answer_is_detected() -> None:
    # Phrasing taken from answers that leaked revision text in an earlier eval run.
    assert mentions_review("On the reviewer's flagged statement: the passages do support...")
    assert mentions_review("I have removed the figure as requested.")
    assert not mentions_review("RAG-Token performs better than RAG-Sequence [c1].")


def case(*quotes: str, match: str = "any") -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": "c",
            "question": "q",
            "reference_answer": "a",
            "gold_quotes": list(quotes),
            "match": match,
        }
    )


UNANSWERABLE = EvalCase(id="u", question="q", reference_answer="none", answerable=False)


def test_normalize_ignores_case_and_line_breaks() -> None:
    assert (
        normalize("  The Document\n  encoder (and\nindex) ") == "the document encoder (and index)"
    )


def test_gold_quote_matches_across_reflowed_text() -> None:
    texts = ["keep the document encoder (and\nindex) fixed, only fine-tuning the query encoder"]
    assert contains_gold(case("Document Encoder (and index) fixed"), texts) is True


def test_any_match_needs_one_quote_and_all_match_needs_every_quote() -> None:
    texts = ["alpha passage", "unrelated"]
    assert contains_gold(case("alpha", "beta"), texts) is True
    assert contains_gold(case("alpha", "beta", match="all"), texts) is False
    assert contains_gold(case("alpha", "beta", match="all"), [*texts, "beta passage"]) is True


def test_unanswerable_cases_have_no_retrieval_target() -> None:
    assert contains_gold(UNANSWERABLE, ["anything"]) is None


def row(**fields: object) -> CaseResult:
    defaults: dict[str, object] = {"case_id": "c", "system": "agent", "answerable": True}
    return CaseResult.model_validate(defaults | fields)


def test_summary_rates_exclude_errors_and_split_answerable_cases() -> None:
    summary = summarize(
        "agent",
        [
            row(
                retrieval_hit=True, citation_hit=True, score="correct", approved=True, latency_s=10
            ),
            row(
                retrieval_hit=False,
                citation_hit=False,
                score="partially_correct",
                approved=False,
                revisions=2,
                invalid_citations=1,
                latency_s=20,
            ),
            row(answerable=False, score="correct", approved=True, latency_s=30),
            row(error="boom"),
        ],
    )
    assert summary.cases == 4 and summary.errors == 1
    assert summary.retrieval_hit_rate == 0.5
    assert summary.correctness == pytest.approx((1 + 0.5 + 1) / 3)
    assert summary.abstention_accuracy == 1.0
    assert summary.fabricated_citation_rate == pytest.approx(1 / 3)
    assert summary.approval_rate == pytest.approx(2 / 3)
    assert summary.median_latency_s == 20


def test_summary_of_an_unreviewed_system_has_no_approval_rate() -> None:
    summary = summarize("single_pass", [row(system="single_pass", score="correct")])
    assert summary.approval_rate is None
    assert summary.mean_revisions is None
