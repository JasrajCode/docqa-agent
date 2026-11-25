from pathlib import Path

import pytest
from pydantic import ValidationError

from docqa_agent.evals.dataset import EvalCase, load_dataset

DATASET = Path(__file__).parents[1] / "evals" / "dataset.json"


def test_bundled_dataset_is_valid() -> None:
    cases = load_dataset(DATASET)
    assert len(cases) >= 25
    assert any(not c.answerable for c in cases)
    assert any(c.match == "all" for c in cases)


def test_answerable_cases_require_gold_quotes() -> None:
    with pytest.raises(ValidationError, match="need gold_quotes"):
        EvalCase(id="x", question="q", reference_answer="a")


def test_unanswerable_cases_reject_gold_quotes() -> None:
    with pytest.raises(ValidationError, match="must not have gold_quotes"):
        EvalCase(id="x", question="q", reference_answer="a", answerable=False, gold_quotes=["z"])


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    entry = '{"id": "same", "question": "q", "reference_answer": "a", "gold_quotes": ["z"]}'
    path.write_text(f"[{entry}, {entry}]")
    with pytest.raises(ValueError, match="duplicate case ids"):
        load_dataset(path)
