from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, TypeAdapter, model_validator


class EvalCase(BaseModel):
    id: str
    question: str
    reference_answer: str
    answerable: bool = True
    # Short exact passages from the source documents. Matching is case- and
    # whitespace-insensitive, so cases survive re-chunking.
    gold_quotes: list[str] = []
    match: Literal["any", "all"] = "any"

    @model_validator(mode="after")
    def _gold_quotes_match_answerability(self) -> Self:
        if self.answerable and not self.gold_quotes:
            raise ValueError(f"{self.id}: answerable cases need gold_quotes")
        if not self.answerable and self.gold_quotes:
            raise ValueError(f"{self.id}: unanswerable cases must not have gold_quotes")
        return self


def load_dataset(path: Path) -> list[EvalCase]:
    cases = TypeAdapter(list[EvalCase]).validate_json(path.read_text(encoding="utf-8"))
    ids = [case.id for case in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate case ids: {sorted(duplicates)}")
    return cases
