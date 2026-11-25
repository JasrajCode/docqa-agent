from pydantic import BaseModel, Field, SecretStr

from docqa_agent.agent.llm import StructuredClaude
from docqa_agent.evals.dataset import EvalCase
from docqa_agent.evals.metrics import Score

JUDGE_PROMPT = """You grade answers from a question-answering system over a set of documents.
Compare the system's answer with the reference answer. Judge factual content only, not \
style, length or citation format.
- correct: contains the key facts of the reference and nothing that contradicts it.
- partially_correct: has some of the key facts, or adds wrong details that are not central.
- incorrect: misses or contradicts the key facts.
When the reference says the documents do not contain the answer, the answer is correct if \
it clearly says the information is not available; related context alongside that statement \
is fine. It is incorrect if it presents an answer to the question as if the documents \
contained one."""


class Verdict(BaseModel):
    # Field order matters: reasoning is written before the score.
    reasoning: str = Field(
        description="One or two sentences comparing the answer to the reference."
    )
    score: Score


class Judge:
    def __init__(self, model: str, api_key: SecretStr) -> None:
        self._claude = StructuredClaude(model, api_key, max_tokens=512)

    async def grade(self, case: EvalCase, answer: str) -> Verdict:
        user = (
            f"Question: {case.question}\n\nReference answer: {case.reference_answer}\n\n"
            f"System answer:\n{answer}"
        )
        return await self._claude.ask(Verdict, JUDGE_PROMPT, user)
