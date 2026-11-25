from docqa_agent.agent.graph import build_result
from docqa_agent.agent.reasoner import Reasoner
from docqa_agent.agent.search import Searcher
from docqa_agent.agent.state import AgentResult


async def answer_single_pass(
    reasoner: Reasoner, searcher: Searcher, question: str, *, k: int = 8
) -> AgentResult:
    """Baseline: one search with the question as written, then one write call, no review."""
    hits = await searcher.search(question, k)
    draft = await reasoner.write(question, hits, None)
    evidence = {hit.chunk_id: hit for hit in hits}
    return build_result(question, draft, evidence, review=None, revisions=0, searches=[question])
