import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from docqa_agent.agent.state import Claim, Draft, Plan, Review
from docqa_agent.api import RateLimiter, Services, create_app
from docqa_agent.config import ApiSettings
from docqa_agent.models import ChunkHit
from tests.test_agent_graph import FakeSearcher, ScriptedReasoner, hit

QUESTION = "What does RAG retrieve?"
ADMIN_KEY = "admin-test-key"


def approved_reasoner() -> ScriptedReasoner:
    answer = Draft(
        answer="Passages from Wikipedia [c1].",
        claims=[Claim(text="Passages from Wikipedia", chunk_ids=["c1"])],
    )
    return ScriptedReasoner(Plan(), [answer], [Review(approved=True)])


class FailingReasoner(ScriptedReasoner):
    async def write(
        self, question: str, evidence: Sequence[ChunkHit], feedback: Review | None
    ) -> Draft:
        raise RuntimeError("upstream model error with internal details")


def client(per_hour: int = 5, daily_cap: int = 100, reasoner=approved_reasoner) -> TestClient:
    @asynccontextmanager
    async def services() -> AsyncIterator[Services]:
        yield Services(
            searcher=FakeSearcher({QUESTION: [hit("c1", index=2)]}), make_reasoner=reasoner
        )

    settings = ApiSettings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        api_keys=[ADMIN_KEY],
        rate_limit_per_hour=per_hour,
        daily_question_cap=daily_cap,
        trusted_proxy_hops=1,
    )
    return TestClient(create_app(services, settings))


def ask(c: TestClient, *, ip: str = "203.0.113.1", **headers: str):
    return c.post("/ask", json={"question": QUESTION}, headers={"x-forwarded-for": ip, **headers})


def test_health() -> None:
    with client() as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_docs_page_is_dark_themed_and_warns_about_ad_blockers() -> None:
    with client() as c:
        page = c.get("/docs")
        description = c.get("/openapi.json").json()["info"]["description"]
    assert page.status_code == 200
    assert "swagger-ui" in page.text
    assert "background: #1b1b1b" in page.text
    assert "uBlock Origin" in description


def test_root_redirects_to_the_interactive_docs() -> None:
    with client() as c:
        response = c.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/docs"
        assert "/" not in c.get("/openapi.json").json()["paths"]


def test_ask_returns_an_answer_with_resolved_citations_and_no_raw_evidence() -> None:
    with client() as c:
        response = ask(c)
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is True
    assert body["citations"] == [
        {"chunk_id": "c1", "document_title": "RAG paper", "chunk_index": 2}
    ]
    assert "evidence" not in body


def test_questions_are_validated() -> None:
    with client() as c:
        assert c.post("/ask", json={"question": "hi"}).status_code == 422


def test_each_client_is_limited_per_hour() -> None:
    with client(per_hour=2) as c:
        assert [ask(c).status_code for _ in range(3)] == [200, 200, 429]
        assert ask(c, ip="198.51.100.9").status_code == 200


def test_forged_forwarded_for_entries_do_not_create_new_clients() -> None:
    # The proxy appends the real address; a client can only add entries to the left of it.
    with client(per_hour=1) as c:
        assert ask(c, ip="1.1.1.1, 203.0.113.1").status_code == 200
        assert ask(c, ip="2.2.2.2, 203.0.113.1").status_code == 429


def test_the_daily_cap_applies_across_clients() -> None:
    with client(daily_cap=2) as c:
        statuses = [ask(c, ip=f"203.0.113.{i}").status_code for i in range(3)]
    assert statuses == [200, 200, 429]


def test_an_api_key_skips_the_limits() -> None:
    with client(per_hour=1, daily_cap=1) as c:
        assert ask(c).status_code == 200
        assert ask(c).status_code == 429
        assert ask(c, **{"x-api-key": ADMIN_KEY}).status_code == 200
        assert ask(c, **{"x-api-key": "wrong"}).status_code == 429


def test_agent_failures_return_502_without_internal_details() -> None:
    with client(reasoner=lambda: FailingReasoner(Plan(), [], [])) as c:
        response = ask(c)
    assert response.status_code == 502
    assert "internal details" not in response.text


def _events(c: TestClient) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    name = ""
    with c.stream("POST", "/ask/stream", json={"question": QUESTION}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("event:"):
                name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line.removeprefix("data:"))))
    return events


def test_stream_reports_each_step_then_the_result() -> None:
    with client() as c:
        events = _events(c)
    assert [(n, e.get("step")) for n, e in events[:-1]] == [
        ("progress", "planner"),
        ("progress", "retriever"),
        ("progress", "writer"),
        ("progress", "reviewer"),
    ]
    assert events[1][1]["message"] == "Retrieved 1 new passages"
    assert events[3][1]["message"] == "Reviewer approved the draft"
    name, result = events[-1]
    assert name == "result" and result["answer"] == "Passages from Wikipedia [c1]."


def test_stream_ends_with_an_error_event_when_the_agent_fails() -> None:
    with client(reasoner=lambda: FailingReasoner(Plan(), [], [])) as c:
        events = _events(c)
    assert events[-1] == ("error", {"message": "The agent failed to answer."})


def test_rate_limiter_resets_each_hour_and_each_day() -> None:
    now = [1_800_000_000.0]
    limiter = RateLimiter(per_hour=1, daily_cap=2, clock=lambda: now[0])
    assert limiter.admit("a") is None
    assert limiter.admit("a") is not None
    now[0] += 3600
    assert limiter.admit("a") is None
    assert limiter.admit("b") is not None  # daily cap of 2 reached
    now[0] += 86_400
    assert limiter.admit("b") is None
