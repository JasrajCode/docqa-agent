import logging
import secrets
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from docqa_agent.agent.graph import answer_question, build_graph, stream_answer
from docqa_agent.agent.reasoner import AnthropicReasoner, Reasoner
from docqa_agent.agent.search import Searcher, connect
from docqa_agent.agent.state import AgentResult, Citation, Progress
from docqa_agent.config import AgentSettings, ApiSettings
from docqa_agent.server import create_server

logger = logging.getLogger(__name__)


@dataclass
class Services:
    searcher: Searcher
    make_reasoner: Callable[[], Reasoner]
    search_k: int = 8
    max_revisions: int = 2


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    approved: bool | None
    revisions: int
    searches: list[str]
    unsupported_claims: list[str]

    @classmethod
    def from_result(cls, result: AgentResult) -> "AskResponse":
        return cls.model_validate(result.model_dump(exclude={"evidence", "invalid_citations"}))


@dataclass
class RateLimiter:
    """Fixed-window limits per client per hour, plus one cap on total questions per UTC day.

    State is in memory, which is correct for a single instance.
    """

    per_hour: int
    daily_cap: int
    clock: Callable[[], float] = time.time
    _hourly: dict[str, tuple[int, int]] = field(default_factory=dict)
    _daily: tuple[str, int] = ("", 0)

    def admit(self, client: str) -> str | None:
        """Records the request and returns None, or returns why it was refused."""
        now = self.clock()
        today = datetime.fromtimestamp(now, UTC).date().isoformat()
        day, used_today = self._daily if self._daily[0] == today else (today, 0)
        if used_today >= self.daily_cap:
            return "The demo has reached its daily question limit. Try again tomorrow."

        window = int(now // 3600)
        bucket_window, count = self._hourly.get(client, (window, 0))
        if bucket_window != window:
            count = 0
        if count >= self.per_hour:
            return f"Limit of {self.per_hour} questions per hour reached. Try again later."

        self._hourly[client] = (window, count + 1)
        self._daily = (day, used_today + 1)
        return None


def client_address(request: Request, trusted_hops: int) -> str:
    """The address the nearest trusted proxy saw.

    Proxies append to X-Forwarded-For, so entries left of the trusted ones can be forged.
    """
    forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",")]
    forwarded = [p for p in forwarded if p]
    if trusted_hops > 0 and len(forwarded) >= trusted_hops:
        return forwarded[-trusted_hops]
    return request.client.host if request.client else "unknown"


ServicesFactory = Callable[[], AbstractAsyncContextManager[Services]]


def create_app(services_factory: ServicesFactory, settings: ApiSettings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with services_factory() as services:
            app.state.services = services
            yield

    app = FastAPI(
        title="docqa-agent",
        description=(
            "Answers questions about the DocQA documents with a LangGraph agent that "
            "checks every citation. Rate limited per client and per day."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    limiter = RateLimiter(settings.rate_limit_per_hour, settings.daily_question_cap)

    def enforce_limits(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
        if x_api_key and any(secrets.compare_digest(x_api_key, k) for k in settings.api_keys):
            return
        refusal = limiter.admit(client_address(request, settings.trusted_proxy_hops))
        if refusal:
            raise HTTPException(status_code=429, detail=refusal)

    def graph_for(request: Request):
        services: Services = request.app.state.services
        return build_graph(
            services.make_reasoner(),
            services.searcher,
            k=services.search_k,
            max_revisions=services.max_revisions,
        )

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", dependencies=[Depends(enforce_limits)])
    async def ask(body: AskRequest, request: Request) -> AskResponse:
        try:
            result = await answer_question(graph_for(request), body.question)
        except Exception:
            logger.exception("agent run failed")
            raise HTTPException(status_code=502, detail="The agent failed to answer.") from None
        return AskResponse.from_result(result)

    @app.post(
        "/ask/stream",
        response_class=EventSourceResponse,
        dependencies=[Depends(enforce_limits)],
    )
    async def ask_stream(body: AskRequest, request: Request) -> AsyncIterable[ServerSentEvent]:
        try:
            async for event in stream_answer(graph_for(request), body.question):
                if isinstance(event, Progress):
                    yield ServerSentEvent(event="progress", data=event.model_dump())
                else:
                    yield ServerSentEvent(
                        event="result", data=AskResponse.from_result(event).model_dump()
                    )
        except Exception:
            logger.exception("agent run failed")
            yield ServerSentEvent(event="error", data={"message": "The agent failed to answer."})

    return app


def create_production_app() -> FastAPI:
    """Entry point for uvicorn --factory: the MCP server runs in-process over MCP's client."""
    load_dotenv()
    agent = AgentSettings()  # pyright: ignore[reportCallIssue]

    @asynccontextmanager
    async def services() -> AsyncIterator[Services]:
        async with connect(create_server()) as searcher:
            yield Services(
                searcher=searcher,
                make_reasoner=lambda: AnthropicReasoner(agent.agent_model, agent.anthropic_api_key),
                search_k=agent.agent_search_k,
                max_revisions=agent.agent_max_revisions,
            )

    return create_app(services, ApiSettings())
