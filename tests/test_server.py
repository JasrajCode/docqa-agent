import asyncpg
from jsonschema import Draft202012Validator
from mcp.client import Client
from mcp_types import CallToolResult, TextContent

from tests.support import QUERY
from tests.support import make_test_server as _server


def _text(result: CallToolResult) -> str:
    return " ".join(b.text for b in result.content if isinstance(b, TextContent))


async def test_exposes_three_read_only_tools(pool: asyncpg.Pool) -> None:
    async with Client(_server(pool)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}

    assert set(tools) == {"search_documents", "get_chunk", "list_documents"}
    for tool in tools.values():
        assert tool.description
        assert tool.annotations is not None and tool.annotations.read_only_hint is True

    k_schema = tools["search_documents"].input_schema["properties"]["k"]
    assert (k_schema["minimum"], k_schema["maximum"]) == (1, 20)


async def test_search_documents_returns_ranked_structured_results(pool: asyncpg.Pool) -> None:
    async with Client(_server(pool)) as client:
        result = await client.call_tool("search_documents", {"query": QUERY, "k": 2})

    assert not result.is_error
    hits = result.structured_content["result"]
    assert [h["chunk_id"] for h in hits] == ["rag_0", "eval_0"]


async def test_get_chunk_hides_other_users_chunks(pool: asyncpg.Pool) -> None:
    async with Client(_server(pool)) as client:
        found = await client.call_tool("get_chunk", {"chunk_id": "rag_0"})
        hidden = await client.call_tool("get_chunk", {"chunk_id": "private_0"})

    assert not found.is_error
    assert found.structured_content["document_id"] == "doc_rag"
    assert hidden.is_error
    assert "Chunk not found" in _text(hidden)


async def test_list_documents_tool(pool: asyncpg.Pool) -> None:
    async with Client(_server(pool)) as client:
        result = await client.call_tool("list_documents", {})

    ids = [d["document_id"] for d in result.structured_content["result"]]
    assert ids == ["doc_eval", "doc_rag", "doc_draft"]


async def test_tool_results_match_their_declared_output_schema(pool: asyncpg.Pool) -> None:
    """Strict clients such as Claude Code validate results, including formats like date-time."""
    # Without rfc3339-validator installed, jsonschema silently skips date-time checks.
    assert "date-time" in Draft202012Validator.FORMAT_CHECKER.checkers

    calls = {
        "search_documents": {"query": QUERY, "k": 3},
        "get_chunk": {"chunk_id": "rag_0"},
        "list_documents": {},
    }
    async with Client(_server(pool)) as client:
        schemas = {t.name: t.output_schema for t in (await client.list_tools()).tools}
        for name, arguments in calls.items():
            schema = schemas[name]
            assert schema is not None, f"{name} has no output schema"
            result = await client.call_tool(name, arguments)
            validator = Draft202012Validator(
                schema, format_checker=Draft202012Validator.FORMAT_CHECKER
            )
            errors = [e.message for e in validator.iter_errors(result.structured_content)]
            assert errors == [], f"{name}: {errors}"


async def test_search_rejects_a_blank_query_or_out_of_range_k(pool: asyncpg.Pool) -> None:
    async with Client(_server(pool)) as client:
        blank = await client.call_tool("search_documents", {"query": "   "})
        too_many = await client.call_tool("search_documents", {"query": QUERY, "k": 50})

    assert blank.is_error and "must not be empty" in _text(blank)
    assert too_many.is_error
