import sys
from pathlib import Path

from mcp import StdioServerParameters
from mcp.client import Client


async def test_real_entry_point_serves_over_stdio(pg_url: str, tmp_path: Path) -> None:
    """Launches the server the way Claude Code does: a subprocess speaking MCP over stdio."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "docqa_agent"],
        cwd=tmp_path,  # no .env in here, so config comes only from the env below
        env={
            "DATABASE_URL": pg_url,
            "OPENAI_API_KEY": "sk-unused-list-documents-makes-no-openai-call",
            "DOCQA_SCOPE_USER_IDS": "user_demo,user_eval",
        },
    )
    async with Client(params) as client:
        result = await client.call_tool("list_documents", {})

    assert not result.is_error
    ids = [d["document_id"] for d in result.structured_content["result"]]
    assert ids == ["doc_eval", "doc_rag", "doc_draft"]
