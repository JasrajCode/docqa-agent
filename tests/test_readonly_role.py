import secrets
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

ROLE_SQL = Path(__file__).parents[1] / "sql" / "readonly_role.sql"


async def test_readonly_role_can_read_the_corpus_and_nothing_else(pg_url: str) -> None:
    """Runs the exact SQL from sql/readonly_role.sql that the README tells you to run on Neon."""
    parts = urlsplit(pg_url)
    database = parts.path.lstrip("/")
    password = secrets.token_urlsafe(16)

    admin = await asyncpg.connect(pg_url)
    try:
        sql = ROLE_SQL.read_text().replace("neondb", database).replace("change-me", password)
        await admin.execute(sql)
    finally:
        await admin.close()

    netloc = f"docqa_agent_ro:{password}@{parts.hostname}:{parts.port}"
    conn = await asyncpg.connect(urlunsplit(parts._replace(netloc=netloc)))
    try:
        assert await conn.fetchval('SELECT count(*) FROM "Chunk"') > 0
        assert await conn.fetchval('SELECT count(*) FROM "Document"') > 0

        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.fetch('SELECT * FROM "Account"')
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute('DELETE FROM "Chunk"')
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute("CREATE TABLE sneaky (id int)")
    finally:
        await conn.close()
