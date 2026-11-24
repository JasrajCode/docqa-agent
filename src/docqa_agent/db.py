from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import asyncpg

# asyncpg forwards URL params it doesn't recognize to Postgres as server settings.
# Neon's copied connection strings include channel_binding, which Postgres rejects.
_UNSUPPORTED_PARAMS = {"channel_binding"}


def to_asyncpg_dsn(url: str) -> str:
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if k not in _UNSUPPORTED_PARAMS]
    return urlunsplit(parts._replace(query=urlencode(query)))


async def create_pool(url: str, max_size: int = 5) -> asyncpg.Pool:
    # Neon's pooled endpoint is PgBouncer in transaction mode, which breaks
    # asyncpg's cached prepared statements, so the cache is turned off.
    return await asyncpg.create_pool(
        to_asyncpg_dsn(url), min_size=1, max_size=max_size, statement_cache_size=0
    )
