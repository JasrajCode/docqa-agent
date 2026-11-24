import pytest
from pydantic import ValidationError

from docqa_agent.config import Settings
from docqa_agent.db import to_asyncpg_dsn


def _settings(scope: str) -> Settings:
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        database_url="postgresql://u:p@localhost/db",  # pyright: ignore[reportArgumentType]
        openai_api_key="sk-test",  # pyright: ignore[reportArgumentType]
        docqa_scope_user_ids=scope,  # pyright: ignore[reportArgumentType]
    )


def test_scope_is_read_from_a_comma_separated_value() -> None:
    assert _settings(" user_demo, user_eval ,").docqa_scope_user_ids == ["user_demo", "user_eval"]


def test_an_empty_scope_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _settings("")


def test_secrets_are_masked_in_repr() -> None:
    assert "sk-test" not in repr(_settings("user_demo"))


def test_dsn_drops_channel_binding_but_keeps_sslmode() -> None:
    url = "postgresql://u:p@ep-x-pooler.neon.tech/neondb?sslmode=require&channel_binding=require"
    assert to_asyncpg_dsn(url) == "postgresql://u:p@ep-x-pooler.neon.tech/neondb?sslmode=require"
