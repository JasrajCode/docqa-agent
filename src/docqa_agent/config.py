from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: SecretStr
    openai_api_key: SecretStr
    # Users whose documents this server may read. Comes from config only, never from tool input.
    docqa_scope_user_ids: Annotated[list[str], NoDecode, Field(min_length=1)]

    # Must match DocQA's lib/embeddings.ts, or query vectors won't line up with stored ones.
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    db_pool_max_size: int = 5

    @field_validator("docqa_scope_user_ids", mode="before")
    @classmethod
    def _split_comma_separated(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value
