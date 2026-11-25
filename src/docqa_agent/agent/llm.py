from dataclasses import dataclass
from typing import Any, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, SecretStr


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def record(self, message: AIMessage) -> None:
        self.calls += 1
        if message.usage_metadata:
            self.input_tokens += message.usage_metadata["input_tokens"]
            self.output_tokens += message.usage_metadata["output_tokens"]


class TruncatedOutputError(RuntimeError):
    """The model hit max_tokens, so its structured output is incomplete."""


DEFAULT_MAX_TOKENS = 8192


class StructuredClaude:
    """Claude calls that return pydantic models, with token usage recorded per instance."""

    def __init__(
        self, model: str, api_key: SecretStr, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.usage = Usage()
        # temperature is left unset: current Claude models reject the parameter.
        self._llm = ChatAnthropic(
            model_name=model,
            api_key=api_key,
            max_tokens_to_sample=max_tokens,
            timeout=None,
            stop=None,
        )

    async def ask[T: BaseModel](self, schema: type[T], system: str, user: str) -> T:
        # json_schema uses Claude's native structured outputs, which constrain generation to
        # the schema. Forced tool calling occasionally nested fields inside a string.
        structured = self._llm.with_structured_output(
            schema, method="json_schema", include_raw=True
        )
        response = cast(
            dict[str, Any],
            await structured.ainvoke([SystemMessage(system), HumanMessage(user)]),
        )
        raw: AIMessage = response["raw"]
        self.usage.record(raw)
        if raw.response_metadata.get("stop_reason") == "max_tokens":
            raise TruncatedOutputError(
                f"{schema.__name__} output was cut off at the {self.max_tokens}-token limit"
            )
        if response["parsing_error"] is not None:
            raise response["parsing_error"]
        parsed = response["parsed"]
        if not isinstance(parsed, schema):
            raise TypeError(f"expected {schema.__name__}, got {type(parsed).__name__}")
        return parsed
