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


class StructuredClaude:
    """Claude calls that return pydantic models, with token usage recorded per instance."""

    def __init__(self, model: str, api_key: SecretStr, max_tokens: int = 2048) -> None:
        self.model = model
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
        self.usage.record(response["raw"])
        if response["parsing_error"] is not None:
            raise response["parsing_error"]
        parsed = response["parsed"]
        if not isinstance(parsed, schema):
            raise TypeError(f"expected {schema.__name__}, got {type(parsed).__name__}")
        return parsed
