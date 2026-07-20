"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM response wrapper."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    response_id: str = ""   # chat completion id (chatcmpl-xxx)，作 logid 用


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, base_url: str, api_key: str, model: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    @abstractmethod
    async def achat(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """Async chat completion."""
        pass

    def chat(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """Sync chat completion (default runs async in loop)."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            return loop.run_until_complete(self.achat(messages, **kwargs))
        except RuntimeError:
            return asyncio.run(self.achat(messages, **kwargs))
