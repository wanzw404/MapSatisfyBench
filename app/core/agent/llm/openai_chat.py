"""OpenAI-compatible LLM provider."""

import logging
import os
from typing import Dict, List, Optional

from openai import AsyncOpenAI, OpenAI

from .base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OpenAICompatProvider(BaseLLMProvider):
    """OpenAI-compatible chat completions provider.

    Uses OpenAI SDK, supports custom headers for DashScope/Whale keys.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "",
        dashscope_api_key: Optional[str] = None,
        whale_api_key: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        super().__init__(base_url, api_key, model)
        self.dashscope_api_key = dashscope_api_key
        self.whale_api_key = whale_api_key
        self.session_id = session_id

        # Build default headers for custom headers
        default_headers: Dict[str, str] = {}
        if self.dashscope_api_key:
            default_headers["DASHSCOPE-API-KEY"] = self.dashscope_api_key
        if self.whale_api_key:
            default_headers["WHALE-API-KEY"] = self.whale_api_key
        if self.session_id:
            default_headers["x-session-id"] = self.session_id

        # Initialize OpenAI clients (sync + async)
        self._sync_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers or None,
        )

    async def achat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Call OpenAI-compatible API using async client.

        Note: `temperature` is only forwarded when explicitly provided. Some
        models (e.g. gpt-5.3-chat-0303-global) reject any non-default value,
        so omitting the field lets the model use its own default.
        """
        req_kwargs: dict = {
            "model": self.model,
            "messages": messages,
        }
        if temperature is not None:
            req_kwargs["temperature"] = temperature
        if max_tokens is not None:
            req_kwargs["max_tokens"] = max_tokens

        logger.debug(
            "[LLM async] model=%s api_key_prefix=%s headers=%s",
            self.model,
            self.api_key[:8] if self.api_key else "NONE",
            list(self._async_client.default_headers.keys()) if self._async_client.default_headers else [],
        )
        response = await self._async_client.chat.completions.create(**req_kwargs)

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=response.model or self.model,
            response_id=getattr(response, "id", "") or "",
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Call OpenAI-compatible API using sync client.

        Note: `temperature` is only forwarded when explicitly provided. Some
        models (e.g. gpt-5.3-chat-0303-global) reject any non-default value,
        so omitting the field lets the model use its own default.
        """
        req_kwargs: dict = {
            "model": self.model,
            "messages": messages,
        }
        if temperature is not None:
            req_kwargs["temperature"] = temperature
        if max_tokens is not None:
            req_kwargs["max_tokens"] = max_tokens

        logger.debug(
            "[LLM sync] model=%s api_key_prefix=%s headers=%s",
            self.model,
            self.api_key[:8] if self.api_key else "NONE",
            list(self._sync_client.default_headers.keys()) if self._sync_client.default_headers else [],
        )
        response = self._sync_client.chat.completions.create(**req_kwargs)

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            model=response.model or self.model,
            response_id=getattr(response, "id", "") or "",
        )
