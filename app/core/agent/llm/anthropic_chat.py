"""Anthropic 协议客户端工厂——claude 系模型走 Anthropic 原生协议
（端点 ``/api/anthropic/v1/messages``）而非 OpenAI-compat。

为什么不走 ChatOpenAI：
  * OpenAI-compat 适配层流式输出 tool_use chunk index 可能错乱
  * Anthropic 原生 Extended Thinking（``thinking={type, budget_tokens}``）
    在 OpenAI-compat 协议下走 ``extra_body``，budget_tokens 大时
    要么被适配层吞、要么 max_tokens 不足报 400
  * Anthropic 原生协议无上述问题，且原生支持 thinking signature 多轮回传

与默认 ChatAnthropic 的差异：
  * 鉴权用 ``Authorization: Bearer <token>``，不是 Anthropic 原生
    的 ``x-api-key``——必须把 ``anthropic_api_key`` 设为空串（否则 SDK 仍
    会附加 x-api-key 头，网关报 401「鉴权 header x-api-key 和
    Authorization 不可同时存在」）
  * ``default_headers`` 注入 Bearer
  * ``anthropic_api_url`` 智能从调用方传的 OpenAI 路径根（``.../api/openai/v1``）
    替换为 anthropic 路径根（``.../api/anthropic``）
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_anthropic import ChatAnthropic

logger = logging.getLogger(__name__)


# Extended Thinking 预算（与原 _THINKING_POLICY['claude'].on 一致）
_THINKING_BUDGET_TOKENS: int = 32000

# 一律给 64000 max_tokens：
#   * thinking=True 时给 budget_tokens(32000) + 余量留 final answer
#   * thinking=False 时也用同值，避免按 thinking 状态分支配置
# claude-opus-4-6 / claude-sonnet-4-6 单次 max_tokens 上限远高于 64000
_MAX_TOKENS: int = 64000


def make_anthropic_client(
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    temperature: float,
    thinking: bool,
    streaming: bool,
) -> ChatAnthropic:
    """构造 ChatAnthropic 客户端，自动将 OpenAI 路径替换为 Anthropic 路径。

    Args:
        model: claude 模型名（如 ``claude-opus-4-6``）
        base_url: OpenAI 兼容的 base_url，内部会将路径 ``/openai/`` 替换为 ``/anthropic/``
        api_key: Bearer token（**通过 default_headers 注入**，不走 SDK 默认）
        timeout: 单次 HTTP 超时（秒）
        temperature: 采样温度；thinking 启用时 Anthropic 强制 1.0（其它值会被忽略）
        thinking: True → 启用 Extended Thinking，budget_tokens=32000
        streaming: 用户 ``--streaming`` 入参；False → ``disable_streaming=True``
    """
    anthropic_base = base_url.rstrip("/")
    if anthropic_base.endswith("/api/openai/v1"):
        anthropic_base = anthropic_base[: -len("/api/openai/v1")] + "/api/anthropic"
    elif "/api/openai" in anthropic_base:
        anthropic_base = anthropic_base.replace("/api/openai", "/api/anthropic", 1)

    thinking_param: Optional[dict] = (
        {"type": "enabled", "budget_tokens": _THINKING_BUDGET_TOKENS}
        if thinking else None
    )

    logger.info(
        "[Agent] claude 模型走 Anthropic 原生协议 (model=%r, base_url=%r, "
        "thinking=%s, streaming=%s, max_tokens=%d)",
        model, anthropic_base, thinking, streaming, _MAX_TOKENS,
    )
    return ChatAnthropic(
        model=model,
        # ⚠️ 必须空串：详见模块顶部 docstring。给非空字符串会导致 SDK 同时
        # 发 x-api-key + 我们注入的 Authorization Bearer，网关报 401。
        anthropic_api_key="",
        anthropic_api_url=anthropic_base,
        default_headers={"Authorization": f"Bearer {api_key}"},
        max_tokens=_MAX_TOKENS,
        timeout=timeout,
        temperature=temperature,
        thinking=thinking_param,
        disable_streaming=not streaming,
    )
