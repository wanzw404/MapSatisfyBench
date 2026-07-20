"""End-to-end smoke test: 5-round multi-turn chat with claude via
Anthropic protocol, exercising bind_tools / tool_use / tool_result / thinking-on
signature preservation across turns.

打的是真实 Anthropic 路径。需要本机环境变量 ``AI_STUDIO_TOKEN`` 才能跑；
缺 token 自动 skip。CI 环境不应运行——本地手动验证用。

只验证"链路通"：
  * 5 轮对话不报 400 / 401 / 5xx
  * 每轮要么有 text 块、要么有 tool_use（决策被填到 tool_calls）
  * 多轮历史含 tool_use → tool_result 配对（含 thinking signature 回传）
  * thinking=True 路径：至少 1 轮 content 含 thinking 块
"""

from __future__ import annotations

import os
import warnings

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


_TOKEN = os.environ.get("AI_STUDIO_TOKEN")
SKIP_REASON = "需要 AI_STUDIO_TOKEN 环境变量；CI 跳过"

pytestmark = pytest.mark.skipif(not _TOKEN, reason=SKIP_REASON)


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """Get weather for a city.

    Args:
        city: City name in Chinese.
    """
    return f"{city}: sunny, 22°C"


@tool(parse_docstring=True)
def get_navigation(start: str, end: str) -> str:
    """Get navigation between two locations.

    Args:
        start: Start location.
        end: End location.
    """
    return f"{start} → {end}: 20 min, 5 km"


def _make_client(thinking: bool):
    from app.core.agent.llm.anthropic_chat import (
        make_anthropic_client,
    )
    client = make_anthropic_client(
        model="claude-opus-4-6",
        base_url=os.environ.get("BASE_URL", ""),
        api_key=_TOKEN,
        timeout=120.0,
        temperature=1.0,
        thinking=thinking,
        streaming=False,
    ).bind_tools([get_weather, get_navigation])
    return client


def _execute_tool(name: str, args: dict) -> str:
    """模拟 tool node：调用真实工具拿响应。"""
    if name == "get_weather":
        return get_weather.invoke(args)
    if name == "get_navigation":
        return get_navigation.invoke(args)
    return f"unknown tool {name}"


@pytest.mark.timeout(300)
def test_claude_thinking_on_multi_turn_5_rounds():
    """thinking=True 5 轮多轮：每轮可能有 text + thinking + tool_use 混合。"""
    client = _make_client(thinking=True)
    history: list = [
        SystemMessage(content="你是一个友善的助手，可调用工具回答用户问题。"),
        HumanMessage(content="你好，我想了解北京天气然后规划从北京到上海的路线。"),
    ]
    saw_thinking = False
    rounds = 0
    max_rounds = 5
    while rounds < max_rounds:
        rounds += 1
        resp = client.invoke(history)
        # content 可能是 list（含 thinking/text/tool_use）或 str
        if isinstance(resp.content, list):
            for b in resp.content:
                if isinstance(b, dict) and b.get("type") == "thinking":
                    saw_thinking = True
        history.append(resp)
        if not resp.tool_calls:
            # 模型自然停止，跳出
            break
        # 执行工具调用，把 tool_result 加到 history
        for tc in resp.tool_calls:
            result = _execute_tool(tc["name"], tc["args"])
            history.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    assert rounds >= 1
    # thinking=True 路径下至少应该看到一次 thinking 块
    assert saw_thinking, "thinking=True 但全程没看到任何 thinking 块"
    # 最后一条 AIMessage 应有 text content
    last_ai = [m for m in history if isinstance(m, AIMessage)][-1]
    text = ""
    if isinstance(last_ai.content, str):
        text = last_ai.content
    elif isinstance(last_ai.content, list):
        text = "".join(b.get("text", "") for b in last_ai.content if isinstance(b, dict) and b.get("type") == "text")
    assert text.strip(), "最后一轮 AIMessage 没有文本内容"


@pytest.mark.timeout(300)
def test_claude_thinking_off_basic_tool_round():
    """thinking=False 基础 tool 调用：1 轮调工具 + 1 轮总结。"""
    client = _make_client(thinking=False)
    history = [HumanMessage(content="北京天气如何？请用 get_weather 查询。")]
    resp1 = client.invoke(history)
    assert resp1.tool_calls, "thinking=False 模型应能正常发起 tool 调用"
    history.append(resp1)
    for tc in resp1.tool_calls:
        result = _execute_tool(tc["name"], tc["args"])
        history.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    resp2 = client.invoke(history)
    text = resp2.content if isinstance(resp2.content, str) else "".join(
        b.get("text", "") for b in resp2.content if isinstance(b, dict) and b.get("type") == "text"
    )
    assert text.strip()
    # 不应在 thinking=False 时返回 thinking 块
    if isinstance(resp2.content, list):
        for b in resp2.content:
            if isinstance(b, dict):
                assert b.get("type") != "thinking", "thinking=False 但响应里有 thinking 块"
