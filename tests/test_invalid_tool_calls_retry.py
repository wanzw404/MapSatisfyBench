"""Unit tests for BaseSimulationAgent invalid_tool_calls 自纠机制。

覆盖：
  * _has_invalid_tool_calls 真值表
  * _build_invalid_tool_calls_hint 内容（含 schema、错误描述、修正指令）
  * _retry_if_invalid_tool_calls 端到端：
      - 首次合法 → 不重试
      - 首次 invalid，第 1 次重试合法 → 救回
      - 重试预算用尽（2 次）仍 invalid → 透传最后一条
      - vertex+ToolMessage 末尾 → skip_hint，重试不带 hint
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from app.core.simulator.agent_simulator import BaseSimulationAgent


# ─── 构造一个最小可用的 BaseSimulationAgent 实例（不真起 LLM） ──────────────


@tool(parse_docstring=True)
def fake_search(query: str, cur_adcode: str | None = None) -> str:
    """伪造工具 schema，给 hint 注入测试用。

    Args:
        query: 搜索关键词。
        cur_adcode: 当前城市 adcode。
    """
    return "ok"


def _make_agent(tools=None) -> BaseSimulationAgent:
    """构造仅供静态 / 实例方法测试用的 agent；不会真初始化 LLM 客户端。"""
    obj = BaseSimulationAgent.__new__(BaseSimulationAgent)
    obj.tools = list(tools or [fake_search])
    obj.tools_by_name = {t.name: t for t in obj.tools}
    # 重试方法依赖 self._llm_client 来判断是否 vertex；mock 一个非 vertex 的
    obj._llm_client = SimpleNamespace(bound=SimpleNamespace())
    obj.llm_metrics_log = []
    obj.streaming = False
    return obj


# ─── _has_invalid_tool_calls ───────────────────────────────────────────────


def test_has_invalid_tool_calls_true():
    msg = AIMessage(
        content="",
        invalid_tool_calls=[{"name": "x", "args": "{", "id": "1", "error": "..."}],
    )
    assert BaseSimulationAgent._has_invalid_tool_calls(msg) is True


def test_has_invalid_tool_calls_false_when_empty():
    assert BaseSimulationAgent._has_invalid_tool_calls(AIMessage(content="ok")) is False


def test_has_invalid_tool_calls_false_when_attr_missing():
    obj = SimpleNamespace(content="x")
    assert BaseSimulationAgent._has_invalid_tool_calls(obj) is False


# ─── _build_invalid_tool_calls_hint ───────────────────────────────────────


def test_build_hint_includes_args_error_and_schema():
    agent = _make_agent()
    invalid = [
        {
            "name": "fake_search",
            "args": '{"query": "x", "query": "x"',
            "id": "call_1",
            "error": "Expecting ',' delimiter: line 1 column 96 (char 95)",
        }
    ]
    hint = agent._build_invalid_tool_calls_hint(invalid)
    assert "fake_search" in hint
    assert "Expecting ',' delimiter" in hint
    assert "schema:" in hint
    # tool 的实际 schema 应被注入（fake_search 的 query 字段名出现）
    assert "query" in hint
    # 通用规则提醒在场
    assert "合法 JSON" in hint
    assert "尾逗号" in hint or "// /* */" in hint


def test_build_hint_unknown_tool_falls_back_with_marker():
    agent = _make_agent()
    invalid = [
        {"name": "totally_unknown_tool", "args": "{}", "id": "x", "error": "boom"}
    ]
    hint = agent._build_invalid_tool_calls_hint(invalid)
    assert "totally_unknown_tool" in hint
    # 找不到 schema 时给出明确占位，不应静默
    assert "未在 tools_by_name" in hint or "schema 不可用" in hint


def test_build_hint_truncates_long_args():
    agent = _make_agent()
    long_args = '{"q":"' + "x" * 2000 + '"'
    invalid = [{"name": "fake_search", "args": long_args, "id": "1", "error": "e"}]
    hint = agent._build_invalid_tool_calls_hint(invalid)
    # 500 字符截断 — 不应含 2000 个 x（任意截断哨兵都行，这里直接看长度上界）
    assert hint.count("x") < 1000


# ─── _retry_if_invalid_tool_calls ────────────────────────────────────────


def _bad_msg() -> AIMessage:
    return AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "fake_search",
                "args": '{"query": "x"',
                "id": "call_1",
                "error": "JSONDecodeError",
            }
        ],
    )


def _good_msg() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "fake_search", "args": {"query": "x"}, "id": "call_2", "type": "tool_call"}],
    )


@pytest.mark.asyncio
async def test_retry_no_op_when_no_invalid():
    agent = _make_agent()
    good = _good_msg()
    with patch.object(agent, "_invoke_and_record_metrics") as mock_invoke:
        result = await agent._retry_if_invalid_tool_calls(good, [], config={})
    assert result is good
    mock_invoke.assert_not_called()


@pytest.mark.asyncio
async def test_retry_first_attempt_recovers():
    agent = _make_agent()
    bad = _bad_msg()
    good = _good_msg()
    with patch.object(
        agent, "_invoke_and_record_metrics", side_effect=[good]
    ) as mock_invoke:
        result = await agent._retry_if_invalid_tool_calls(
            bad, [HumanMessage(content="hi")], config={}
        )
    assert result is good
    assert mock_invoke.call_count == 1


@pytest.mark.asyncio
async def test_retry_second_attempt_recovers():
    agent = _make_agent()
    bad1 = _bad_msg()
    bad2 = _bad_msg()
    good = _good_msg()
    with patch.object(
        agent, "_invoke_and_record_metrics", side_effect=[bad2, good]
    ) as mock_invoke:
        result = await agent._retry_if_invalid_tool_calls(
            bad1, [HumanMessage(content="hi")], config={}
        )
    assert result is good
    assert mock_invoke.call_count == 2


@pytest.mark.asyncio
async def test_retry_budget_exhausted_returns_last_bad():
    agent = _make_agent()
    bad1 = _bad_msg()
    bad2 = _bad_msg()
    bad3 = _bad_msg()
    with patch.object(
        agent, "_invoke_and_record_metrics", side_effect=[bad2, bad3]
    ) as mock_invoke:
        result = await agent._retry_if_invalid_tool_calls(
            bad1, [HumanMessage(content="hi")], config={}
        )
    assert agent._has_invalid_tool_calls(result)
    # _MAX_INVALID_TOOL_CALL_RETRIES = 2 → 共 2 次重试调用
    assert mock_invoke.call_count == 2


@pytest.mark.asyncio
async def test_retry_skips_hint_for_vertex_tool_message_end():
    """vertex + 末尾是 ToolMessage → 重试 prompt 不应追加 HumanMessage hint。"""
    agent = _make_agent()
    # 模拟 vertex client（按类名识别）
    agent._llm_client = SimpleNamespace(bound=type("VertexChat", (), {})())
    bad = _bad_msg()
    good = _good_msg()

    captured_messages: list[list] = []

    async def _fake_invoke(retry_messages, config):
        captured_messages.append(list(retry_messages))
        return good

    with patch.object(agent, "_invoke_and_record_metrics", side_effect=_fake_invoke):
        prompt_messages = [
            HumanMessage(content="user query"),
            ToolMessage(content="tool result", name="fake_search", tool_call_id="x"),
        ]
        await agent._retry_if_invalid_tool_calls(bad, prompt_messages, config={})

    # 重试 prompt 必须与原 prompt 等长；末尾仍是 ToolMessage（没追加 hint）
    assert len(captured_messages) == 1
    retry_msgs = captured_messages[0]
    assert len(retry_msgs) == len(prompt_messages)
    assert isinstance(retry_msgs[-1], ToolMessage)


@pytest.mark.asyncio
async def test_retry_injects_hint_when_not_vertex_tool_end():
    """非 vertex 路径或末尾不是 ToolMessage → 重试 prompt 末尾应是 hint HumanMessage。"""
    agent = _make_agent()  # 默认非 vertex
    bad = _bad_msg()
    good = _good_msg()

    captured: list[list] = []

    async def _fake_invoke(retry_messages, config):
        captured.append(list(retry_messages))
        return good

    with patch.object(agent, "_invoke_and_record_metrics", side_effect=_fake_invoke):
        prompt_messages = [HumanMessage(content="user query")]
        await agent._retry_if_invalid_tool_calls(bad, prompt_messages, config={})

    retry_msgs = captured[0]
    assert len(retry_msgs) == len(prompt_messages) + 1
    assert isinstance(retry_msgs[-1], HumanMessage)
    assert "arguments 不是合法 JSON" in retry_msgs[-1].content
    assert "fake_search" in retry_msgs[-1].content
