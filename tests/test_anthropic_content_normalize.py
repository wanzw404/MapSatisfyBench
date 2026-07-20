"""Tests for Anthropic-flavored content list normalization + invalid_tool_calls
retry behavior under list-content responses.

ChatAnthropic 在含 thinking / tool_use 的响应中把 ``content`` 返成
``list[dict]`` 而非 str。本测试覆盖：

  * ``_normalize_anthropic_content_to_str``: list / str / None / 杂糅 block
  * ``_is_empty_response``: list 含 text / 仅 thinking / 仅 tool_use 各情况
  * ``_merge_reasoning_into_content_if_empty``: list 形态走 no-op
  * ``_retry_if_invalid_tool_calls``: 当 ChatAnthropic 把异常 tool 调用塞进
    ``invalid_tool_calls`` 时仍能触发重试 + 构造 hint
  * 下游消费点不再把 ``signature`` 漏到字符串通道：
    ``DialogueSimulator._extract_agent_output``
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from app.core.simulator.agent_simulator import BaseSimulationAgent
from app.core.simulator.dialogue_simulator import DialogueSimulator


# ─── _normalize_anthropic_content_to_str ─────────────────────────────────


def test_normalize_str_passthrough():
    assert BaseSimulationAgent._normalize_anthropic_content_to_str("hello") == "hello"


def test_normalize_empty_str():
    assert BaseSimulationAgent._normalize_anthropic_content_to_str("") == ""


def test_normalize_list_text_blocks_only():
    content = [
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"},
    ]
    assert BaseSimulationAgent._normalize_anthropic_content_to_str(content) == "hello world"


def test_normalize_list_mixed_blocks_keeps_only_text():
    """thinking + text + tool_use 只保留 text 拼接。"""
    content = [
        {"type": "thinking", "thinking": "let me think...", "signature": "abc"},
        {"type": "text", "text": "好的，我来查询。"},
        {"type": "tool_use", "id": "x", "name": "f", "input": {"q": "v"}},
    ]
    assert BaseSimulationAgent._normalize_anthropic_content_to_str(content) == "好的，我来查询。"


def test_normalize_list_only_thinking_returns_empty():
    content = [{"type": "thinking", "thinking": "internal", "signature": "s"}]
    assert BaseSimulationAgent._normalize_anthropic_content_to_str(content) == ""


def test_normalize_list_only_tool_use_returns_empty():
    content = [{"type": "tool_use", "id": "x", "name": "f", "input": {}}]
    assert BaseSimulationAgent._normalize_anthropic_content_to_str(content) == ""


def test_normalize_list_with_non_dict_items():
    """边界：list 里混了非 dict 项不应炸。"""
    content = [{"type": "text", "text": "a"}, "stray-string", None]
    assert BaseSimulationAgent._normalize_anthropic_content_to_str(content) == "a"


def test_normalize_none_returns_empty():
    assert BaseSimulationAgent._normalize_anthropic_content_to_str(None) == ""


def test_normalize_other_types_return_empty():
    assert BaseSimulationAgent._normalize_anthropic_content_to_str(42) == ""
    assert BaseSimulationAgent._normalize_anthropic_content_to_str({"x": 1}) == ""


# ─── _is_empty_response: list-content 形态 ───────────────────────────────


def test_is_empty_anthropic_list_with_text_not_empty():
    msg = AIMessage(content=[{"type": "text", "text": "hi"}])
    assert BaseSimulationAgent._is_empty_response(msg) is False


def test_is_empty_anthropic_list_with_only_thinking_and_no_tool_calls():
    """只 thinking 块 + 无 text + 无 tool_calls → 真空。"""
    msg = AIMessage(content=[{"type": "thinking", "thinking": "x", "signature": "s"}])
    assert BaseSimulationAgent._is_empty_response(msg) is True


def test_is_empty_anthropic_list_with_tool_calls_not_empty():
    """list 里有 tool_use → tool_calls 字段被 LangChain 自动填 → 非空。"""
    msg = AIMessage(
        content=[{"type": "tool_use", "id": "tu_1", "name": "f", "input": {"x": 1}}],
        tool_calls=[{"name": "f", "args": {"x": 1}, "id": "tu_1", "type": "tool_call"}],
    )
    assert BaseSimulationAgent._is_empty_response(msg) is False


def test_is_empty_anthropic_invalid_tool_calls_not_empty():
    msg = AIMessage(
        content=[],
        invalid_tool_calls=[{"name": "f", "args": "{bad", "id": "x", "error": "..."}],
    )
    assert BaseSimulationAgent._is_empty_response(msg) is False


# ─── _merge_reasoning_into_content_if_empty: list 走 no-op ────────────────


def test_merge_reasoning_skips_when_content_is_list():
    """ChatAnthropic content=list[blocks] 自身已含完整信息，
    不该把任何东西合并进去（list → str 会破坏结构）。"""
    msg = AIMessage(
        content=[
            {"type": "thinking", "thinking": "internal", "signature": "s"},
            {"type": "text", "text": "answer"},
        ],
        additional_kwargs={"reasoning_content": "should NOT be merged"},
    )
    out = BaseSimulationAgent._merge_reasoning_into_content_if_empty(msg)
    # 仍是 list，未被替换
    assert isinstance(out.content, list)
    assert len(out.content) == 2


def test_merge_reasoning_str_path_unchanged():
    """str 路径行为保持原样。"""
    msg = AIMessage(content="", additional_kwargs={"reasoning_content": "thought"})
    out = BaseSimulationAgent._merge_reasoning_into_content_if_empty(msg)
    assert out.content == "thought"


# ─── _retry_if_invalid_tool_calls: anthropic-flavored payload ────────────


@tool(parse_docstring=True)
def fake_search(q: str) -> str:
    """Fake search.

    Args:
        q: query
    """
    return "ok"


def _make_agent_with_tools():
    """构造一个最小可用的 BaseSimulationAgent（绕过真实 LLM 构造）。"""
    obj = BaseSimulationAgent.__new__(BaseSimulationAgent)
    obj.tools = [fake_search]
    obj.tools_by_name = {fake_search.name: fake_search}
    obj._llm_client = SimpleNamespace(bound=SimpleNamespace())
    obj.llm_metrics_log = []
    obj.streaming = False
    return obj


@pytest.mark.asyncio
async def test_retry_invalid_tool_calls_anthropic_shape():
    """ChatAnthropic 也可能产生 invalid_tool_calls（schema 校验失败 / 边角解析失败）。
    此时 ``_has_invalid_tool_calls`` 应识别、``_retry_if_invalid_tool_calls`` 触发重试。"""
    agent = _make_agent_with_tools()

    # 构造一个 anthropic 风格的"非法 tool 调用"AIMessage：
    # content 为 list（含 text 块表明模型有想法），同时 invalid_tool_calls 非空
    bad_msg = AIMessage(
        content=[{"type": "text", "text": "let me try..."}],
        invalid_tool_calls=[
            {
                "name": "fake_search",
                "args": '{"q": "missing_quote',
                "id": "tu_1",
                "error": "JSON decode error",
            }
        ],
    )
    # 第二次重试返一个正常的 tool_calls 响应
    good_msg = AIMessage(
        content=[],
        tool_calls=[{"name": "fake_search", "args": {"q": "v"}, "id": "tu_2", "type": "tool_call"}],
    )

    assert agent._has_invalid_tool_calls(bad_msg) is True
    assert agent._has_invalid_tool_calls(good_msg) is False

    with patch.object(agent, "_invoke_and_record_metrics", side_effect=[good_msg]) as mk:
        result = await agent._retry_if_invalid_tool_calls(
            bad_msg, [HumanMessage(content="hi")], config={}
        )

    assert result is good_msg
    assert mk.call_count == 1


# ─── _stash_empty_dump_if_needed: list-content 不能被误判成"空响应" ────────


def test_stash_empty_dump_skips_anthropic_list_with_text():
    """回归 csv 第三行 empty_response_dump 出现 AIMessage repr：
    ChatAnthropic content=list[thinking, text] 是含文本的非空响应，
    不该触发空响应 dump（不该把 thinking signature 一起 repr 进 CSV）。
    """
    from langchain_core.messages import AIMessage as AM

    big_signature = "EoINCmU" + "X" * 200
    msg = AM(
        content=[
            {"type": "thinking", "thinking": "...", "signature": big_signature},
            {"type": "text", "text": "这里是最终回答的文本"},
        ],
    )
    out = BaseSimulationAgent._stash_empty_dump_if_needed(msg)
    # 不应在 additional_kwargs 里挂 dump
    assert "empty_response_dump" not in (out.additional_kwargs or {})


def test_stash_empty_dump_still_triggers_for_truly_empty_str_content():
    """对照：纯 str 空内容 + 无 tool_calls 仍按原逻辑挂 dump。"""
    from langchain_core.messages import AIMessage as AM

    msg = AM(content="")
    out = BaseSimulationAgent._stash_empty_dump_if_needed(msg)
    assert "empty_response_dump" in (out.additional_kwargs or {})


def test_stash_empty_dump_still_triggers_for_only_thinking_list():
    """边界：list 里只有 thinking 块、没 text、没 tool_use → 真正空响应，
    应保留 dump 行为用于诊断。"""
    from langchain_core.messages import AIMessage as AM

    msg = AM(
        content=[{"type": "thinking", "thinking": "...", "signature": "s"}],
    )
    out = BaseSimulationAgent._stash_empty_dump_if_needed(msg)
    assert "empty_response_dump" in (out.additional_kwargs or {})


def test_stash_empty_dump_skips_when_tool_calls_present():
    """已有 tool_calls 的 list-content（含 tool_use 块）不该 dump。"""
    from langchain_core.messages import AIMessage as AM

    msg = AM(
        content=[{"type": "tool_use", "id": "tu_1", "name": "f", "input": {}}],
        tool_calls=[{"name": "f", "args": {}, "id": "tu_1", "type": "tool_call"}],
    )
    out = BaseSimulationAgent._stash_empty_dump_if_needed(msg)
    assert "empty_response_dump" not in (out.additional_kwargs or {})


# ─── _compact_history: anthropic list-content 不能让 .strip() 炸 ──────────


def test_compact_history_handles_anthropic_list_content():
    """回归 row 5/第六行 assistant 空 + ``'list' object has no attribute 'strip'``：
    上一轮 ChatAnthropic 返回 ``content=list[blocks]``，下一轮 _compact_history
    在该 AIMessage 上调 ``.strip()`` 直接炸。修复后应：
    1) 不抛异常
    2) 该 AIMessage 被保留（因为含 text 块）
    3) 保留下来的 content 是纯字符串（已剥 thinking/signature/tool_use）
    """
    from langchain_core.messages import HumanMessage as HM, AIMessage as AM

    big_signature = "EqYNCmU" + "X" * 200
    historical_ai = AM(
        content=[
            {"type": "thinking", "thinking": "...", "signature": big_signature},
            {"type": "tool_use", "id": "tu_1", "name": "f", "input": {}},
            {"type": "text", "text": "上一轮的最终回复"},
        ],
        tool_calls=[{"name": "f", "args": {}, "id": "tu_1", "type": "tool_call"}],
    )
    messages = [
        HM(content="第一轮 query"),
        historical_ai,
        # 中间 ToolMessage 略
        HM(content="第二轮 query"),  # 当前轮起点
    ]

    # 不应抛 AttributeError
    compacted = BaseSimulationAgent._compact_history(messages)

    # 历史轮 (HM, AI) + 当前轮 (HM) = 3 条
    assert len(compacted) == 3
    historical = compacted[1]
    # AIMessage 被保留
    assert getattr(historical, "type", "") == "ai"
    # content 已变成纯字符串，不再含 signature / thinking
    assert isinstance(historical.content, str)
    assert "上一轮的最终回复" in historical.content
    assert "signature" not in historical.content
    assert big_signature not in historical.content
    # tool_calls 被清空（对应的 ToolMessage 已丢，避免 orphan）
    assert historical.tool_calls == []


def test_compact_history_skips_anthropic_only_thinking_ai():
    """仅 thinking 块、无 text 的历史 AIMessage 应被跳过（视为无文本回复）。"""
    from langchain_core.messages import HumanMessage as HM, AIMessage as AM

    only_thinking_ai = AM(
        content=[{"type": "thinking", "thinking": "...", "signature": "s"}],
    )
    messages = [
        HM(content="q1"),
        only_thinking_ai,
        HM(content="q2"),  # 当前轮起点
    ]
    compacted = BaseSimulationAgent._compact_history(messages)
    # 只有两条 HumanMessage，AIMessage 被跳过
    types = [getattr(m, "type", "") for m in compacted]
    assert types == ["human", "human"]


def test_compact_history_str_content_unchanged():
    """str content 路径行为保持原样：保留 AIMessage 不改写 content。"""
    from langchain_core.messages import HumanMessage as HM, AIMessage as AM

    historical_ai = AM(content="纯字符串回复")
    messages = [HM(content="q1"), historical_ai, HM(content="q2")]
    compacted = BaseSimulationAgent._compact_history(messages)
    assert len(compacted) == 3
    assert compacted[1].content == "纯字符串回复"


# ─── _extract_agent_output: 不能把 anthropic signature 漏进 content ────────


def test_extract_agent_output_strips_thinking_signature():
    """回归：ChatAnthropic content=list[thinking_block, text_block] 时，
    ``_extract_agent_output`` 必须返回纯文本，不能 ``str(list)`` 把
    base64 signature 字面塞进 DialogueTurn.content（→ CSV / history 污染）。
    """
    big_signature = "EqYNCmUIDhACGAIqQBI" + "X" * 200  # 假装的 base64
    ai_msg = AIMessage(
        content=[
            {
                "type": "thinking",
                "thinking": "let me think about restaurants...",
                "signature": big_signature,
            },
            {"type": "text", "text": "推荐三家餐厅：四季明湖、银丰华美达、爱丁堡。"},
        ],
    )
    state = {"messages": [HumanMessage(content="附近有啥饭店"), ai_msg]}

    content, tool_calls, dump = DialogueSimulator._extract_agent_output(state)

    # 1) 纯字符串 ──────────────────────────────────────
    assert isinstance(content, str)
    # 2) 不应含 signature / thinking 关键词
    assert "signature" not in content
    assert big_signature not in content
    assert "let me think" not in content
    # 3) 应包含 text 块原文
    assert "四季明湖" in content
    # 4) 不应是 Python list repr
    assert not content.startswith("[")


def test_extract_agent_output_str_content_unchanged():
    """str content 路径行为保持原样（不走 normalize 也应输出原字符串）。"""
    ai_msg = AIMessage(content="简单回答")
    state = {"messages": [HumanMessage(content="你好"), ai_msg]}
    content, _, _ = DialogueSimulator._extract_agent_output(state)
    assert content == "简单回答"


def test_extract_agent_output_only_thinking_returns_empty():
    """仅 thinking 块（无 text、无 tool_calls）→ 真空 content。"""
    ai_msg = AIMessage(
        content=[{"type": "thinking", "thinking": "internal", "signature": "s"}],
    )
    state = {"messages": [HumanMessage(content="x"), ai_msg]}
    content, _, _ = DialogueSimulator._extract_agent_output(state)
    assert content == ""


def test_build_hint_for_anthropic_invalid_tool_call_uses_schema():
    """invalid_tool_calls hint 同样能从 tools_by_name 找到 fake_search 的 schema。"""
    agent = _make_agent_with_tools()
    invalid = [{
        "name": "fake_search",
        "args": '{"q":incomplete',
        "id": "tu_x",
        "error": "Expecting value",
    }]
    hint = agent._build_invalid_tool_calls_hint(invalid)
    # hint 应含工具名 + 错误说明 + schema（schema 里至少含字段名 q）
    assert "fake_search" in hint
    assert "Expecting value" in hint
    assert "q" in hint  # schema 中的字段名
    assert "合法 JSON" in hint  # 通用规则提醒
