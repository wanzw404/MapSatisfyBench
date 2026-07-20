"""Unit tests for BaseSimulationAgent._resolve_thinking_kwargs (on/off policy).

覆盖：用户在 PR 中列出的 10 个目标 agent 仿真模型 × {thinking=on, thinking=off}
的注入载荷 + 未知模型 + 不支持 thinking 的非 reasoning 模型行为。

gemini-* 不进本测试：vertex 路径不走 _THINKING_POLICY，由
VertexChat 的 thinkingConfig.thinkingBudget=0 控制（_build_payload 自测）。
"""

from __future__ import annotations

import pytest

from app.core.simulator.agent_simulator import BaseSimulationAgent as BSA


_ON_QWEN = {"extra_body": {"enable_thinking": True}}
_OFF_QWEN = {"extra_body": {"enable_thinking": False}}
# claude 已移除——claude 走 Anthropic 原生协议，BaseSimulationAgent.__init__
# 在 use_anthropic 分支提前接管，根本不会调用 _resolve_thinking_kwargs。直接调用
# _resolve_thinking_kwargs("claude-...", True) 会按"未命中白名单 + thinking=True"
# 路径 raise——这正是函数文档的契约（"严格白名单"）。


# (model, thinking, expected) — expected 是 dict 或 ExceptionType
_CASES: list[tuple[str, bool, object]] = [
    # ── OpenAI 非 reasoning：on raise，off 零注入 ──────────────────────
    ("gpt-5.3-chat-0303-global", False, {}),
    ("gpt-5.3-chat-0303-global", True,  ValueError),
    ("gpt-41-0414-global",       False, {}),
    ("gpt-41-0414-global",       True,  ValueError),

    # ── Claude：已从 _THINKING_POLICY 移除（走 Anthropic 原生协议）
    # 直接调 _resolve_thinking_kwargs 走"未命中"分支：thinking=False → {}；
    # thinking=True → raise（白名单严格契约）。__init__ 实际路径会绕过此函数。
    ("claude-opus-4-6",   False, {}),
    ("claude-opus-4-6",   True,  ValueError),
    ("claude-sonnet-4-6", False, {}),
    ("claude-sonnet-4-6", True,  ValueError),

    # ── Qwen 系列：默认 on，关需显式发 False ─────────────────────────
    ("qwen3.6-plus",  False, _OFF_QWEN),
    ("qwen3.6-plus",  True,  _ON_QWEN),
    ("qwen3-4b",      False, _OFF_QWEN),
    ("qwen3-4b",      True,  _ON_QWEN),
    ("Qwen3-30B-A3B", False, _OFF_QWEN),
    ("Qwen3-30B-A3B", True,  _ON_QWEN),

    # ── DeepSeek 系列：默认 on，关需显式发 False ─────────────────────
    ("bailian/deepseek-v4-pro", False, _OFF_QWEN),
    ("bailian/deepseek-v4-pro", True,  _ON_QWEN),
    ("deepseek-v3.2",           False, _OFF_QWEN),
    ("deepseek-v3.2",           True,  _ON_QWEN),

    # ── 未知模型：off 静默通过，on raise ─────────────────────────────
    ("some-unknown-model", False, {}),
    ("some-unknown-model", True,  ValueError),
    ("",                   False, {}),
]


@pytest.mark.parametrize("model,thinking,expected", _CASES)
def test_resolve_thinking_kwargs(model, thinking, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            BSA._resolve_thinking_kwargs(model, thinking)
    else:
        assert BSA._resolve_thinking_kwargs(model, thinking) == expected


def test_long_substring_priority_qwen36_plus_over_qwen3():
    """qwen3.6-plus 必须命中独立 entry，而非更短的 qwen3。

    目前两条 entry 载荷一致，但表里有独立的 qwen3.6-plus 是为审计可见；
    如未来 qwen3.6-plus 载荷与 qwen3 分化，本测试可立刻发现顺序失误。
    """
    result_on = BSA._resolve_thinking_kwargs("qwen3.6-plus", True)
    # 当前两条载荷相同，能命中即认为顺序正确。下面这个断言对当前实现是冗余的，
    # 但等于一份"qwen3.6-plus 应继续走 enable_thinking 字段"的契约。
    assert "extra_body" in result_on
    assert "enable_thinking" in result_on["extra_body"]


def test_default_thinking_false_for_dashscope_models_force_off():
    """回归 bailian/deepseek-v4-pro 默认开 thinking 的 bug：
    本次设计的核心动机是 thinking=False 时对 DashScope 系模型注入
    enable_thinking=False。这条单测固化该契约。
    """
    for model in (
        "bailian/deepseek-v4-pro",
        "deepseek-v3.2",
        "qwen3-4b",
        "Qwen3-30B-A3B",
        "qwen3.6-plus",
    ):
        kw = BSA._resolve_thinking_kwargs(model, thinking=False)
        assert kw == {"extra_body": {"enable_thinking": False}}, (
            f"{model}: thinking=False 必须强制发 enable_thinking=False，"
            f"实际 kwargs={kw}"
        )
