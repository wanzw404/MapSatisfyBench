"""Unit tests for anthropic_chat factory + agent_simulator branch."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.core.agent.llm.anthropic_chat import (
    _MAX_TOKENS,
    _THINKING_BUDGET_TOKENS,
    make_anthropic_client,
)


# ─── factory: base_url 替换 ───────────────────────────────────────────────


def test_factory_replaces_openai_v1_suffix():
    c = make_anthropic_client(
        model="claude-opus-4-6",
        base_url="https://llm-service.example.com/api/openai/v1",
        api_key="tok",
        timeout=30, temperature=1.0,
        thinking=False, streaming=False,
    )
    assert c.anthropic_api_url == "https://llm-service.example.com/api/anthropic"


def test_factory_replaces_openai_in_middle():
    c = make_anthropic_client(
        model="claude-sonnet-4-6",
        base_url="https://llm-service.example.com/api/openai/foo",
        api_key="tok", timeout=30, temperature=1.0,
        thinking=False, streaming=False,
    )
    assert c.anthropic_api_url == "https://llm-service.example.com/api/anthropic/foo"


def test_factory_passes_url_through_when_no_match():
    c = make_anthropic_client(
        model="claude-x",
        base_url="https://example.com/custom",
        api_key="tok", timeout=30, temperature=1.0,
        thinking=False, streaming=False,
    )
    assert c.anthropic_api_url == "https://example.com/custom"


# ─── factory: 鉴权头 ──────────────────────────────────────────────────────


def test_factory_sets_bearer_auth_header():
    c = make_anthropic_client(
        model="claude-opus-4-6",
        base_url="https://llm-service.example.com/api/openai/v1",
        api_key="my-secret-token",
        timeout=30, temperature=1.0,
        thinking=False, streaming=False,
    )
    assert c.default_headers == {"Authorization": "Bearer my-secret-token"}


def test_factory_sets_api_key_to_empty_string():
    """关键约束：api_key 必须是空串，否则 SDK 同时发 x-api-key + Bearer 触发 401。"""
    c = make_anthropic_client(
        model="claude-opus-4-6",
        base_url="https://llm-service.example.com/api/openai/v1",
        api_key="my-secret-token",
        timeout=30, temperature=1.0,
        thinking=False, streaming=False,
    )
    # SecretStr，看 .get_secret_value() 应是空串
    secret_val = c.anthropic_api_key.get_secret_value() if c.anthropic_api_key else ""
    assert secret_val == ""


# ─── factory: thinking 参数 ───────────────────────────────────────────────


def test_factory_thinking_true_sets_enabled_with_budget():
    c = make_anthropic_client(
        model="claude-opus-4-6",
        base_url="https://x/api/openai/v1", api_key="t",
        timeout=30, temperature=1.0,
        thinking=True, streaming=False,
    )
    assert c.thinking == {"type": "enabled", "budget_tokens": _THINKING_BUDGET_TOKENS}
    assert _THINKING_BUDGET_TOKENS == 32000


def test_factory_thinking_false_param_is_none():
    c = make_anthropic_client(
        model="claude-opus-4-6",
        base_url="https://x/api/openai/v1", api_key="t",
        timeout=30, temperature=1.0,
        thinking=False, streaming=False,
    )
    assert c.thinking is None


# ─── factory: streaming / max_tokens / temperature / timeout ─────────────


def test_factory_streaming_false_disables_streaming():
    c = make_anthropic_client(
        model="claude-opus-4-6",
        base_url="https://x/api/openai/v1", api_key="t",
        timeout=30, temperature=1.0,
        thinking=False, streaming=False,
    )
    assert c.disable_streaming is True


def test_factory_streaming_true_enables_streaming():
    c = make_anthropic_client(
        model="claude-opus-4-6",
        base_url="https://x/api/openai/v1", api_key="t",
        timeout=30, temperature=1.0,
        thinking=False, streaming=True,
    )
    assert c.disable_streaming is False


def test_factory_max_tokens_64000_constant():
    c = make_anthropic_client(
        model="claude-opus-4-6",
        base_url="https://x/api/openai/v1", api_key="t",
        timeout=30, temperature=1.0,
        thinking=False, streaming=False,
    )
    assert c.max_tokens == _MAX_TOKENS == 64000


def test_factory_temperature_passthrough():
    c = make_anthropic_client(
        model="claude-opus-4-6",
        base_url="https://x/api/openai/v1", api_key="t",
        timeout=30, temperature=0.7,
        thinking=False, streaming=False,
    )
    assert c.temperature == 0.7


# ─── agent_simulator __init__ branch dispatch ────────────────────────────


def test_agent_simulator_uses_anthropic_for_claude():
    """model 以 claude 开头时 __init__ 走 anthropic 工厂，不走 ChatOpenAI。"""
    from app.core.simulator import agent_simulator as mod

    captured = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        from unittest.mock import MagicMock
        client = MagicMock()
        client.bind_tools = MagicMock(return_value=client)
        return client

    with patch(
        "app.core.agent.llm.anthropic_chat.make_anthropic_client",
        side_effect=fake_factory,
    ), patch.object(mod, "ChatOpenAI") as mock_openai:
        agent = mod.BaseSimulationAgent(
            base_url="https://llm-service.example.com/api/openai/v1",
            api_key="tok",
            model="claude-opus-4-6",
            tools=[],
            thinking=True,
            streaming=False,
        )

    # ChatOpenAI 不应被调用
    assert not mock_openai.called
    # factory 收到正确入参
    assert captured["model"] == "claude-opus-4-6"
    assert captured["thinking"] is True
    assert captured["streaming"] is False
    assert captured["api_key"] == "tok"


def test_agent_simulator_thinking_true_for_claude_does_not_raise():
    """claude + thinking=True 不应被 _resolve_thinking_kwargs 白名单 raise。"""
    from app.core.simulator import agent_simulator as mod

    def fake_factory(**kwargs):
        from unittest.mock import MagicMock
        client = MagicMock()
        client.bind_tools = MagicMock(return_value=client)
        return client

    with patch(
        "app.core.agent.llm.anthropic_chat.make_anthropic_client",
        side_effect=fake_factory,
    ):
        # 不抛异常即可
        mod.BaseSimulationAgent(
            base_url="https://x/api/openai/v1",
            api_key="tok",
            model="claude-opus-4-6",
            tools=[],
            thinking=True,
        )


def test_streaming_policy_no_longer_forces_claude():
    from app.core.simulator.agent_simulator import BaseSimulationAgent
    # claude 不应在 _STREAMING_POLICY 里（删了之后走 default False，但用户可 opt-in True）
    assert "claude" not in BaseSimulationAgent._STREAMING_POLICY
    # claude-opus / claude-sonnet 走 _DEFAULT_STREAMING_FOR_UNLISTED 路径
    resolved = BaseSimulationAgent._resolve_streaming("claude-opus-4-6", False)
    assert resolved is False
    # 用户传 True 时由于 claude 不在 policy 里，会被默认值覆盖
    # （这是与其它非 listed 模型一致的行为）


def test_thinking_policy_no_claude_entry():
    """claude 已从 _THINKING_POLICY 移除——anthropic 路径完全自管。"""
    from app.core.simulator.agent_simulator import BaseSimulationAgent
    assert "claude" not in BaseSimulationAgent._THINKING_POLICY
