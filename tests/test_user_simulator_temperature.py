"""UserSimulator must call its LLM with temperature=1.0 strictly.

理由：用户仿真 LLM 是对照实验的稳定 baseline；不同被测 agent 模型切换时
user simulator 的随机性应保持不变。绕过模型 server 默认值（不同
后端默认温度不一致）。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.agent.llm.base import LLMResponse
from app.core.simulator.user_simulator import UserSimulator


def _make_simulator_with_mock_provider() -> tuple[UserSimulator, AsyncMock]:
    mock_provider = AsyncMock()
    mock_provider.achat = AsyncMock(
        return_value=LLMResponse(
            content="ok",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            model="x",
            response_id="rid",
        )
    )
    sim = UserSimulator(
        llm_provider=mock_provider,
        current_time="2026-05-31 09:00",
        current_location="北京",
        persona="",
        query="附近吃啥",
        full_intent="找餐厅",
        language="chinese",
    )
    return sim, mock_provider


@pytest.mark.asyncio
async def test_user_simulator_passes_temperature_one():
    """每次 generate_next_message 都应以 temperature=1.0 调 achat。"""
    sim, mock_provider = _make_simulator_with_mock_provider()
    history = [
        {"role": "user", "content": "附近吃啥"},
        {"role": "assistant", "content": "推荐三家餐厅..."},
    ]
    await sim.generate_next_message(history)

    assert mock_provider.achat.call_count == 1
    _, kwargs = mock_provider.achat.call_args
    assert kwargs.get("temperature") == 1.0, (
        f"UserSimulator 必须以 temperature=1.0 调 achat，实际 kwargs={kwargs}"
    )


@pytest.mark.asyncio
async def test_user_simulator_temperature_stays_one_across_calls():
    """多次调用都必须保持 temperature=1.0（不漂移、不随历史长度改）。"""
    sim, mock_provider = _make_simulator_with_mock_provider()
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    for _ in range(3):
        await sim.generate_next_message(history)

    assert mock_provider.achat.call_count == 3
    for call in mock_provider.achat.call_args_list:
        _, kwargs = call
        assert kwargs.get("temperature") == 1.0
