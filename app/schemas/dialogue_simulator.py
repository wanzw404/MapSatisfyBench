from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class DialogueCase(BaseModel):
    """单条多轮对话评测用例（从 Excel 读取）"""
    task_id: str | None = Field(default=None, description="评测用例唯一标识，作为 conversation_id")
    query: str = Field(..., description="用户首轮提问")
    context: str | None = Field(default=None, description="上下文信息，直接作为 SystemMessage 传入")
    time: str | None = Field(default=None, description="当前时间上下文")
    location: str | None = Field(default=None, description="当前位置上下文")
    tool: str | None = Field(default=None, description="逗号分隔的可调用工具白名单")
    persona: str | None = Field(default=None, description="用户画像")
    full_intent: str | None = Field(default=None, description="用户完整内心意图")
    expected: str | None = Field(default=None, description="期望结果")
    ground_truth: str | None = Field(default=None, description="评测 ground_truth JSON")
    user_simulator_input: str | None = Field(default=None, description="UserSimulator 实际入口入参")


class DialogueTurn(BaseModel):
    """单轮对话记录"""
    conversation_id: str = Field(..., description="唯一会话 ID")
    turn_index: int = Field(..., description="轮次序号")
    role: Literal["user", "assistant"] = Field(..., description="发言角色")
    content: str = Field(default="", description="消息内容")
    query: str = Field(default="", description="原始用户query")
    context: str | None = Field(default=None, description="原始用户context")
    time: str | None = Field(default=None, description="用户会话发起时间")
    location: str | None = Field(default=None, description="用户所在位置")

    tool_calls: list[dict] = Field(default_factory=list, description="工具调用列表（assistant 轮次）")

    is_stop: bool = Field(default=False, description="是否包含终止标记")
    is_forced_stop: bool = Field(default=False, description="是否因达到最大轮次被强制终止")
    status: Literal["success", "error", "timeout", "recursion_limit"] = Field(
        default="success",
        description=(
            "success / error / timeout / recursion_limit。"
            "recursion_limit 单列：LangGraph React loop 触发 graph.ainvoke "
            "的 recursion_limit（默认 50），通常意味着 LLM 工具调用纠错失败"
            "进入死循环。content 仍为 best-effort 还原的最后一条 AI 文本。"
        ),
    )
    execution_time_ms: int = Field(default=0)
    error_message: str | None = Field(default=None)
    user_simulator_input: str | None = Field(default=None, description="UserSimulator 实际入口入参")
    llm_metrics: str | None = Field(default=None, description="Agent LLM 调用指标 JSON")
    input_tokens: int = Field(default=0, description="本轮 LLM 调用的 prompt tokens（agent 多 call 时是跨 call 求和）")
    output_tokens: int = Field(default=0, description="本轮 LLM 调用的 completion tokens（agent 多 call 时是跨 call 求和）")
    reasoning_tokens: int = Field(
        default=0,
        description=(
            "本轮 LLM 调用的 reasoning/thinking tokens（agent 多 call 时跨 call 求和）。"
            "仅 Vertex 协议（gemini 模型）能拿到 thoughtsTokenCount 单列；"
            "OpenAI 兼容协议恒为 0。不并入 output_tokens 便于横向对比。"
        ),
    )
    logid: str = Field(default="", description="LLM 响应的 id（agent 多 call 时逗号拼接全部）")
    tool: str | None = Field(default=None, description="case 维度的工具白名单（逗号分隔），透传自 DialogueCase.tool")
    ground_truth: str | None = Field(default=None, description="评测 ground_truth JSON，透传自 DialogueCase.ground_truth")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="该 turn 生成时刻（本地时间）",
    )
    empty_response_dump: str | None = Field(
        default=None,
        description=(
            "诊断字段：仅当显式 ReAct agent 的一轮 LLM 调用最终抛出 "
            "'(empty LLM response)' 时填入。值为 retry helper 出口处对原始 "
            "LLM response 的 repr() 快照（在 reasoning_content 合并之前），"
            "保留 content / additional_kwargs / response_metadata / "
            "tool_calls / usage_metadata 等全部字段供故障定位。其余场景该列空。"
        ),
    )


class DialogueResult(BaseModel):
    """一次完整对话评测结果"""
    conversation_id: str = Field(...)
    case: DialogueCase
    turns: list[DialogueTurn] = Field(default_factory=list)
    total_turns: int = Field(default=0)
    is_natural_stop: bool = Field(default=False, description="是否自然终止（非强制截断）")
