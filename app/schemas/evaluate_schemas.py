"""Pydantic schemas for the evaluation endpoint."""

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ConversationMessage(BaseModel):
    """Single turn in the conversation history (accepts tool turns too)."""

    model_config = {"extra": "allow"}

    role: Literal["user", "assistant", "tool", "system"]
    content: str = ""


class SessionStats(BaseModel):
    """Token usage stats for the entire conversation session."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0


class EvaluateRequest(BaseModel):
    """Request body for the single-case evaluation endpoint."""

    # Accept and silently drop legacy fields like ``persona`` so old
    # callers don't get 422'd during the persona-removal rollout.
    model_config = ConfigDict(extra="ignore")

    case_id: str = Field(..., examples=["case-001"])

    # ── session-level stats ──────────────────────────────────────────────
    session_stats: SessionStats = Field(
        default_factory=SessionStats,
        description="Token usage for the entire multi-turn conversation.",
    )

    # ── task context ────────────────────────────────────────────────────
    query: str = Field(..., description="Original first-turn user query")
    full_intent: str = Field(
        default="", description="User's complete inner intent (incl. hard constraints)"
    )
    current_time: str = Field(default="", examples=["2026-05-08 14:30"])
    current_location: str = Field(default="", examples=["北京市朝阳区"])

    # ── evidence ────────────────────────────────────────────────────────
    # Canonical inference-result key is `conversation_history_messages`;
    # the legacy `conversation_history` is kept as a validation alias.
    conversation_history_messages: list[ConversationMessage] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "conversation_history_messages", "conversation_history"
        ),
        description=(
            "Full conversation, including any tool messages. "
            "Legacy alias `conversation_history` is also accepted."
        ),
    )
    ground_truth: dict[str, Any] = Field(
        ..., description="Ground-truth rubric (parsed into schema.GroundTruth)"
    )
    tools_schema: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional: tool function-call schema shown to the Agent.",
    )

    # ── runtime options ─────────────────────────────────────────────────
    language: Literal["chinese", "english"] = "chinese"
    enable_verification: bool = Field(
        default=True,
        description="Set to false to skip fact verification (faster, offline).",
    )
    use_mock_search: bool = Field(
        default=False,
        description=(
            "Use MockWebSearchClient (offline, canned fixtures) vs. the real "
            "GoogleWebSearch MCP gateway."
        ),
    )
    enable_meta_judge: bool = Field(
        default=False,
        description=(
            "Run Stage 4.5 Meta-Judge ('Devil's Advocate') audit. Patches "
            "the verdict via a strict allowlist BEFORE metrics are computed, "
            "so corrections directly modify ECR/TS/IFS/IISR numerators "
            "(no post-hoc multiplicative weight)."
        ),
    )
    model: str = Field(
        default="gpt-5.3-chat-0303-global",
        description=(
            "LLM model id used by JudgeAgent for this evaluation. Defaults "
            "to gpt-5.3-chat-0303-global so evaluations are stable across "
            "deployments regardless of Diamond / .env settings.MODEL_NAME. "
            "Pass explicitly to override (e.g. 'qwen-plus', 'qwen-max')."
        ),
    )


class EvaluateResponse(BaseModel):
    """Response: deterministic scores + LLM-summarised explanation."""

    case_id: str
    results: dict[str, Any]
    reason: str = Field(
        default="",
        description=(
            "LLM-summarised Chinese markdown explaining which tools were "
            "多选 / 漏选 / 不匹配, per-metric scoring reasons, and other "
            "key failure points. Falls back to a deterministic render when "
            "the LLM call fails."
        ),
    )

