"""Simulator router.

POST /api/v1/simulate/dialogue
  — Runs full multi-turn dialogue simulation (agent + user loop).
    The ``model`` parameter controls the agent model; the user simulator
    model stays locked to ``USER_SIMULATOR_MODEL``.
"""

import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...config import settings
from ...core.simulator.dialogue_simulator import DialogueSimulator
from ...schemas.dialogue_simulator import DialogueCase
from ...services.dialogue_recorder import build_agent_simulator
from ...services.user_simulator_factory import build_user_simulator

import logging
logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1", tags=["simulator"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class DialogueSimulateRequest(BaseModel):
    """Request body for the full multi-turn dialogue simulation."""

    query: str = Field(..., description="User's first-turn query")
    full_intent: str = Field(default="", description="User's complete intent (incl. hard constraints)")
    current_time: str = Field(default="", examples=["2026-06-08 14:30"])
    current_location: str = Field(default="", examples=["北京市朝阳区"])
    persona: str = Field(default="", description="User persona description")
    context: str = Field(default="", description="Context JSON string (passed as agent SystemMessage)")
    tool: str = Field(default="", description="Comma-separated tool whitelist for the agent")
    ground_truth: str = Field(default="", description="Ground truth JSON string for evaluation")
    task_id: str = Field(
        default="",
        description="Optional conversation ID; auto-generated UUID if empty.",
    )

    model: str = Field(
        default="",
        description=(
            "Agent model name (e.g. 'qwen3-plus', 'gpt-4o'). "
            "Falls back to settings.MODEL_NAME when empty. "
            "Only affects the agent; user simulator stays locked to USER_SIMULATOR_MODEL."
        ),
    )
    max_turns: int = Field(default=20, ge=1, le=100, description="Maximum dialogue turns")
    sandbox: bool = Field(default=False, description="Use sandbox/mock tools (user simulator still calls real LLM)")
    streaming: bool = Field(default=False, description="Enable streaming LLM calls for the agent")
    thinking: bool = Field(default=False, description="Enable thinking/reasoning mode for the agent")


class TurnSummary(BaseModel):
    """Condensed single-turn output for the dialogue API response."""

    turn_index: int
    role: Literal["user", "assistant"]
    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    is_stop: bool = False
    is_forced_stop: bool = False
    status: str = "success"
    execution_time_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error_message: str | None = None


class DialogueSimulateResponse(BaseModel):
    """Response body for the multi-turn dialogue simulation."""

    conversation_id: str
    turns: list[TurnSummary]
    total_turns: int
    is_natural_stop: bool = Field(description="True if conversation ended naturally (not forced/error)")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/simulate/dialogue",
    response_model=DialogueSimulateResponse,
    summary="Run full multi-turn dialogue simulation",
    description=(
        "Runs a complete agent ↔ user simulation loop for a single case. "
        "The `model` parameter controls the agent model; the user simulator "
        "model is locked to USER_SIMULATOR_MODEL for experimental consistency. "
        "Returns the full dialogue trace."
    ),
)
async def simulate_dialogue(req: DialogueSimulateRequest) -> DialogueSimulateResponse:
    """Run DialogueSimulator with agent model controlled by req.model."""

    try:
        agent = build_agent_simulator(
            sandbox=req.sandbox,
            streaming=req.streaming,
            model=req.model or None,
            thinking=req.thinking,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    case = DialogueCase(
        task_id=req.task_id or None,
        query=req.query,
        full_intent=req.full_intent or None,
        context=req.context or None,
        time=req.current_time or None,
        location=req.current_location or None,
        persona=req.persona or None,
        tool=req.tool or None,
        ground_truth=req.ground_truth or None,
    )

    try:
        user = build_user_simulator(case)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    conversation_id = req.task_id or str(uuid.uuid4())

    dialogue = DialogueSimulator(
        agent=agent,
        user=user,
        conversation_id=conversation_id,
        writer=None,
        max_turns=req.max_turns,
    )

    try:
        result = await dialogue.simulate(case)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    turns = [
        TurnSummary(
            turn_index=t.turn_index,
            role=t.role,
            content=t.content,
            tool_calls=t.tool_calls,
            is_stop=t.is_stop,
            is_forced_stop=t.is_forced_stop,
            status=t.status,
            execution_time_ms=t.execution_time_ms,
            input_tokens=t.input_tokens,
            output_tokens=t.output_tokens,
            error_message=t.error_message,
        )
        for t in result.turns
    ]

    return DialogueSimulateResponse(
        conversation_id=result.conversation_id,
        turns=turns,
        total_turns=result.total_turns,
        is_natural_stop=result.is_natural_stop,
    )
