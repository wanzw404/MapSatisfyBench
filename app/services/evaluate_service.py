"""Evaluation business logic.

Encapsulates the full evaluation pipeline so that the router layer
remains thin and only deals with HTTP concerns.
"""

import logging
import os
from typing import Any

from fastapi import HTTPException

from app.config import JUDGE_MODEL, settings
from app.core.evaluation import evaluate_case as run_single_case
from app.core.evaluation.explainer import generate_explanation
from app.core.evaluation.schema import EvalResult, GroundTruth

from app.schemas.evaluate_schemas import EvaluateRequest, EvaluateResponse

logger = logging.getLogger(__name__)


# Wrapped provider cache — 同一 (base_url, model, api_key prefix) 复用同一个
# RateLimitedRetryLLMProvider 实例，保证 token bucket 和 in-flight Semaphore 在
# 所有 case / judge / verifier / meta_judge 之间**全局共享**，QPS 限制才真生效。
# 若每次 run_evaluation 都新建 wrapper，每个 case 各持一桶，限流形同虚设。
_WRAPPED_PROVIDER_CACHE: dict[tuple, Any] = {}


def _build_llm_provider(model_override: str = ""):
    """Instantiate the LLM provider, wrapped with rate-limit + retry.

    Reads ``LLM_QPS`` env var (default ``100``) to configure the global QPS
    cap. ``<=0`` disables rate limiting (transparent passthrough). CLI 入口
    ``batch_evaluate_from_simulator`` 用 ``--llm-qps`` 设置该环境变量。
    """
    from ..core.agent.llm.openai_chat import OpenAICompatProvider
    from ..core.agent.llm.rate_limit import RateLimitedRetryLLMProvider

    base_url = settings.BASE_URL
    api_key = os.environ.get("AI_STUDIO_TOKEN", "") or settings.AI_STUDIO_TOKEN
    if api_key in ("", "0", "your-api-key"):
        raise HTTPException(
            status_code=500,
            detail=(
                "No valid API key configured. Set AI_STUDIO_TOKEN"
                "before calling this endpoint."
            ),
        )

    # 评分模型**强制锁定**到 JUDGE_MODEL：被测 agent 切换模型评测时，
    # judge / fact_verifier / meta_judge / explainer 必须用同一个稳定基线，
    # 否则评分尺度跟着 agent 漂移、跨实验不可比。EvaluateRequest.model
    # 仍允许调用方传值，但这里直接忽略并打 WARNING。
    if model_override and model_override != JUDGE_MODEL:
        logger.warning(
            "[evaluate_service] 收到 model_override=%r，已忽略；强制锁定到 JUDGE_MODEL=%r",
            model_override, JUDGE_MODEL,
        )
    model = JUDGE_MODEL

    try:
        qps = float(os.environ.get("LLM_QPS", "100"))
    except (TypeError, ValueError):
        qps = 100.0

    cache_key = (base_url, model, api_key[:8] if api_key else "", qps)
    cached = _WRAPPED_PROVIDER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    inner = OpenAICompatProvider(
        base_url=base_url,
        api_key=api_key,
        model=model,
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY") or None,
        whale_api_key=os.environ.get("WHALE_API_KEY") or None,
        session_id=os.environ.get("LLM_SESSION_ID") or None,
    )
    wrapped = RateLimitedRetryLLMProvider(inner, qps=qps)
    _WRAPPED_PROVIDER_CACHE[cache_key] = wrapped
    return wrapped


def _build_scores(scores: Any) -> dict[str, Any]:
    """Nest all metric values (including runtime metrics) under a 'metrics' key."""
    dump = scores.model_dump()
    details = dump.pop("details", {})
    return {
        "metrics": {k: v for k, v in dump.items()},
        "details": details,
    }


async def run_evaluation(req: EvaluateRequest) -> EvaluateResponse:
    """Run the full JudgeAgent evaluation pipeline for a single case."""
    # Validate ground_truth up-front so we return 400 on malformed input.
    try:
        gt = GroundTruth.model_validate(req.ground_truth)
    except Exception as exc:  # pydantic ValidationError
        raise HTTPException(
            status_code=400, detail=f"ground_truth validation failed: {exc}"
        ) from exc

    llm_provider = _build_llm_provider(model_override=req.model)
    history = [msg.model_dump() for msg in req.conversation_history_messages]

    # Package the request as the canonical "case" dict consumed by the
    # facade; let evaluate_case build a JudgeAgent internally and run
    # the full pipeline.
    case: dict[str, Any] = {
        "case_id": req.case_id,
        "query": req.query,
        "full_intent": req.full_intent,
        "current_time": req.current_time,
        "current_location": req.current_location,
        "conversation_history_messages": history,
        "ground_truth": gt,
        "tools_schema": req.tools_schema,
        "session_stats": req.session_stats.model_dump() if req.session_stats else None,
    }

    # NOTE: `req.use_mock_search` is currently a no-op — the schema declares
    # the toggle but no `MockWebSearchClient` implementation exists yet.
    # Forwarding it here would crash `evaluate_case` (unexpected kwarg). Wire
    # it through once a mock client is implemented in core.evaluation.verifiers.
    try:
        result: EvalResult = await run_single_case(
            case,
            llm_provider=llm_provider,
            language=req.language,
            enable_verification=req.enable_verification,
            enable_meta_judge=req.enable_meta_judge,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        # Non-JSON LLM reply / malformed verdict.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Build a human-readable explanation via the same LLM provider.
    explanation = await generate_explanation(
        result.scores, result.verdict, gt, llm_provider
    )

    return EvaluateResponse(
        case_id=result.case_id,
        results=_build_scores(result.scores),
        reason=explanation,
    )
