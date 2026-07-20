"""JudgeAgent evaluation router.

POST /api/v1/evaluate/case
  — Runs JudgeAgent on a single case (ground_truth + conversation_history)
    and returns the deterministic MetricScores along with the raw verdict.

The endpoint delegates to :func:`app.services.evaluate_service.run_evaluation`
so that the router layer remains thin and only deals with HTTP concerns.
"""

from fastapi import APIRouter

from app.schemas.evaluate_schemas import EvaluateRequest, EvaluateResponse
from app.services.evaluate_service import run_evaluation

router = APIRouter(prefix="/api/v1", tags=["evaluate"])


@router.post(
    "/evaluate/case",
    response_model=EvaluateResponse,
    summary="Run JudgeAgent on a single evaluation case",
    description=(
        "Single-case evaluator. Feeds GroundTruth + conversation_history into "
        "the LLM judge (one call), verifies `need_verify` facts via an "
        "external web-search tool (GoogleWebSearch via MCP gateway) plus a "
        "short LLM judgment prompt, and returns deterministic MetricScores "
        "(ECR/TS/IFS/IISR/Eff/SES/CEI) plus the raw verdict for auditing."
    ),
)
async def evaluate_case(req: EvaluateRequest) -> EvaluateResponse:
    return await run_evaluation(req)
