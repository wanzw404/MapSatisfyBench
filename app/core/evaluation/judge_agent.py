"""JudgeAgent — orchestrates the 4-way standalone evaluation pipeline.

Stages::

    1. ``prepare_judge_inputs`` builds a frozen :class:`JudgeInputs`
       snapshot (normalised + annotated history, assistant turns,
       factual rubric, TS dim1∧dim2 candidates) so every judge reads
       from the same pre-processed bundle.
    2. Fan-out 4 LLM calls in parallel via
       ``asyncio.gather(..., return_exceptions=True)``:
         * ECRJudge   → ``explicit_intent_summary``
         * TSJudge    → ``tool_call_summary`` + ``ts_dim3_pass`` (for metrics)
         * IFSJudge   → ``fact_summary`` + ``rubric_row_judgments``
         * IISRJudge  → ``implicit_intent_judgments``
       Each judge returns ``None`` on LLM / JSON / schema failure;
       the corresponding verdict slot then falls back to its schema
       default. The case never fails because of a single judge failure.
       ``judge_status`` records per-judge "ok" / "failed" / "skipped_*".
    3. (Optional) FactVerifier back-fills ``verified_ok`` on any
       ``need_verify`` facts using an external web search (GoogleWebSearch
       via the Amap MCP gateway) plus a short LLM judgment prompt. Must
       run *after* the IFS override above because the verifier writes
       in-place on the rubric_row_judgments instances.
    4. (Optional) MetaJudge audits the verdict and emits a structured
       correction list; VerdictPatcher applies the allowlisted corrections
       in place. Metrics are then computed against the *patched* verdict.
    5. MetricCalculator computes deterministic MetricScores (8 metrics:
       ECR/TS/IFS/IISR/AR/Eff/SES/CEI). TS receives ``ts_dim3_pass``
       so its Jaccard numerator includes the dim3 verdict.
    6. Return EvalResult envelope (case_id + final verdict + scores +
       judge_status).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from app.core.agent.llm.base import BaseLLMProvider
from app.core.evaluation.ecr_judge import ECRJudge
from app.core.evaluation.ifs_judge import IFSJudge
from app.core.evaluation.iisr_judge import IISRJudge
from app.core.evaluation.judge_inputs import JudgeInputs, prepare_judge_inputs
from app.core.evaluation.meta_judge import MetaJudge, VerdictPatcher
from app.core.evaluation.metrics import MetricCalculator
from app.core.evaluation.schema import (
    EvalResult,
    GroundTruth,
    JudgeVerdict,
    MetricScores,
)
from app.core.evaluation.ts_judge import TSJudge
from app.core.evaluation.verifiers import (
    FactVerifier,
    GoogleWebSearchClient,
    WebSearchClient,
)

logger = logging.getLogger(__name__)


class JudgeAgent:
    """4-way standalone evaluator: 4 parallel LLM judges → web verify → Meta-Judge → metrics."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        *,
        search_client: Optional[WebSearchClient] = None,
        metric_calculator: Optional[MetricCalculator] = None,
        meta_judge: Optional[MetaJudge] = None,
        ecr_judge: Optional[ECRJudge] = None,
        ts_judge: Optional[TSJudge] = None,
        ifs_judge: Optional[IFSJudge] = None,
        iisr_judge: Optional[IISRJudge] = None,
        language: str = "chinese",
        enable_verification: bool = True,
        enable_meta_judge: bool = False,
    ) -> None:
        self.llm = llm_provider
        self.client: WebSearchClient = search_client or GoogleWebSearchClient()
        self.verifier = FactVerifier(self.client, llm_provider)
        self.calculator = metric_calculator or MetricCalculator()
        self.enable_verification = enable_verification
        self.enable_meta_judge = enable_meta_judge

        self.ecr_judge: ECRJudge = ecr_judge or ECRJudge(llm_provider, language=language)
        self.ts_judge: TSJudge = ts_judge or TSJudge(llm_provider, language=language)
        self.ifs_judge: IFSJudge = ifs_judge or IFSJudge(llm_provider, language=language)
        self.iisr_judge: IISRJudge = iisr_judge or IISRJudge(llm_provider, language=language)

        self.meta_judge: Optional[MetaJudge] = (
            meta_judge
            if meta_judge is not None
            else (MetaJudge(llm_provider, language=language) if enable_meta_judge else None)
        )
        self.patcher = VerdictPatcher()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    async def _evaluate_with_retry(
        judge_name: str,
        judge_coro,
    ) -> Any:
        """Run a judge evaluate() with one retry on failure.

        Returns the judge result on success, or ``None`` if both attempts fail.
        """
        for attempt in range(2):
            try:
                result = await judge_coro
                if result is not None:
                    return result
                logger.warning(
                    "[%s] evaluate() returned None (attempt %d/2)",
                    judge_name, attempt + 1,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] evaluate() raised %s (attempt %d/2): %s",
                    judge_name, type(exc).__name__, attempt + 1, exc,
                )
            if attempt == 0:
                logger.info("[%s] retrying...", judge_name)
        return None

    async def evaluate(
        self,
        *,
        case_id: str,
        query: str,
        full_intent: str,
        current_time: str,
        current_location: str,
        conversation_history: list[dict[str, Any]],
        ground_truth: GroundTruth | dict[str, Any],
        tools_schema: list[dict[str, Any]] | dict[str, Any] | None = None,
        session_stats: dict[str, Any] | None = None,
    ) -> EvalResult:
        # ---- Runtime metrics from raw history (before normalisation) -------
        e2e_latency_ms = self._compute_avg_e2e_latency(conversation_history)
        _stats = session_stats or {}
        session_input_tokens = int(_stats.get("total_input_tokens", 0))
        session_output_tokens = int(_stats.get("total_output_tokens", 0))
        session_total_tokens = session_input_tokens + session_output_tokens

        # ---- Stage 1: shared input bundle ---------------------------------
        inputs: JudgeInputs = prepare_judge_inputs(
            case_id=case_id,
            query=query,
            full_intent=full_intent,
            current_time=current_time,
            current_location=current_location,
            raw_conversation_history=conversation_history,
            ground_truth=ground_truth,
            tools_schema=tools_schema,
        )
        gt = inputs.ground_truth

        # ---- Stage 2: 4-way parallel LLM fan-out (with retry) -------------
        # Each judge gets one retry on failure. ``return_exceptions=True``
        # so one judge crashing doesn't take down the others; each judge
        # also returns ``None`` on its own parse/LLM failure. Both signals
        # are folded into judge_status.
        results = await asyncio.gather(
            self._evaluate_with_retry("ECRJudge", self.ecr_judge.evaluate(inputs)),
            self._evaluate_with_retry("TSJudge", self.ts_judge.evaluate(inputs)),
            self._evaluate_with_retry("IFSJudge", self.ifs_judge.evaluate(inputs)),
            self._evaluate_with_retry("IISRJudge", self.iisr_judge.evaluate(inputs)),
            return_exceptions=True,
        )
        ecr_r, ts_r, ifs_r, iisr_r = results
        verdict = JudgeVerdict()
        judge_status: dict[str, str] = {}
        ts_dim3_pass: Optional[set[str]] = None

        # ECR
        if isinstance(ecr_r, BaseException) or ecr_r is None:
            judge_status["ecr"] = "failed"
            if isinstance(ecr_r, BaseException):
                logger.warning("ECRJudge raised: %s", ecr_r)
        else:
            verdict.explicit_intent_summary = ecr_r
            judge_status["ecr"] = "ok"

        # TS (also drives ts_dim3_pass into the calculator)
        if isinstance(ts_r, BaseException) or ts_r is None:
            judge_status["ts"] = "failed"
            if isinstance(ts_r, BaseException):
                logger.warning("TSJudge raised: %s", ts_r)
        else:
            verdict.tool_call_summary = ts_r.tool_call_summary
            ts_dim3_pass = ts_r.dim3_pass
            judge_status["ts"] = "ok"

        # IFS (must land before FactVerifier — verifier mutates in-place)
        if isinstance(ifs_r, BaseException) or ifs_r is None:
            judge_status["ifs"] = "failed"
            if isinstance(ifs_r, BaseException):
                logger.warning("IFSJudge raised: %s", ifs_r)
        else:
            verdict.fact_summary = ifs_r.fact_summary
            verdict.rubric_row_judgments = ifs_r.rubric_row_judgments
            judge_status["ifs"] = "ok"

        # IISR
        if isinstance(iisr_r, BaseException) or iisr_r is None:
            # Empty implicit_intent ⇒ IISR=1.0 unconditionally; treat as
            # a vacuous skip rather than a failure so audit dashboards
            # don't paint the case red.
            if not gt.implicit_intent:
                judge_status["iisr"] = "skipped_empty_gt"
            else:
                judge_status["iisr"] = "failed"
                if isinstance(iisr_r, BaseException):
                    logger.warning("IISRJudge raised: %s", iisr_r)
        else:
            verdict.implicit_intent_judgments = iisr_r
            judge_status["iisr"] = "ok"

        # ---- Runtime stats back-fill --------------------------------------
        verdict.conversation_turn_count = len(inputs.annotated_history)
        verdict.total_tokens = session_total_tokens

        # ---- Stage 3: Web-search verification -----------------------------
        if self.enable_verification:
            try:
                await self.verifier.verify(verdict)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("FactVerifier (fact_summary) failed: %s", exc)
            try:
                await self.verifier.verify_rubric_rows(verdict)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("FactVerifier (rubric_rows) failed: %s", exc)

        # ---- Stage 4: Meta-Judge audit (pre-metric verdict patching) ------
        if self.enable_meta_judge and self.meta_judge is not None:
            try:
                audit_report = await self.meta_judge.audit(
                    verdict=verdict,
                    ground_truth=gt,
                    conversation_history=inputs.annotated_history,
                )
                patched, _annotated = self.patcher.apply(
                    verdict, audit_report.corrections
                )
                verdict = patched
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("MetaJudge failed, keeping raw verdict: %s", exc)

        # ---- Stage 5: deterministic scoring -------------------------------
        scores: MetricScores = self.calculator.compute(
            verdict,
            gt,
            inputs.raw_conversation_history,
            ts_dim3_pass=ts_dim3_pass,
            judge_status=judge_status,
        )
        scores.e2e_latency_ms = round(e2e_latency_ms, 2)
        scores.input_tokens = session_input_tokens
        scores.output_tokens = session_output_tokens
        scores.total_tokens = session_total_tokens
        # CEI depends on SES + total_tokens; must be computed after tokens are set.
        self.calculator.backfill_cei(scores)

        # Deprecated legacy provenance fields — kept one release for
        # downstream Excel consumers. Read judge_status instead.
        iisr_source = (
            "standalone" if judge_status["iisr"] == "ok"
            else ("skipped_empty_gt" if judge_status["iisr"] == "skipped_empty_gt"
                  else "main_judge_fallback")
        )
        ifs_source = (
            "standalone" if judge_status["ifs"] == "ok"
            else "main_judge_fallback"
        )

        return EvalResult(
            case_id=case_id,
            verdict=verdict,
            scores=scores,
            judge_status=judge_status,
            iisr_source=iisr_source,  # type: ignore[arg-type]
            ifs_source=ifs_source,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_avg_e2e_latency(messages: list[dict[str, Any]]) -> float:
        """Average end-to-end latency (ms) across all assistant turns.

        数据源：``conversation_history[].TTFT``（dialogue_simulator 历史以来用
        ``TTFT`` 字段挂 per-turn 全链路耗时；这里保留原始字段读法，仅在评估
        模块出口侧统一改名为 e2e_latency）。
        """
        latencies: list[float] = []
        for msg in messages:
            if msg.get("role") == "assistant":
                v = msg.get("TTFT")
                if v is not None:
                    latencies.append(float(v))
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_default_agent(
    llm_provider: BaseLLMProvider,
    *,
    language: str = "chinese",
    enable_meta_judge: bool = False,
) -> JudgeAgent:
    """Convenience factory used by the FastAPI route and by tests.

    Uses the real :class:`GoogleWebSearchClient` (MCP gateway). All 5
    standalone judges are always wired; there is no opt-out flag.
    ``enable_meta_judge=True`` runs the Stage-4 Devil's-Advocate audit
    and applies allowlisted corrections before metric calculation.
    """
    return JudgeAgent(
        llm_provider,
        search_client=GoogleWebSearchClient(),
        language=language,
        enable_meta_judge=enable_meta_judge,
    )
