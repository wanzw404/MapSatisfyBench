"""Top-level evaluation facade — two high-level entry points.

This module is the **public API** of the evaluation subsystem. It hides the
internal JudgeAgent / MetaJudge / MetricCalculator / BatchAggregator wiring
behind two coroutines that most callers should use:

* :func:`evaluate_case`  — run the full per-case pipeline on ONE case and
  return a :class:`EvalResult` carrying the six metrics + verdict + audit.
* :func:`evaluate_batch` — run the per-case pipeline over N cases and
  return a :class:`BatchEvalReport` carrying every :class:`EvalResult`
  plus the cross-case statistical roll-up (:class:`BatchReport`).

Both functions accept either a pre-built :class:`JudgeAgent` (reuse across
many cases to avoid rebuilding LLM / search clients) or just an
:class:`BaseLLMProvider`, in which case a default agent is constructed on
the fly.

A "case" is a plain dict with the following keys (matching the existing
fixture format used by ``scripts/mock_eval_cases.py``):

    {
        "case_id" | "conversation_id":  str,
        "query":              str,
        "full_intent":        str,
        "current_time":       str,
        "current_location":   str,
        "conversation_history" | "conversation_history_messages": list[dict],
        "ground_truth":       dict | GroundTruth,
        "tools_schema":       list[dict] | None,     # optional
        "session_stats":      dict | None,            # optional
    }

Legacy callers may still set a ``persona`` key in the case dict — it is
silently ignored (no longer used by the evaluation pipeline).

The inference-result top-level shape ``{conversation_id,
conversation_history_messages, session_stats}`` is therefore consumed
directly — ``conversation_id`` doubles as ``case_id`` when the latter is
absent.

Design note (2026-05): per-case scoring is six independent metrics with no
weighted aggregation; batch scoring is the per-case arithmetic mean of each
metric (+ median/std/min/max as diagnostics). Neither layer produces a
pass/fail classification.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Optional

from app.core.agent.llm.base import BaseLLMProvider
from app.core.evaluation.metrics_summary import (
    BatchReport,
    aggregate_batch,
    format_batch_report,
    zero_metric_scores,
)
from app.core.evaluation.judge_agent import JudgeAgent, build_default_agent
from app.core.evaluation.schema import EvalResult, GroundTruth

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result envelope for the batch API
# ---------------------------------------------------------------------------


@dataclass
class BatchEvalReport:
    """Bundle of per-case results + cross-case statistical roll-up.

    Attributes
    ----------
    results : list[EvalResult]
        One entry per successfully-evaluated case, in the same order as the
        input iterable (failed cases may be omitted if ``on_case_error`` is
        set to ``"skip"``).
    stats : BatchReport
        Per-metric mean / median / std / min / max across ``results``.
    failures : list[tuple[str, str]]
        ``(case_id, error_message)`` pairs for cases that raised when
        ``on_case_error="skip"`` was active. Always empty when the default
        ``"raise"`` policy is used.
    """

    results: list[EvalResult]
    stats: BatchReport
    failures: list[tuple[str, str]]

    @property
    def n_cases(self) -> int:
        return len(self.results)

    def format(self) -> str:
        """Human-readable CLI summary — metric table + failure tally."""
        lines = [format_batch_report(self.stats)]
        if self.failures:
            lines.append("")
            lines.append(f"  failures: {len(self.failures)}")
            for cid, msg in self.failures:
                lines.append(f"    - {cid}: {msg}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_agent(
    agent: Optional[JudgeAgent],
    llm_provider: Optional[BaseLLMProvider],
    *,
    language: str,
    enable_verification: bool,
    enable_meta_judge: bool,
) -> JudgeAgent:
    """Return the supplied agent, or build a default one from the provider."""
    if agent is not None:
        # Honour the verification switch even when a pre-built agent is
        # injected — callers commonly flip this per-invocation.
        agent.enable_verification = enable_verification
        return agent
    if llm_provider is None:
        raise ValueError(
            "evaluate_case/evaluate_batch: either `agent` or `llm_provider` "
            "must be supplied."
        )
    built = build_default_agent(
        llm_provider,
        language=language,
        enable_meta_judge=enable_meta_judge,
    )
    built.enable_verification = enable_verification
    return built


def _extract_history(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Accept either the bundled or flat OpenAI conversation schema."""
    history = (
        case.get("conversation_history_messages")
        or case.get("conversation_history")
        or []
    )
    return list(history)


# ---------------------------------------------------------------------------
# Interface 1 — single case
# ---------------------------------------------------------------------------


async def evaluate_case(
    case: Mapping[str, Any],
    *,
    agent: Optional[JudgeAgent] = None,
    llm_provider: Optional[BaseLLMProvider] = None,
    language: str = "chinese",
    enable_verification: bool = True,
    enable_meta_judge: bool = False,
) -> EvalResult:
    """Evaluate a single case through the full JudgeAgent pipeline.

    Parameters
    ----------
    case :
        Mapping with the keys documented in the module docstring.
    agent :
        A pre-built :class:`JudgeAgent` to reuse. When omitted, one is
        constructed from ``llm_provider`` + the toggle kwargs.
    llm_provider :
        Required when ``agent`` is not supplied; ignored otherwise.
    language, enable_verification, enable_meta_judge :
        Passed through to :func:`build_default_agent` when a fresh agent
        has to be built. ``enable_verification`` is additionally applied
        to a supplied ``agent`` so the caller can flip it per call.

    Returns
    -------
    EvalResult
        Envelope carrying the six metrics, the (patched) verdict, the
        fact-verification log and — when Meta-Judge ran — the audit report.
    """
    resolved = _resolve_agent(
        agent,
        llm_provider,
        language=language,
        enable_verification=enable_verification,
        enable_meta_judge=enable_meta_judge,
    )

    ground_truth = case["ground_truth"]
    if not isinstance(ground_truth, GroundTruth):
        ground_truth = GroundTruth.model_validate(ground_truth)

    # Accept either ``case_id`` (canonical) or ``conversation_id`` (the field
    # name used by the inference-result top-level dict).
    case_id = case.get("case_id") or case.get("conversation_id")
    if not case_id:
        raise KeyError("case must contain either 'case_id' or 'conversation_id'")

    return await resolved.evaluate(
        case_id=case_id,
        query=case["query"],
        full_intent=case["full_intent"],
        current_time=case["current_time"],
        current_location=case["current_location"],
        conversation_history=_extract_history(case),
        ground_truth=ground_truth,
        tools_schema=case.get("tools_schema") or [],
        session_stats=case.get("session_stats"),
    )


# ---------------------------------------------------------------------------
# Interface 2 — batch of cases
# ---------------------------------------------------------------------------


OnCaseError = Literal["raise", "skip"]


async def evaluate_batch(
    cases: Iterable[Mapping[str, Any]],
    *,
    agent: Optional[JudgeAgent] = None,
    llm_provider: Optional[BaseLLMProvider] = None,
    language: str = "chinese",
    enable_verification: bool = True,
    enable_meta_judge: bool = False,
    concurrency: int = 1,
    on_case_error: OnCaseError = "raise",
    e2e_latency_ms_per_case: Optional[Iterable[float]] = None,
) -> BatchEvalReport:
    """Evaluate a batch of cases and return per-case + cross-case statistics.

    All cases share a single :class:`JudgeAgent`; build it once via
    ``llm_provider`` or pass one in via ``agent``. ``concurrency > 1``
    caps the number of simultaneously in-flight LLM calls using an
    ``asyncio.Semaphore`` — helpful when the provider tolerates
    parallelism but sequential is safer by default.

    Parameters
    ----------
    cases :
        Iterable of case dicts (see module docstring for the schema).
    agent / llm_provider / language /
    enable_verification / enable_meta_judge :
        Same meaning as in :func:`evaluate_case`.
    concurrency :
        Maximum in-flight evaluations. ``1`` = strictly sequential.
    on_case_error :
        * ``"raise"`` (default): the first failing case aborts the batch.
        * ``"skip"``:            the case is dropped, logged, and its
          ``(case_id, error)`` tuple appears in
          :attr:`BatchEvalReport.failures`.
    e2e_latency_ms_per_case :
        Optional per-case end-to-end latency (ms). Must match the order
        of ``cases`` when supplied.

    Returns
    -------
    BatchEvalReport
        ``results`` preserves input order (modulo skipped failures);
        ``stats`` is the :class:`BatchReport` produced by
        :func:`aggregate_batch`.
    """
    cases_list: list[Mapping[str, Any]] = list(cases)
    if not cases_list:
        return BatchEvalReport(results=[], stats=BatchReport(n_cases=0), failures=[])

    resolved_agent = _resolve_agent(
        agent,
        llm_provider,
        language=language,
        enable_verification=enable_verification,
        enable_meta_judge=enable_meta_judge,
    )

    sem = asyncio.Semaphore(max(1, concurrency))
    results: list[Optional[EvalResult]] = [None] * len(cases_list)
    failures: list[tuple[str, str]] = []

    async def _run_one(idx: int, c: Mapping[str, Any]) -> None:
        async with sem:
            try:
                results[idx] = await evaluate_case(c, agent=resolved_agent)
            except Exception as exc:
                case_id = str(
                    c.get("case_id") or c.get("conversation_id") or f"#{idx}"
                )
                if on_case_error == "raise":
                    raise
                logger.exception(
                    "evaluate_batch: case %s failed, skipping: %s", case_id, exc,
                )
                failures.append((case_id, f"{type(exc).__name__}: {exc}"))

    await asyncio.gather(
        *(_run_one(i, c) for i, c in enumerate(cases_list)),
    )

    successful: list[EvalResult] = [r for r in results if r is not None]

    # Denominator = total case count. Failed cases (results[i] is None)
    # are folded in as zero-score placeholders so the mean reflects the
    # full task population, not the survivor subset.
    scores_for_stats = [
        (r.scores if r is not None else zero_metric_scores())
        for r in results
    ]
    verdicts_for_stats = [
        (r.verdict if r is not None else None) for r in results
    ]

    # Runtime data (E2E_Latency) stays survivor-only: missing physical
    # measurements are not zero-filled (would poison p95 / p99).
    e2e_aligned: Optional[list[float]] = None
    if e2e_latency_ms_per_case is not None:
        e2e_full = list(e2e_latency_ms_per_case)
        if len(e2e_full) == len(cases_list):
            e2e_aligned = [
                e2e_full[i] for i, r in enumerate(results) if r is not None
            ]
        else:
            logger.warning(
                "evaluate_batch: e2e_latency_ms_per_case length (%d) does not match "
                "cases (%d); ignoring runtime stats.",
                len(e2e_full), len(cases_list),
            )

    stats = aggregate_batch(
        scores=scores_for_stats,
        verdicts=verdicts_for_stats,
        e2e_latency_ms_per_case=e2e_aligned,
    )

    return BatchEvalReport(results=successful, stats=stats, failures=failures)
