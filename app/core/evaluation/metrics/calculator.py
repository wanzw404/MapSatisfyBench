"""MetricCalculator — wires the six deterministic sub-metrics together.

Per-case scoring (deterministic, no LLM, no weighted aggregation):

    ECR   = N_success / N_total                          (explicit intent completion rate)
    TS    = |T_gold ∩ T_pred_correct| / |T_gold ∪ T_pred_all|
                                                         (tool selection Jaccard / IoU)
    IFS   = N_faithful_rows / N_total_rows              (per-row all-or-nothing faithfulness)
    IISR  = Σ(Wi·Ci) / Σ(Wi)   over triggered rubrics   (weighted implicit satisfaction)
    Eff   = S_median / (S_median + S_actual)            (bounded harmonic ratio)
    SES   = AR × Eff                                    (satisfaction efficiency score)
    CEI   = (SES / TotalTokens) × 10^6                  (cost efficiency index)

Design decision (2026-05):

* A single case is reported as six independent metrics. There is no
  weighted aggregation and no "overall pass/fail" at the per-case level.
  The legacy ``overall`` / ``breakdown`` convenience fields have been
  removed — callers should read the six primitives directly.
* Cross-case statistics (mean/median/std/min/max per metric across the
  whole sample set) live in :mod:`app.core.evaluation.metrics_summary`
  and are computed as macro means of each per-case metric independently.
"""

from __future__ import annotations

from typing import Any

from app.core.evaluation.schema import GroundTruth, JudgeVerdict, MetricScores

from .cei import compute_cei, details_cei
from .eff import compute_eff, details_eff
from .es import compute_ses, details_ses
from .ecr import compute_ecr, details_ecr
from .ifs import compute_ifs, details_ifs
from .iisr import compute_iisr, details_iisr
from .ts import compute_ts, details_ts

# Names of the seven per-case metrics, in canonical display order.
# IISR is surfaced top-level (alongside AR = ECR · IISR) so Excel /
# batch-report consumers don't need to dig into details["AR"] to find it.
METRIC_NAMES: tuple[str, ...] = (
    "ECR", "TS", "IFS", "IISR", "AR", "Eff", "SES", "CEI",
)


def _compute_ar(verdict: JudgeVerdict, gt: GroundTruth) -> float:
    """AR = ECR · IISR.

    * If ECR == 0 → AR = 0 (multiplication ensures this).
    * If GT has no implicit intent → IISR returns 1.0, so AR = ECR.
    """
    ecr = compute_ecr(verdict, gt)
    iisr = compute_iisr(verdict, gt)
    return max(0.0, min(1.0, ecr * iisr))


def _details_ar(verdict: JudgeVerdict, gt: GroundTruth) -> dict[str, Any]:
    ecr = compute_ecr(verdict, gt)
    iisr = compute_iisr(verdict, gt)
    return {
        "formula": "AR = ECR · IISR",
        "ecr_component": round(ecr, 4),
        "iisr_component": round(iisr, 4),
        "ar": round(ecr * iisr, 4),
        "iisr_breakdown": details_iisr(verdict, gt),
    }


class MetricCalculator:
    """Deterministic scorer. Stateless; no weights, no thresholds.

    ``compute()`` returns a :class:`MetricScores` carrying exactly the
    six primitive per-case metrics plus a ``details`` dict that records
    the numerator / denominator (or equivalent provenance) for each.

    ``conversation_history`` (optional) is the bundled-format history dict
    list straight from the inference pipeline. When supplied, TS and Eff
    switch to deterministic data-driven computation (TS: dedup by tool name
    + parameter_rules required-param check; Eff: max turn_index / clarification_policy.max_allowed).
    When omitted, both fall back to LLM-verdict-derived behaviour for
    legacy callers (e.g. unit-test fixtures).
    """

    def __init__(self) -> None:
        # No configuration needed; kept as a class only to match the
        # original interface used by JudgeAgent (self.calculator.compute).
        pass

    def compute(
        self,
        verdict: JudgeVerdict,
        gt: GroundTruth,
        conversation_history: list[dict[str, Any]] | None = None,
        *,
        ts_dim3_pass: set[str] | None = None,
        judge_status: dict[str, str] | None = None,
    ) -> MetricScores:
        """Score one case.

        ``ts_dim3_pass`` (keyword-only): the set of tool names the TS
        standalone dim3 LLM approved. Passed through to ``compute_ts`` /
        ``details_ts``. When ``None`` (legacy / unit-test path), TS dim3
        defaults to "pass" so the score reduces to dim1 ∧ dim2 only.

        ``judge_status`` (keyword-only): per-judge status dict from
        JudgeAgent. When a judge has ``"failed"``, its corresponding metric
        returns ``None`` instead of computing from schema defaults.
        """
        # Determine which metrics should be None due to judge failure
        failed_metrics: set[str] = set()
        if judge_status:
            if judge_status.get("ecr") == "failed":
                failed_metrics.add("ECR")
            if judge_status.get("ts") == "failed":
                failed_metrics.add("TS")
            if judge_status.get("ifs") == "failed":
                failed_metrics.add("IFS")
            if judge_status.get("iisr") == "failed":
                failed_metrics.add("IISR")
            # AR depends on ECR and IISR
            if "ECR" in failed_metrics or "IISR" in failed_metrics:
                failed_metrics.add("AR")

        # Compute metrics, returning None for failed ones
        ecr = None if "ECR" in failed_metrics else compute_ecr(verdict, gt)
        ts = None if "TS" in failed_metrics else compute_ts(verdict, gt, conversation_history, dim3_pass=ts_dim3_pass)
        ifs = None if "IFS" in failed_metrics else compute_ifs(verdict, gt)
        iisr = None if "IISR" in failed_metrics else compute_iisr(verdict, gt)

        # AR = ECR · IISR (None if either component is None)
        if "AR" in failed_metrics:
            ar = None
        elif ecr is not None and iisr is not None:
            ar = max(0.0, min(1.0, ecr * iisr))
        else:
            ar = None

        eff = compute_eff(verdict, gt, conversation_history)

        # SES = AR × Eff (None if either component is None)
        if ar is None or eff is None:
            ses = None
        else:
            ses = compute_ses(ar, eff)

        return MetricScores(
            ECR=ecr,
            TS=ts,
            IFS=ifs,
            IISR=iisr,
            AR=ar,
            Eff=eff,
            SES=ses,
            details={
                "ECR": None if "ECR" in failed_metrics else details_ecr(verdict, gt),
                "TS": None if "TS" in failed_metrics else details_ts(
                    verdict, gt, conversation_history, dim3_pass=ts_dim3_pass
                ),
                "IFS": None if "IFS" in failed_metrics else details_ifs(verdict, gt),
                "IISR": None if "IISR" in failed_metrics else details_iisr(verdict, gt),
                "AR": None if "AR" in failed_metrics else _details_ar(verdict, gt),
                "Eff": details_eff(verdict, gt, conversation_history),
                "CEI": None,
                "SES": None if ses is None or ar is None else details_ses(ar, eff),
            },
        )

    @staticmethod
    def backfill_cei(scores: MetricScores) -> None:
        """Back-fill CEI after ``total_tokens`` has been set on *scores*.

        CEI depends on ``SES`` (computed in :meth:`compute`) **and**
        ``total_tokens`` (back-filled by the caller, e.g. JudgeAgent,
        after ``compute()`` returns). This method must therefore be
        called *after* ``scores.total_tokens`` is assigned.
        """
        cei = compute_cei(scores.SES, scores.total_tokens)
        scores.CEI = cei
        if isinstance(scores.details, dict):
            scores.details["CEI"] = details_cei(scores.SES, scores.total_tokens)
