"""metrics_summary — cross-case statistical summary for a batch of EvalResults.

Design (2026-05):

    A single case is scored by eight independent metrics (ECR, TS, IFS,
    IISR, AR, Eff, SES, CEI) with NO weighted aggregation. The statistic for
    the whole sample set is therefore just the **per-case arithmetic mean**
    of each metric, optionally accompanied by cheap diagnostics (median,
    std, min, max) so that outliers are visible.

    Runtime metrics that are not in [0, 1] (E2E latency in ms, Tokens per task)
    are reported with mean + p50 / p95 / p99 + min / max + sum, because
    they are heavy-tailed and a single mean would be misleading.

Denominator policy (2026-05-26):
    The mean denominator is **the length of the scores iterable passed
    in** — this module does no filtering of "successful vs failed" cases.
    Callers are expected to fold failures into the input by inserting
    :func:`zero_metric_scores` placeholders so failed cases count as 0
    for ECR/TS/IFS/AR. ``Eff`` still self-filters cases that are
    mathematically invalid (``Eff is None`` or ``details.Eff.skipped``)
    — a failed case must therefore pass ``Eff=0.0`` (not ``None``) if it
    should appear in the Eff mean.
    Runtime stats (E2E latency / tokens) intentionally only see successful
    cases — physical measurements with no data should not be zero-filled
    (would poison p95 / p99).

This module is pure Python — no LLM, no IO, no heavy pydantic logic.
All returned numbers are plain ``float``/``int`` — easy to serialise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable, Sequence

from app.core.evaluation.schema import GroundTruth, JudgeVerdict, MetricScores


# ---------------------------------------------------------------------------
# Low-level statistics helpers
# ---------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return var ** 0.5


def _percentile(xs: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile. ``p`` is a fraction in [0, 1]."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    k = p * (len(s) - 1)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    frac = k - f
    return float(s[f] + (s[c] - s[f]) * frac)


# ---------------------------------------------------------------------------
# Dataclasses for the aggregate report
# ---------------------------------------------------------------------------


@dataclass
class MetricStats:
    """Per-case mean and basic dispersion for a single metric.

    The primary number is :attr:`mean` — the arithmetic mean of the
    metric value across all cases. ``median / std / min / max`` are
    cheap extras useful for spotting a heavy-tailed distribution or a
    single-case outlier.
    """

    metric: str
    n: int
    mean: float
    median: float
    std: float
    min: float
    max: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "n": self.n,
            "mean": round(self.mean, 4),
            "median": round(self.median, 4),
            "std": round(self.std, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
        }


@dataclass
class RuntimeMetricStats:
    """Heavy-tailed runtime metric: latency / token counts."""

    metric: str
    count: int
    mean: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float
    sum: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "count": self.count,
            "mean": round(self.mean, 2),
            "p50": round(self.p50, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "sum": round(self.sum, 2),
        }


@dataclass
class BatchReport:
    """Complete cross-case statistical report.

    For each of the six evaluation metrics we report the per-case mean
    (the primary summary) plus median / std / min / max.

    Runtime blocks (``e2e_latency_ms`` / ``tokens_per_task``) are ``None`` when
    no data is supplied.
    """

    n_cases: int
    ecr: MetricStats = field(default_factory=lambda: _zero_stats("ECR"))
    ts: MetricStats = field(default_factory=lambda: _zero_stats("TS"))
    ifs: MetricStats = field(default_factory=lambda: _zero_stats("IFS"))
    iisr: MetricStats = field(default_factory=lambda: _zero_stats("IISR"))
    ar: MetricStats = field(default_factory=lambda: _zero_stats("AR"))
    eff: MetricStats = field(default_factory=lambda: _zero_stats("Eff"))
    ses: MetricStats = field(default_factory=lambda: _zero_stats("SES"))
    # CEI 由 add_cei.py 后处理写入 metrics.CEI，公式见该脚本；不在 [0,1] 区间，
    # 取 case-mean 即可，沿用 MetricStats（避免引入新数据类）。
    cei: MetricStats = field(default_factory=lambda: _zero_stats("CEI"))
    e2e_latency_ms: RuntimeMetricStats | None = None

    tokens_per_task: RuntimeMetricStats | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_cases": self.n_cases,
            "ECR": self.ecr.as_dict(),
            "TS": self.ts.as_dict(),
            "IFS": self.ifs.as_dict(),
            "IISR": self.iisr.as_dict(),
            "AR": self.ar.as_dict(),
            "Eff": self.eff.as_dict(),
            "SES": self.ses.as_dict(),
            "CEI": self.cei.as_dict(),
            "E2E_Latency_ms": (
                self.e2e_latency_ms.as_dict() if self.e2e_latency_ms else None
            ),
            "Tokens_per_task": (
                self.tokens_per_task.as_dict() if self.tokens_per_task else None
            ),
        }


def _zero_stats(name: str) -> MetricStats:
    return MetricStats(
        metric=name, n=0, mean=0.0, median=0.0, std=0.0, min=0.0, max=0.0,
    )


def zero_metric_scores() -> MetricScores:
    """Failure placeholder: ECR/TS/IFS/AR/Eff/SES/CEI all 0.0.
    Used by callers to inject failed / unparseable cases into the
    denominator so the mean reflects the full task population. ``Eff``
    is set to ``0.0`` (not ``None``) so the failure also counts in the
    Eff mean — ``None`` would route the case through the Eff
    self-filter and quietly drop it.
    """
    return MetricScores(
        ECR=0.0, TS=0.0, IFS=0.0, AR=0.0, Eff=0.0, SES=0.0, CEI=0.0, details={},
    )


# ---------------------------------------------------------------------------
# Main aggregation entry point
# ---------------------------------------------------------------------------


def _metric_stats(name: str, values: Sequence[float]) -> MetricStats:
    if not values:
        return _zero_stats(name)
    return MetricStats(
        metric=name,
        n=len(values),
        mean=_mean(values),
        median=float(median(values)),
        std=_std(values),
        min=float(min(values)),
        max=float(max(values)),
    )


def _runtime_stats(name: str, values: Sequence[float]) -> RuntimeMetricStats | None:
    if not values:
        return None
    return RuntimeMetricStats(
        metric=name,
        count=len(values),
        mean=_mean(values),
        p50=_percentile(values, 0.50),
        p95=_percentile(values, 0.95),
        p99=_percentile(values, 0.99),
        min=float(min(values)),
        max=float(max(values)),
        sum=float(sum(values)),
    )


def aggregate_batch(
    scores: Iterable[MetricScores],
    verdicts: Iterable[JudgeVerdict] | None = None,
    ground_truths: Iterable[GroundTruth] | None = None,
    *,
    e2e_latency_ms_per_case: Iterable[float] | None = None,
    tokens_per_case: Iterable[int] | None = None,
    cei_per_case: Iterable[float] | None = None,
) -> BatchReport:
    """Compute a :class:`BatchReport` across a collection of per-case scores.

    Parameters
    ----------
    scores : required
        Iterable of ``MetricScores`` (one per case).
    verdicts : optional
        Iterable of the ``JudgeVerdict`` that produced each score, in
        the same order. Currently only used to back-fill token counts
        when ``tokens_per_case`` is not supplied.
    ground_truths : optional
        Iterable of ``GroundTruth`` for each case. Currently unused but
        kept for forward compatibility.
    e2e_latency_ms_per_case : optional
        Per-case end-to-end latency (ms). Absent entries are skipped.
    tokens_per_case : optional
        Per-case total token count. Falls back to ``verdict.total_tokens``
        when omitted.
    cei_per_case : optional
        Per-case CEI (Cost Efficiency Index) values，由 add_cei.py 后处理
        写入。CEI 不在 [0,1]，按 case-mean 计入 BatchReport.cei；失败/缺测
        case 应由调用方填 0 入 list（与其它指标一致的 zero-fill 口径）。
    """

    scores_list = list(scores)
    verdicts_list = list(verdicts) if verdicts is not None else []
    _ = list(ground_truths) if ground_truths is not None else []  # reserved

    n = len(scores_list)
    if n == 0:
        return BatchReport(n_cases=0)

    ecr_vals = [float(s.ECR) for s in scores_list]
    ts_vals = [float(s.TS) for s in scores_list]
    ifs_vals = [float(s.IFS) for s in scores_list]
    iisr_vals = [float(s.IISR) for s in scores_list]
    ar_vals = [float(s.AR) for s in scores_list]
    # Eff: skip cases where the GT is invalid (e.g. max_allowed=0). Such
    # cases have ``s.Eff is None`` AND/OR ``details.Eff.skipped == True``;
    # both are treated as excluded so they don't bias mean/median/std.
    eff_vals = [
        float(s.Eff)
        for s in scores_list
        if s.Eff is not None
        and not (isinstance(s.details, dict) and (s.details.get("Eff") or {}).get("skipped"))
    ]
    # SES: same skip logic as Eff (SES depends on Eff).
    ses_vals = [
        float(s.SES)
        for s in scores_list
        if s.SES is not None
        and not (isinstance(s.details, dict) and (s.details.get("SES") or {}).get("skipped"))
    ]
    # CEI: skip when None (SES invalid or total_tokens ≤ 0).
    cei_vals = [
        float(s.CEI)
        for s in scores_list
        if s.CEI is not None
        and not (isinstance(s.details, dict) and (s.details.get("CEI") or {}).get("skipped"))
    ]

    # Runtime metrics
    e2e_values: list[float] = []
    if e2e_latency_ms_per_case is not None:
        e2e_values = [float(x) for x in e2e_latency_ms_per_case if x is not None]

    token_values: list[float] = []
    if tokens_per_case is not None:
        token_values = [float(x) for x in tokens_per_case if x is not None]
    elif verdicts_list:
        token_values = [
            float(v.total_tokens)
            for v in verdicts_list
            if v is not None and v.total_tokens > 0
        ]

    cei_values: list[float] = []
    if cei_per_case is not None:
        cei_values = [float(x) for x in cei_per_case if x is not None]

    return BatchReport(
        n_cases=n,
        ecr=_metric_stats("ECR", ecr_vals),
        ts=_metric_stats("TS", ts_vals),
        ifs=_metric_stats("IFS", ifs_vals),
        iisr=_metric_stats("IISR", iisr_vals),
        ar=_metric_stats("AR", ar_vals),
        eff=_metric_stats("Eff", eff_vals),
        ses=_metric_stats("SES", ses_vals),
        cei=_metric_stats("CEI", cei_values),
        e2e_latency_ms=_runtime_stats("E2E_Latency_ms", e2e_values),
        tokens_per_task=_runtime_stats("Tokens_per_task", token_values),
    )


# ---------------------------------------------------------------------------
# Pretty-printer (for CLI scripts)
# ---------------------------------------------------------------------------


def format_batch_report(report: BatchReport) -> str:
    """Human-readable multi-line summary of a :class:`BatchReport`."""

    if report.n_cases == 0:
        return "  (no cases to aggregate)"

    lines: list[str] = []
    lines.append(f"  N = {report.n_cases} cases")
    lines.append("")

    hdr = (
        f"  {'metric':<8} {'n':>3} {'mean':>7} {'median':>7} "
        f"{'std':>7} {'min':>7} {'max':>7}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for m in (
        report.ecr, report.ts, report.ifs, report.iisr,
        report.ar, report.eff, report.ses, report.cei,
    ):
        lines.append(
            f"  {m.metric.upper():<8} {m.n:>3} {m.mean:>7.3f} "
            f"{m.median:>7.3f} {m.std:>7.3f} {m.min:>7.3f} {m.max:>7.3f}"
        )
    lines.append("")

    if report.e2e_latency_ms:
        r = report.e2e_latency_ms
        lines.append(
            f"  E2E_Latency(ms)   N={r.count}  mean={r.mean:.1f}  "
            f"p50={r.p50:.1f}  p95={r.p95:.1f}  p99={r.p99:.1f}  "
            f"min={r.min:.1f}  max={r.max:.1f}"
        )
    if report.tokens_per_task:
        r = report.tokens_per_task
        lines.append(
            f"  Tokens/task       N={r.count}  mean={r.mean:.1f}  "
            f"p50={r.p50:.1f}  p95={r.p95:.1f}  "
            f"min={r.min:.0f}  max={r.max:.0f}  sum={r.sum:.0f}"
        )

    return "\n".join(lines)
