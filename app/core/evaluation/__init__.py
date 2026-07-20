"""Evaluation module: JudgeAgent, metrics, fact verification, Meta-Judge.

Recommended public API — two top-level coroutines hide the internal
wiring for most callers::

    from app.core.evaluation import evaluate_case, evaluate_batch

    result = await evaluate_case(case, llm_provider=provider)
    report = await evaluate_batch(cases, llm_provider=provider)

Lower-level building blocks are also re-exported for advanced use
(custom pipelines, tests, CLI scripts)::

    from app.core.evaluation import JudgeAgent, build_default_agent
    from app.core.evaluation.schema import (
        GroundTruth, JudgeVerdict, MetricScores, EvalResult,
        Correction, MetaJudgeReport,
    )
    from app.core.evaluation.metrics_summary import (
        aggregate_batch, BatchReport, format_batch_report,
    )
"""

from .metrics_summary import (
    BatchReport,
    MetricStats,
    RuntimeMetricStats,
    aggregate_batch,
    format_batch_report,
    zero_metric_scores,
)
from .schema import (
    Correction,
    EvalResult,
    FactStatement,
    GroundTruth,
    ImplicitIntent,
    ImplicitJudgment,
    JudgeVerdict,
    MetaJudgeReport,
    MetricScores,
)

# judge_agent imports MetaJudge / VerdictPatcher transitively; defer the
# import so the package remains usable even if a heavyweight dependency
# (e.g. httpx) is missing during installation.
try:  # pragma: no cover - import guard
    from .judge_agent import JudgeAgent, build_default_agent  # type: ignore F401
except ImportError:  # judge_agent.py not yet created
    JudgeAgent = None  # type: ignore
    build_default_agent = None  # type: ignore

try:  # pragma: no cover - import guard
    from .meta_judge import MetaJudge, VerdictPatcher  # type: ignore F401
except ImportError:
    MetaJudge = None  # type: ignore
    VerdictPatcher = None  # type: ignore

# 5-way standalone judges + shared input preprocessor. Imported lazily so
# unit tests / scripts that only need the data contracts don't pay the
# pydantic/llm-provider import cost; failures fall back to ``None`` so a
# missing transitive dep doesn't break ``from app.core.evaluation import …``.
try:  # pragma: no cover - import guard
    from .judge_inputs import JudgeInputs, prepare_judge_inputs  # type: ignore F401
except ImportError:
    JudgeInputs = None  # type: ignore
    prepare_judge_inputs = None  # type: ignore

try:  # pragma: no cover - import guard
    from .ecr_judge import ECRJudge  # type: ignore F401
    from .ts_judge import TSJudge  # type: ignore F401
    from .ifs_judge import IFSJudge  # type: ignore F401
    from .iisr_judge import IISRJudge  # type: ignore F401
except ImportError:
    ECRJudge = None  # type: ignore
    TSJudge = None  # type: ignore
    IFSJudge = None  # type: ignore
    IISRJudge = None  # type: ignore

# Top-level facade — depends on JudgeAgent import above.
try:  # pragma: no cover - import guard
    from .evaluator import (  # type: ignore F401
        BatchEvalReport,
        evaluate_batch,
        evaluate_case,
    )
except ImportError:
    BatchEvalReport = None  # type: ignore
    evaluate_case = None  # type: ignore
    evaluate_batch = None  # type: ignore

__all__ = [
    # Facade (recommended entry points)
    "evaluate_case",
    "evaluate_batch",
    "BatchEvalReport",
    # Low-level orchestration
    "JudgeAgent",
    "build_default_agent",
    "MetaJudge",
    "VerdictPatcher",
    # 4-way standalone judges + shared input bundle
    "JudgeInputs",
    "prepare_judge_inputs",
    "ECRJudge",
    "TSJudge",
    "IFSJudge",
    "IISRJudge",
    # Batch aggregation primitives
    "aggregate_batch",
    "BatchReport",
    "MetricStats",
    "RuntimeMetricStats",
    "format_batch_report",
    "zero_metric_scores",
    # Data contracts
    "GroundTruth",
    "ImplicitIntent",
    "ImplicitJudgment",
    "FactStatement",
    "JudgeVerdict",
    "MetricScores",
    "EvalResult",
    "Correction",
    "MetaJudgeReport",
]
