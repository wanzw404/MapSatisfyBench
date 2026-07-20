"""ECR - Explicit-decision-factor Completion Rate.

ECR = success_count / total_count   (explicit intents)

If there is no explicit intent in the ground truth, ECR is defined as 1.0
(vacuously satisfied).
"""

from __future__ import annotations

from typing import Any

from app.core.evaluation.schema import GroundTruth, JudgeVerdict


def compute_ecr(verdict: JudgeVerdict, gt: GroundTruth) -> float:
    total = verdict.explicit_intent_summary.total_count or len(gt.explicit_intent)
    success = verdict.explicit_intent_summary.success_count
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, success / total))


def details_ecr(verdict: JudgeVerdict, gt: GroundTruth) -> dict[str, Any]:
    total = verdict.explicit_intent_summary.total_count or len(gt.explicit_intent)
    success = verdict.explicit_intent_summary.success_count
    if total <= 0:
        return {
            "formula": "success / total (vacuous: 0/0 defined as 1.0)",
            "numerator": 0,
            "denominator": 0,
        }
    return {
        "formula": "success / total",
        "numerator": int(success),
        "denominator": int(total),
    }
