"""SES — Satisfaction Efficiency Score (满意效能分).

Formula:

    SES = AR × Eff

Where:
  * AR  — Accepted-response Probability = ECR · IISR，取值 [0, 1]，衡量 Agent 是否
    真正完成了用户需求。
  * Eff — Interaction Efficiency = S_median / (S_median + S_actual)，
    取值 (0, 1)，衡量 Agent 相对于人工基线的轮次消耗。

Properties:
  * SES ∈ [0, 1)，天然有界。
  * SES = 0.5 → Agent 综合表现与人工基线持平（AR=1, Eff=0.5）
  * SES > 0.5 → Agent 综合表现优于人工基线
  * SES < 0.5 → Agent 综合表现劣于人工基线
  * AR 管"做没做完"，Eff 管"有没有浪费轮次"，乘积意味着
    任一维度的缺陷都无法被另一维度弥补（短板效应）。

Skip 条件：当 ``Eff`` 为 ``None``（GT 无效标注）时，
``compute_ses`` 也返回 ``None``，batch 聚合时排除该 case。
"""

from __future__ import annotations

from typing import Any


def compute_ses(ar: float, eff: float | None) -> float | None:
    """Compute SES = AR × Eff.

    Returns ``None`` when Eff is ``None`` (invalid GT → skip in batch).
    """
    if eff is None:
        return None
    return max(0.0, min(1.0, ar * eff))


def details_ses(ar: float, eff: float | None) -> dict[str, Any]:
    """Provenance dict for SES."""
    if eff is None:
        return {
            "formula": "AR × Eff",
            "ar": None,
            "eff": None,
            "ses_score": None,
            "skipped": True,
            "skip_reason": (
                "Eff is None (invalid GT annotation); SES is not computed "
                "and this case is excluded from batch SES aggregation."
            ),
        }
    ses = max(0.0, min(1.0, ar * eff))
    return {
        "formula": "AR × Eff",
        "ar": round(ar, 4),
        "eff": round(eff, 4),
        "ses_score": round(ses, 4),
        "skipped": False,
    }
