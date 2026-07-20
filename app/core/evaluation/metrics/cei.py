"""CEI — Cost Efficiency Index (Token 性价比).

Formula:

    CEI = (SES / TotalTokens) × 10^6

Where:
  * SES         — Satisfaction Efficiency Score = AR × Eff，满意效能分，取值 [0, 1)。
  * TotalTokens — Agent 在整个对话 session 中消耗的总 token 数
                  （input_tokens + output_tokens）。

Properties:
  * CEI ≥ 0，无上界。
  * CEI 含义：每百万 Token 消耗所产出的综合效能。越高 → 性价比越优。
  * CEI = 0 时表示 SES=0（任务完全未完成）或 token 数据缺失。
  * 当 TotalTokens ≤ 0 或 SES 为 None 时，CEI 返回 None，
    batch 聚合时排除该 case。

Typical ranges (参考):
  * CEI > 1000   → 极高效（低 token 消耗 + 高 SES）
  * CEI 200~1000 → 正常区间
  * CEI < 200    → token 开销偏大或任务完成度/效率偏低
"""

from __future__ import annotations

from typing import Any

SCALE_FACTOR = 1_000_000  # 10^6, 归一化到"每百万 token"


def compute_cei(ses: float | None, total_tokens: int) -> float | None:
    """Compute CEI = (SES / TotalTokens) × 10^6.

    Returns ``None`` when:
      - SES is ``None`` (Eff invalid → SES invalid → CEI invalid)
      - total_tokens ≤ 0 (token 数据缺失)
    """
    if ses is None or total_tokens <= 0:
        return None
    return (ses / total_tokens) * SCALE_FACTOR


def details_cei(ses: float | None, total_tokens: int) -> dict[str, Any]:
    """Provenance dict for CEI."""
    if ses is None or total_tokens <= 0:
        return {
            "formula": "(SES / TotalTokens) × 10^6",
            "ses": ses,
            "total_tokens": total_tokens,
            "cei_score": None,
            "skipped": True,
            "skip_reason": (
                "SES is None or TotalTokens ≤ 0; CEI is not computed "
                "and this case is excluded from batch CEI aggregation."
            ),
        }
    cei = (ses / total_tokens) * SCALE_FACTOR
    return {
        "formula": "(SES / TotalTokens) × 10^6",
        "ses": round(ses, 4),
        "total_tokens": total_tokens,
        "cei_score": round(cei, 4),
        "skipped": False,
    }
