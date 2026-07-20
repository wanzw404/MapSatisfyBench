"""Eff — Interaction Efficiency (bounded harmonic ratio).

Formula:

    Eff = S_median / (S_median + S_actual)

Properties:
  * Eff ∈ (0, 1)，天然有界，无需截断。
  * Eff = 0.5 → 实际轮次与人工基线持平（S_actual == S_median）
  * Eff > 0.5 → Agent 比人工更高效（轮次更少）
  * Eff < 0.5 → Agent 比人工更冗余（轮次更多）
  * Eff → 1   → 极高效（S_actual → 0，理论上限）
  * Eff → 0   → 极冗余（S_actual → ∞）

Where:
  * S_median — Agent 澄清预算（baseline）：
      ``gt.clarification_policy.max_allowed``。当 ``conversation_history``
      不可用时（legacy 调用方 / 老测试），回落到
      ``gt.efficiency.human_median_turns``。
      Clamped at minimum 1 so 除零安全。
  * S_actual — 实际轮次：
      ``conversation_history`` 可用时取 ``max(turn_index)``；
      否则回落到 ``verdict.conversation_turn_count``。Clamped at min 1。

Skip 条件：``clarification_policy.max_allowed == 0`` 视为 GT 无效标注，
``compute_eff`` 返回 ``None``，batch 聚合时排除该 case。
"""

from __future__ import annotations

from typing import Any

from app.core.evaluation.schema import GroundTruth, JudgeVerdict

DEFAULT_HUMAN_MEDIAN = 10


def _read_human_median(gt: GroundTruth) -> int:
    """Pull `human_median_turns` from either attribute or pydantic extras (legacy)."""
    val = getattr(gt, "human_median_turns", None)
    if isinstance(val, (int, float)) and val > 0:
        return int(val)
    extra: dict[str, Any] | None = getattr(gt, "model_extra", None)
    if extra:
        raw = extra.get("human_median_turns")
        if isinstance(raw, (int, float)) and raw > 0:
            return int(raw)
        eff_cfg = extra.get("efficiency")
        if isinstance(eff_cfg, dict):
            raw2 = eff_cfg.get("human_median_turns")
            if isinstance(raw2, (int, float)) and raw2 > 0:
                return int(raw2)
    return DEFAULT_HUMAN_MEDIAN


def _read_human_median_from_clarification(gt: GroundTruth) -> int:
    """Pull ``max_allowed`` from ``clarification_policy``.

    ``max_allowed == 0`` 是合法值（不允许澄清），但对效率公式而言
    median=0 会除零，所以这里仍然返回 0；上游 ``_is_gt_invalid`` 会先把
    这种 case 当作无效 GT 跳过。其它非数值 / 缺失走 fallback default。
    """
    cp = getattr(gt, "clarification_policy", None)
    raw = getattr(cp, "max_allowed", None)
    if isinstance(raw, bool):
        return DEFAULT_HUMAN_MEDIAN
    if isinstance(raw, (int, float)) and raw >= 0:
        return int(raw)
    return DEFAULT_HUMAN_MEDIAN


def _max_turn_index(history: list[dict[str, Any]] | None) -> int | None:
    """Highest ``turn_index`` across **assistant** messages only.

    User messages are excluded because the last user turn is often a closing
    remark (e.g. "好的，谢谢") that should not count as an interaction round.
    In the bundled history format, user turns also receive ``turn_index + 1``
    relative to the paired assistant turn, so including them would inflate the
    actual turn count by 1.
    """
    if not history:
        return None
    indices: list[int] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        if "turn_index" in msg:
            try:
                indices.append(int(msg["turn_index"]))
            except (TypeError, ValueError):
                pass
    return max(indices) if indices else None


def _is_gt_invalid(gt: GroundTruth, conversation_history: list[dict[str, Any]] | None) -> bool:
    """``clarification_policy.max_allowed == 0`` 当作无效标注（除零）。"""
    if conversation_history is None:
        return False
    cp = getattr(gt, "clarification_policy", None)
    raw = getattr(cp, "max_allowed", None)
    if isinstance(raw, bool):
        return False
    return isinstance(raw, (int, float)) and int(raw) == 0


def _resolve_actual_and_median(
    verdict: JudgeVerdict,
    gt: GroundTruth,
    conversation_history: list[dict[str, Any]] | None,
) -> tuple[int, int]:
    if conversation_history is not None:
        from_turn_index = _max_turn_index(conversation_history)
        actual = from_turn_index if from_turn_index is not None else verdict.conversation_turn_count
        median = _read_human_median_from_clarification(gt)
    else:
        actual = verdict.conversation_turn_count
        median = _read_human_median(gt)
    return max(1, int(actual)), max(1, int(median))


def compute_eff(
    verdict: JudgeVerdict,
    gt: GroundTruth,
    conversation_history: list[dict[str, Any]] | None = None,
) -> float | None:
    if _is_gt_invalid(gt, conversation_history):
        return None

    actual, median = _resolve_actual_and_median(verdict, gt, conversation_history)
    return median / (median + actual)

def details_eff(
    verdict: JudgeVerdict,
    gt: GroundTruth,
    conversation_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if _is_gt_invalid(gt, conversation_history):
        return {
            "formula": "median / (median + actual)",
            "actual_turns": None,
            "human_median": None,
            "eff_score": None,
            "source": "max(turn_index) / clarification_policy.max_allowed",
            "skipped": True,
            "skip_reason": (
                "ground_truth.clarification_policy.max_allowed=0 — invalid "
                "GT annotation; Eff is not computed and this case is excluded "
                "from batch Eff aggregation."
            ),
        }

    actual, median = _resolve_actual_and_median(verdict, gt, conversation_history)
    return {
        "formula": "median / (median + actual)",
        "actual_turns": actual,
        "human_median": median,
        "eff_score": round(median / (median + actual), 4),
        "source": (
            "max(turn_index) / clarification_policy.max_allowed"
            if conversation_history is not None
            else "verdict.conversation_turn_count / efficiency.human_median_turns"
        ),
        "skipped": False,
    }
