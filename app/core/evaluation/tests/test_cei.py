"""Unit tests for CEI (Cost Efficiency Index).

公式：CEI = AR · 1/(1+exp(Eff)) / total_tokens · 1_000_000

测试覆盖：
  * _eff_score：sigmoid 边界 + 数值溢出软兜底
  * compute_cei：标准公式 / Eff=None 透传 / total_tokens<=0 / AR=0 / 极端 eff
  * details_cei：skip 信息 / 中间值 round
  * MetricCalculator.compute() 端到端：scores.CEI 与独立 compute_cei 一致
  * aggregate_batch：CEI 进 BatchReport.cei，cei_per_case 入参 fallback 行为
"""

from __future__ import annotations

import math

import pytest

from app.core.evaluation.metrics.calculator import MetricCalculator, _compute_ar
from app.core.evaluation.metrics.cei import (
    _eff_score,
    compute_cei,
    details_cei,
)
from app.core.evaluation.metrics_summary import aggregate_batch
from app.core.evaluation.schema import (
    ClarificationPolicy,
    ExplicitIntentSummary,
    GroundTruth,
    ImplicitIntent,
    ImplicitJudgment,
    JudgeVerdict,
    MetricScores,
    ToolCallDetail,
    ToolCallRubric,
    ToolCallSummary,
    TruthTrajectory,
)


# ─── _eff_score ──────────────────────────────────────────────────────────


def test_eff_score_at_zero():
    """Eff=0 → eff_score=0.5（agent 贴 baseline）。"""
    assert _eff_score(0.0) == 0.5


def test_eff_score_positive_below_half():
    """Eff>0（agent 慢）→ eff_score<0.5。"""
    assert _eff_score(1.0) < 0.5
    assert _eff_score(2.0) < _eff_score(1.0)


def test_eff_score_negative_above_half():
    """Eff<0（agent 快）→ eff_score>0.5。"""
    assert _eff_score(-1.0) > 0.5
    assert _eff_score(-2.0) > _eff_score(-1.0)


def test_eff_score_overflow_clamps_to_zero():
    """Eff 过大（math.exp 溢出阈值 ~709）软兜底返 0.0，不应抛 OverflowError。"""
    assert _eff_score(1000.0) == 0.0
    assert _eff_score(800.0) == 0.0


def test_eff_score_underflow_clamps_to_one():
    """Eff 极负 → 软兜底返 1.0。"""
    assert _eff_score(-1000.0) == 1.0
    assert _eff_score(-800.0) == 1.0


# ─── 测试用 fixtures ─────────────────────────────────────────────────────


def _gt(max_allowed: int = 3, with_implicit: bool = False) -> GroundTruth:
    kwargs = dict(
        explicit_intent=["a", "b"],
        clarification_policy=ClarificationPolicy(max_allowed=max_allowed),
        truth_trajectory=TruthTrajectory(
            tool_calls=ToolCallRubric(expected_tools=["poi_search"]),
        ),
    )
    if with_implicit:
        kwargs["implicit_intent"] = [
            ImplicitIntent(
                rubric_instruction="x", constraint_type="soft",
                evidence_confidence=1.0, evidence="",
            ),
        ]
    return GroundTruth(**kwargs)


def _verdict(
    *,
    ecr_succ: int = 2,
    ecr_total: int = 2,
    turns: int = 3,
    total_tokens: int = 1_000_000,
    iisr_score: float | None = None,
) -> JudgeVerdict:
    """构造 verdict；ecr=succ/total，conversation_turn_count=turns。"""
    impl_judgments = []
    if iisr_score is not None:
        impl_judgments = [
            ImplicitJudgment(
                rubric_instruction="x",
                confidence_Wi=1.0,
                satisfaction_score_Ci=iisr_score,
                triggered=True,
            ),
        ]
    return JudgeVerdict(
        explicit_intent_summary=ExplicitIntentSummary(
            total_count=ecr_total, success_count=ecr_succ,
            success_intents=["s"] * ecr_succ,
        ),
        tool_call_summary=ToolCallSummary(
            correct_calls=[ToolCallDetail(tool_name="poi_search")],
        ),
        implicit_intent_judgments=impl_judgments,
        conversation_turn_count=turns,
        total_tokens=total_tokens,
    )


def _history(turn_index: int) -> list[dict]:
    """构造最小 conversation_history，让 Eff 走 max_allowed 路径。

    Eff 在有 history 时：actual=max(turn_index)，median=clarification_policy.max_allowed。
    """
    return [{"turn_index": turn_index, "role": "assistant"}]


# ─── compute_cei: 标准路径 ───────────────────────────────────────────────


def test_compute_cei_baseline_eff_zero():
    """Eff=0（actual_turns == max_allowed）+ AR=1 + tokens=1M → CEI=0.5。

    AR=1.0, eff_score=0.5, CEI = 1.0 * 0.5 / 1_000_000 * 1_000_000 = 0.5
    """
    gt = _gt(max_allowed=3)
    v = _verdict(turns=3, total_tokens=1_000_000)
    cei = compute_cei(v, gt, conversation_history=_history(3))
    assert cei == pytest.approx(0.5)


def test_compute_cei_faster_than_baseline_higher_score():
    """Eff<0（agent 快）→ eff_score>0.5 → CEI 比 baseline 高。"""
    gt = _gt(max_allowed=10)
    v_fast = _verdict(turns=3, total_tokens=1_000_000)
    v_baseline = _verdict(turns=10, total_tokens=1_000_000)

    cei_fast = compute_cei(v_fast, gt, conversation_history=_history(3))
    cei_baseline = compute_cei(v_baseline, gt, conversation_history=_history(10))

    assert cei_fast > cei_baseline
    assert cei_baseline == pytest.approx(0.5)


def test_compute_cei_slower_than_baseline_lower_score():
    """Eff>0（agent 慢）→ eff_score<0.5 → CEI 比 baseline 低。"""
    gt = _gt(max_allowed=3)
    v_slow = _verdict(turns=10, total_tokens=1_000_000)
    cei_slow = compute_cei(v_slow, gt, conversation_history=_history(10))
    assert cei_slow < 0.5
    assert cei_slow > 0


def test_compute_cei_tokens_scale_inverse():
    """CEI 与 total_tokens 反比。tokens 翻倍 → CEI 减半。"""
    gt = _gt(max_allowed=3)
    v_low_token = _verdict(turns=3, total_tokens=500_000)
    v_high_token = _verdict(turns=3, total_tokens=1_000_000)

    cei_low = compute_cei(v_low_token, gt, conversation_history=_history(3))
    cei_high = compute_cei(v_high_token, gt, conversation_history=_history(3))

    assert cei_low == pytest.approx(2 * cei_high, rel=1e-6)


# ─── compute_cei: 边界与跳过 ─────────────────────────────────────────────


def test_compute_cei_eff_none_returns_none():
    """clarification_policy.max_allowed=0 → Eff=None → CEI=None（与 add_cei.py 一致）。"""
    gt = _gt(max_allowed=0)
    v = _verdict(turns=3, total_tokens=1_000_000)
    history = [{"turn_index": 1, "role": "user"}]  # 让 Eff 走 history 路径
    assert compute_cei(v, gt, conversation_history=history) is None


def test_compute_cei_zero_tokens_returns_zero():
    """total_tokens=0 → CEI=0.0（防除零）。"""
    gt = _gt(max_allowed=3)
    v = _verdict(turns=3, total_tokens=0)
    assert compute_cei(v, gt) == 0.0


def test_compute_cei_negative_tokens_returns_zero():
    """total_tokens<0（异常输入）→ CEI=0.0。"""
    gt = _gt(max_allowed=3)
    v = _verdict(turns=3, total_tokens=-100)
    assert compute_cei(v, gt) == 0.0


def test_compute_cei_zero_ar_returns_zero():
    """AR=0（ECR=0）→ CEI=0.0（公式分子为 0）。"""
    gt = _gt(max_allowed=3)
    v = _verdict(ecr_succ=0, ecr_total=2, turns=3, total_tokens=1_000_000)
    assert compute_cei(v, gt) == 0.0


def test_compute_cei_partial_ar():
    """AR=0.5（ECR=0.5, IISR=1）+ Eff=0 + tokens=1M → CEI = 0.5·0.5/1 = 0.25。"""
    gt = _gt(max_allowed=3)
    v = _verdict(ecr_succ=1, ecr_total=2, turns=3, total_tokens=1_000_000)
    assert compute_cei(v, gt, conversation_history=_history(3)) == pytest.approx(0.25)


def test_compute_cei_with_iisr_component():
    """AR = ECR·IISR；IISR=0.6 时 AR=0.6 → CEI = 0.6·0.5/1 = 0.3。"""
    gt = _gt(max_allowed=3, with_implicit=True)
    v = _verdict(turns=3, total_tokens=1_000_000, iisr_score=0.6)
    assert compute_cei(v, gt, conversation_history=_history(3)) == pytest.approx(0.3)


# ─── details_cei ─────────────────────────────────────────────────────────


def test_details_cei_skipped_when_eff_none():
    gt = _gt(max_allowed=0)
    v = _verdict(turns=3, total_tokens=1_000_000)
    history = [{"turn_index": 1, "role": "user"}]
    d = details_cei(v, gt, conversation_history=history)
    assert d["skipped"] is True
    assert d["cei"] is None
    assert d["eff"] is None
    assert "max_allowed=0" in d["skip_reason"]


def test_details_cei_zero_tokens_floored():
    gt = _gt(max_allowed=3)
    v = _verdict(turns=3, total_tokens=0)
    d = details_cei(v, gt)
    assert d["skipped"] is False
    assert d["cei"] == 0.0
    assert "total_tokens<=0" in d["skip_reason"]


def test_details_cei_normal_path_includes_intermediates():
    gt = _gt(max_allowed=3)
    v = _verdict(turns=3, total_tokens=1_000_000)
    d = details_cei(v, gt, conversation_history=_history(3))
    assert d["skipped"] is False
    assert d["ar"] == pytest.approx(1.0)
    assert d["eff"] == pytest.approx(0.0)
    assert d["eff_score"] == pytest.approx(0.5)
    assert d["total_tokens"] == 1_000_000
    assert d["cei"] == pytest.approx(0.5)


# ─── MetricCalculator 端到端 ─────────────────────────────────────────────


def test_calculator_emits_cei_in_scores_and_details():
    """MetricCalculator.compute() 输出 MetricScores.CEI + details["CEI"]。"""
    calc = MetricCalculator()
    gt = _gt(max_allowed=3)
    v = _verdict(turns=3, total_tokens=1_000_000)

    scores = calc.compute(v, gt, conversation_history=_history(3))

    assert scores.CEI == pytest.approx(0.5)
    assert "CEI" in scores.details
    assert scores.details["CEI"]["cei"] == pytest.approx(0.5)
    # 与独立调用一致
    assert scores.CEI == compute_cei(v, gt, conversation_history=_history(3))


def test_calculator_cei_none_when_eff_none():
    """端到端：Eff=None 时 scores.CEI 也是 None（与 add_cei.py 行为一致）。"""
    calc = MetricCalculator()
    gt = _gt(max_allowed=0)
    v = _verdict(turns=3, total_tokens=1_000_000)
    history = [{"turn_index": 1, "role": "user"}]
    scores = calc.compute(v, gt, conversation_history=history)
    assert scores.Eff is None
    assert scores.CEI is None


def test_calculator_cei_uses_same_ar():
    """cei.py 内部的 _compute_ar 与 calculator.py 的 _compute_ar 数值一致。"""
    gt = _gt(max_allowed=3, with_implicit=True)
    v = _verdict(turns=3, total_tokens=1_000_000, iisr_score=0.7)
    calc = MetricCalculator()
    scores = calc.compute(v, gt, conversation_history=None)
    # cei 内部用 _compute_ar 算 AR；calculator 也用 _compute_ar 填 scores.AR
    assert _compute_ar(v, gt) == pytest.approx(scores.AR)


def test_metricscores_round_4dp_applies_to_cei():
    """schema field_validator 把 CEI 也 round 到 4 位小数。"""
    s = MetricScores(
        ECR=1, TS=1, IFS=1, IISR=1, AR=1,
        Eff=0.0, CEI=0.123456789, details={},
    )
    assert s.CEI == 0.1235


# ─── aggregate_batch ─────────────────────────────────────────────────────


def test_aggregate_batch_reads_cei_from_scores():
    """CEI 已进 MetricScores → aggregate_batch 默认从 scores 抽，无需 cei_per_case。"""
    scores_list = [
        MetricScores(ECR=1, TS=1, IFS=1, IISR=1, AR=1, Eff=0.0, CEI=0.5, details={}),
        MetricScores(ECR=1, TS=1, IFS=1, IISR=1, AR=1, Eff=0.0, CEI=0.3, details={}),
    ]
    report = aggregate_batch(scores_list)
    assert report.cei.n == 2
    assert report.cei.mean == pytest.approx(0.4)


def test_aggregate_batch_skips_none_cei():
    """CEI=None 的 case 从聚合中排除（与 Eff 一致语义）。"""
    scores_list = [
        MetricScores(ECR=1, TS=1, IFS=1, IISR=1, AR=1, Eff=0.0, CEI=0.5, details={}),
        MetricScores(ECR=1, TS=1, IFS=1, IISR=1, AR=1, Eff=None, CEI=None, details={}),
        MetricScores(ECR=1, TS=1, IFS=1, IISR=1, AR=1, Eff=0.0, CEI=0.3, details={}),
    ]
    report = aggregate_batch(scores_list)
    assert report.cei.n == 2  # None 排除
    assert report.cei.mean == pytest.approx(0.4)


def test_aggregate_batch_falls_back_to_cei_per_case_when_scores_have_no_cei():
    """所有 scores.CEI=None 时，使用 cei_per_case 入参（add_cei.py 路径）。"""
    scores_list = [
        MetricScores(ECR=1, TS=1, IFS=1, IISR=1, AR=1, Eff=0.0, CEI=None, details={}),
        MetricScores(ECR=1, TS=1, IFS=1, IISR=1, AR=1, Eff=0.0, CEI=None, details={}),
    ]
    report = aggregate_batch(scores_list, cei_per_case=[0.7, 0.9])
    assert report.cei.n == 2
    assert report.cei.mean == pytest.approx(0.8)


def test_aggregate_batch_prefers_scores_over_cei_per_case():
    """scores 里有 CEI 时优先用 scores，cei_per_case 入参被忽略。"""
    scores_list = [
        MetricScores(ECR=1, TS=1, IFS=1, IISR=1, AR=1, Eff=0.0, CEI=0.5, details={}),
    ]
    report = aggregate_batch(scores_list, cei_per_case=[99.9])  # 入参故意给离谱值
    assert report.cei.mean == pytest.approx(0.5)  # 走 scores，不走入参


def test_aggregate_batch_empty_inputs_zero_stats():
    """空入参 → BatchReport.cei 全 0。"""
    report = aggregate_batch([])
    assert report.cei.n == 0
    assert report.cei.mean == 0.0
