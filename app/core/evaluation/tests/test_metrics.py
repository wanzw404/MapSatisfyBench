"""Unit tests for the deterministic metric calculators.

These tests do NOT require LLM or network — they feed hand-built JudgeVerdict
and GroundTruth objects into each ``compute_*`` function and assert exact
numeric outputs.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from app.core.evaluation.metrics import (
    MetricCalculator,
    compute_eff,
    compute_ecr,
    compute_ifs,
    compute_iisr,
    compute_ts,
)
from app.core.evaluation.schema import (
    ClarificationPolicy,
    ExplicitIntentSummary,
    FactStatement,
    FactSummary,
    GroundTruth,
    ImplicitIntent,
    ImplicitJudgment,
    IncorrectToolCall,
    JudgeVerdict,
    ToolCallDetail,
    ToolCallRubric,
    ToolCallSummary,
    TruthTrajectory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gt_basic() -> GroundTruth:
    return GroundTruth(
        explicit_intent=["POI类型为川菜馆", "提供路线规划方案"],
        implicit_intent=[
            ImplicitIntent(
                rubric_instruction="附近=步行15分钟或驾车10分钟",
                constraint_type="soft",
                evidence_confidence=0.85,
                evidence="偏好驾车",
            ),
            ImplicitIntent(
                rubric_instruction="驾车导航必须推荐有停车场的餐厅",
                constraint_type="hard",
                evidence_confidence=0.68,
                evidence="历史驾车到达的餐厅85%有停车场",
            ),
        ],
        clarification_policy=ClarificationPolicy(max_allowed=3),
        truth_trajectory=TruthTrajectory(
            tool_calls=ToolCallRubric(
                expected_tools=["poi_search", "driving_route"],
            )
        ),
    )


def _verdict(**kwargs: Any) -> JudgeVerdict:
    """Build a JudgeVerdict with sensible defaults and field overrides."""
    defaults: dict[str, Any] = dict(
        explicit_intent_summary=ExplicitIntentSummary(
            total_count=2, success_count=2, success_intents=["a", "b"]
        ),
        tool_call_summary=ToolCallSummary(
            correct_calls=[
                ToolCallDetail(tool_name="poi_search"),
                ToolCallDetail(tool_name="driving_route"),
            ]
        ),
        fact_summary=FactSummary(),
        implicit_intent_judgments=[],
        clarification_turns=1,
        conversation_turn_count=4,
        total_tokens=800,
    )
    defaults.update(kwargs)
    return JudgeVerdict(**defaults)


# ---------------------------------------------------------------------------
# ECR
# ---------------------------------------------------------------------------


def test_ecr_full(gt_basic: GroundTruth) -> None:
    v = _verdict(
        explicit_intent_summary=ExplicitIntentSummary(total_count=2, success_count=2)
    )
    assert compute_ecr(v, gt_basic) == 1.0


def test_ecr_partial(gt_basic: GroundTruth) -> None:
    v = _verdict(
        explicit_intent_summary=ExplicitIntentSummary(total_count=2, success_count=1)
    )
    assert compute_ecr(v, gt_basic) == 0.5


def test_ecr_empty_ground_truth() -> None:
    gt = GroundTruth()
    v = _verdict(
        explicit_intent_summary=ExplicitIntentSummary(total_count=0, success_count=0)
    )
    assert compute_ecr(v, gt) == 1.0


# ---------------------------------------------------------------------------
# TS
# ---------------------------------------------------------------------------


def test_ts_perfect(gt_basic: GroundTruth) -> None:
    v = _verdict()
    assert compute_ts(v, gt_basic) == 1.0


def test_ts_missed_one(gt_basic: GroundTruth) -> None:
    v = _verdict(
        tool_call_summary=ToolCallSummary(
            correct_calls=[ToolCallDetail(tool_name="poi_search")],
            missed_calls=["driving_route"],
        )
    )
    # 1 correct (poi_search), 1 missed (driving_route).
    # numerator = 1; denominator = |correct|=1 + |incorrect|=0 + |extra|=0 + |missed|=1 = 2
    assert math.isclose(compute_ts(v, gt_basic), 0.5, rel_tol=1e-6)


def test_ts_wrong_parameters_penalised(gt_basic: GroundTruth) -> None:
    v = _verdict(
        tool_call_summary=ToolCallSummary(
            correct_calls=[ToolCallDetail(tool_name="poi_search")],
            incorrect_calls=[
                IncorrectToolCall(tool_name="driving_route", reason="bad origin")
            ],
            missed_calls=["driving_route"],
        )
    )
    # 1 correct (poi_search), 1 incorrect (driving_route, wrong params).
    # driving_route 已落 incorrect 桶 → 不再重复进 missed → missed = ∅。
    # numerator = 1; denominator = 1 + 1 + 0 + 0 = 2
    assert math.isclose(compute_ts(v, gt_basic), 1 / 2, rel_tol=1e-6)


def test_ts_same_tool_mixed_params(gt_basic: GroundTruth) -> None:
    """Regression: identical tool name with one correct + one wrong-params call
    must NOT collapse via set dedup. Old name-keyed Jaccard scored 1.0; new
    (name, params) keying scores 0.5."""
    v = _verdict(
        tool_call_summary=ToolCallSummary(
            correct_calls=[
                ToolCallDetail(
                    tool_name="poi_search", parameters={"keyword": "川菜"}
                ),
                ToolCallDetail(tool_name="driving_route"),
            ],
            incorrect_calls=[
                IncorrectToolCall(tool_name="poi_search", reason="missing city")
            ],
        )
    )
    # gold = {poi_search, driving_route}; both names covered → missed = ∅.
    # numerator = 2 (both correct keys' names are in gold)
    # denominator = |correct|=2 + |incorrect|=1 + |extra|=0 + |missed|=0 = 3
    assert math.isclose(compute_ts(v, gt_basic), 2 / 3, rel_tol=1e-6)


def test_ts_multiple_correct_same_tool(gt_basic: GroundTruth) -> None:
    """Two correct invocations of the same tool with different params should
    each count as a distinct numerator entry (and equal denominator entry)."""
    v = _verdict(
        tool_call_summary=ToolCallSummary(
            correct_calls=[
                ToolCallDetail(
                    tool_name="poi_search", parameters={"keyword": "川菜"}
                ),
                ToolCallDetail(
                    tool_name="poi_search", parameters={"keyword": "火锅"}
                ),
                ToolCallDetail(tool_name="driving_route"),
            ],
        )
    )
    # 3 distinct correct keys; all names ∈ gold; no incorrect/extra/missed.
    # numerator = 3; denominator = 3 → 1.0
    assert compute_ts(v, gt_basic) == 1.0


def test_ts_details_shape_matches_target_format() -> None:
    """Regression: details_ts 输出形状契约（用户实测 case）。

    所有 4 个工具列表都是 ``list[str]``；reason 不在 details 里，而是落在
    ``verdict.tool_call_summary.incorrect_calls[*].reason`` 上（TSJudge 已写）。
    需要 reason 的下游（meta_judge / explainer）直接读 verdict slot。

    场景：GT 期待 3 个工具
      * search_user_action_summary（无 required params）→ 调用了 → correct
      * search_poi（required: keyword）→ 调用时缺 keyword → dim2 失败 → incorrect
        （桶互斥：不再重复进 missed）
      * ainative_kuake_search（required: query, query_write）→ 调用齐全 + dim3 pass → correct

    另外 2 个 extras：get_navigation / search_products_by_poiid（非 gold）
    """
    from app.core.evaluation.metrics.ts import details_ts

    gt = GroundTruth(
        truth_trajectory=TruthTrajectory(
            tool_calls=ToolCallRubric(
                expected_tools=[
                    "search_user_action_summary",
                    "search_poi",
                    "ainative_kuake_search",
                ],
                parameter_rules={
                    "search_poi": {"paramter": ["keyword"]},
                    "ainative_kuake_search": {
                        "paramter": ["query", "query_write"]
                    },
                },
            )
        ),
    )
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {"name": "search_user_action_summary", "args": {}},
                {"name": "search_poi", "args": {"city": "杭州"}},  # 缺 keyword
                {
                    "name": "ainative_kuake_search",
                    "args": {"query": "西湖", "query_write": "杭州西湖"},
                },
                {"name": "get_navigation", "args": {"dest": "西湖"}},
                {
                    "name": "search_products_by_poiid",
                    "args": {"poi_id": "B0001"},
                },
            ],
        },
    ]
    # TSJudge 已把 dim2 失败原因写到 verdict.tool_call_summary.incorrect_calls；
    # 此 fixture 模拟 _synthesise_summary 写完后的 verdict 状态。
    v = _verdict(
        tool_call_summary=ToolCallSummary(
            incorrect_calls=[
                IncorrectToolCall(
                    tool_name="search_poi",
                    reason="missing_required_params: keyword",
                ),
            ],
        )
    )
    dim3_pass = {"search_user_action_summary", "ainative_kuake_search"}

    out = details_ts(v, gt, conversation_history=history, dim3_pass=dim3_pass)

    # 旧字段彻底下线
    assert "dim3_pass_tools" not in out
    assert "dim3_fail_tools" not in out

    # 形状契约 — 4 个工具列表都是 list[str]
    assert out["correct_tools"] == [
        "ainative_kuake_search",
        "search_user_action_summary",
    ]
    assert out["incorrect_tools"] == ["search_poi"]
    assert out["extra_tools"] == ["get_navigation", "search_products_by_poiid"]
    # search_poi 落在 incorrect 桶 → 不再重复进 missed；本 case 三个 gold
    # 工具全部被调用过，所以 missed = ∅。
    assert out["missed_tools"] == []
    assert out["gold_tools"] == [
        "ainative_kuake_search",
        "search_poi",
        "search_user_action_summary",
    ]
    # 分子 2 / 分母 (2 correct + 1 incorrect + 2 extra + 0 missed) = 2/5
    assert out["numerator"] == 2
    assert out["denominator"] == 5
    assert "formula" in out and "buckets disjoint" in out["formula"]

    # reason 仍在 verdict slot 上可被下游读取
    assert v.tool_call_summary.incorrect_calls[0].reason == (
        "missing_required_params: keyword"
    )


def test_ts_buckets_are_disjoint_wrong_params() -> None:
    """互斥不变量：参数错的 gold 工具只能进 incorrect，不能也进 missed。

    覆盖 deterministic 路径 + legacy 路径两条计算分支；任一桶出现重复，
    分母就会虚高，TS 被低估（用户实测的 1/6 vs 1/5 就是这条 bug）。
    """
    from app.core.evaluation.metrics.ts import (
        _ts_breakdown_from_history,
        details_ts,
    )

    gt = GroundTruth(
        truth_trajectory=TruthTrajectory(
            tool_calls=ToolCallRubric(
                expected_tools=["A", "B", "C"],
                parameter_rules={"B": {"paramter": ["x"]}},
            )
        ),
    )
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {"name": "A", "args": {}},          # gold ∩ pred, dim2 trivially ok
                {"name": "B", "args": {"y": 1}},    # gold ∩ pred, 缺 x → incorrect
                {"name": "Z", "args": {}},          # 非 gold → extra
            ],
        },
    ]

    # —— deterministic 路径
    correct, incorrect, extra, missed = _ts_breakdown_from_history(gt, history)
    assert correct & incorrect == set()
    assert correct & extra == set()
    assert correct & missed == set()
    assert incorrect & extra == set()
    assert incorrect & missed == set(), "B 不能同时进 incorrect 和 missed"
    assert extra & missed == set()
    # 显式断言桶内容
    assert correct == {"A"}
    assert incorrect == {"B"}
    assert extra == {"Z"}
    assert missed == {"C"}  # 只有 C 没被调用

    # —— legacy 路径（构造一个等效的 verdict）
    v = _verdict(
        tool_call_summary=ToolCallSummary(
            correct_calls=[ToolCallDetail(tool_name="A")],
            incorrect_calls=[IncorrectToolCall(tool_name="B", reason="缺 x")],
            extra_calls=["Z"],
        )
    )
    out = details_ts(v, gt)
    assert "B" not in out["missed_tools"], "legacy 路径同样不能双计 B"
    assert set(out["missed_tools"]) == {"C"}
    # 分子 1 / 分母 1+1+1+1 = 1/4
    assert out["numerator"] == 1
    assert out["denominator"] == 4


# ---------------------------------------------------------------------------
# IFS
# ---------------------------------------------------------------------------


def test_ifs_all_faithful(gt_basic: GroundTruth) -> None:
    v = _verdict(
        fact_summary=FactSummary(
            faithful_facts=[
                FactStatement(statement="POI名称=川味轩", faithful=True),
            ]
        )
    )
    assert compute_ifs(v, gt_basic) == 1.0


def test_ifs_zero_tolerance_unfaithful() -> None:
    """新版 IFS：rubric 行有 1 条要素被标 fabricated → score=0 → IFS=0。"""
    from app.core.evaluation.schema import (
        RubricElementEvidence,
        RubricRowJudgment,
    )

    gt = GroundTruth(
        truth_trajectory=TruthTrajectory(
            tool_calls=ToolCallRubric(
                expected_tools=[],
                factual_answer_rubric=["营业时间"],
            )
        ),
    )
    v = _verdict(
        rubric_row_judgments=[
            RubricRowJudgment(
                rubric_row="营业时间",
                elements=[
                    RubricElementEvidence(
                        element="营业时间",
                        grounded=False,
                        reason="no_tool_grounding",
                        explanation="编造的营业时间，无 tool 支撑",
                    ),
                ],
                score=0,
                reasoning="营业时间无 tool 依据",
            ),
        ],
    )
    assert compute_ifs(v, gt) == 0.0


def test_ifs_rubric_row_zero_when_element_contradicted() -> None:
    """新版 IFS：行内任一要素 grounded=False（含 contradicted）→ 整行 0。"""
    from app.core.evaluation.schema import (
        RubricElementEvidence,
        RubricRowJudgment,
    )

    gt = GroundTruth(
        truth_trajectory=TruthTrajectory(
            tool_calls=ToolCallRubric(
                expected_tools=[],
                factual_answer_rubric=["距离信息"],
            )
        ),
    )
    v = _verdict(
        rubric_row_judgments=[
            RubricRowJudgment(
                rubric_row="距离信息",
                elements=[
                    RubricElementEvidence(
                        element="距离",
                        grounded=False,
                        reason="contradicted",
                        explanation="content 说 1.2km 但 tool 返回 2.8km",
                    ),
                ],
                score=0,
                reasoning="距离与工具返回矛盾，行判 0",
            ),
        ],
    )
    assert compute_ifs(v, gt) == 0.0


def test_ifs_vacuous_satisfaction_overrides_llm_zero() -> None:
    """LLM 把 element 标 absent 却写 score=0，代码应按 element deterministic 判 1。

    Regression: 实测 LLM 把 维度 1 ECR 的"答非所问"扣分逻辑误带到 IFS，对一个
    rubric_row（content 没提及该信息时）写出 element.reason='absent' 但 score=0。
    新版 _row_is_faithful 不再读 row.score，直接从 elements 推：
    全部非 skipped 元素 grounded=False 且 reason='absent' → 行 = 1（空满足）。
    """
    from app.core.evaluation.schema import (
        RubricElementEvidence,
        RubricRowJudgment,
    )

    gt = GroundTruth(
        truth_trajectory=TruthTrajectory(
            tool_calls=ToolCallRubric(
                expected_tools=[],
                factual_answer_rubric=["如答案提及用户画像，应与 search_user_profile 一致"],
            )
        ),
    )
    v = _verdict(
        rubric_row_judgments=[
            RubricRowJudgment(
                rubric_row="如答案提及用户画像，应与 search_user_profile 一致",
                elements=[
                    RubricElementEvidence(
                        element="用户画像",
                        grounded=False,
                        reason="absent",
                        explanation="content 未提及任何用户画像信息",
                    ),
                ],
                score=0,  # LLM 写错了
                reasoning="（LLM 误判：未识别意图就给 0）",
            ),
        ],
    )
    assert compute_ifs(v, gt) == 1.0


def test_ifs_mixed_absent_and_grounded_row_is_faithful() -> None:
    """混合行：absent 要素被忽略，剩余声称的要素全 grounded → 行 = 1。

    这是 IFS 真正的语义："只检查被声称的事实是否有依据"。content 没声称
    的部分（reason=absent）独立空满足，不能拿来卡 all-or-nothing。
    用户实测 case：行 = "起点和终点坐标必须一致"，content 只给了终点 +
    tool 支撑，起点没给 → 行应判 1（真实问题里的常见模式）。
    """
    from app.core.evaluation.schema import (
        RubricElementEvidence,
        RubricRowJudgment,
    )

    gt = GroundTruth(
        truth_trajectory=TruthTrajectory(
            tool_calls=ToolCallRubric(
                expected_tools=[],
                factual_answer_rubric=["起点和终点坐标必须与工具返回一致"],
            )
        ),
    )
    v = _verdict(
        rubric_row_judgments=[
            RubricRowJudgment(
                rubric_row="起点和终点坐标必须与工具返回一致",
                elements=[
                    RubricElementEvidence(
                        element="终点坐标",
                        grounded=True,
                        source_tool="search_poi",
                        content_quote="(119.388, 33.215)",
                        explanation="同轮工具返回吻合",
                    ),
                    RubricElementEvidence(
                        element="起点坐标",
                        grounded=False,
                        reason="absent",
                        explanation="回答未给出起点经纬度",
                    ),
                ],
                score=0,  # LLM 误把 absent 当扣分项
                reasoning="（LLM 误判：absent 触发 all-or-nothing）",
            ),
        ],
    )
    # absent 被忽略，剩余 [终点坐标] 全 grounded → 1
    assert compute_ifs(v, gt) == 1.0


def test_ifs_mixed_absent_and_unfaithful_row_is_unfaithful() -> None:
    """混合行：absent 被忽略，剩余有 contradicted → 行 = 0。

    把握原则：absent 不卡行，但其他 grounded=False 类型（contradicted /
    no_tool_grounding / external_verify_failed）依旧扣分。
    """
    from app.core.evaluation.schema import (
        RubricElementEvidence,
        RubricRowJudgment,
    )

    gt = GroundTruth(
        truth_trajectory=TruthTrajectory(
            tool_calls=ToolCallRubric(
                expected_tools=[],
                factual_answer_rubric=["A、B、C 三项一致"],
            )
        ),
    )
    v = _verdict(
        rubric_row_judgments=[
            RubricRowJudgment(
                rubric_row="A、B、C 三项一致",
                elements=[
                    RubricElementEvidence(
                        element="A", grounded=True, source_tool="t1",
                    ),
                    RubricElementEvidence(
                        element="B", grounded=False, reason="absent",
                    ),
                    RubricElementEvidence(
                        element="C", grounded=False, reason="contradicted",
                        explanation="content 5.0 但 tool 3.2",
                    ),
                ],
                score=0,
                reasoning="C 与 tool 矛盾",
            ),
        ],
    )
    # 剔 absent 后剩 [A grounded, C contradicted] → 不全 grounded → 0
    assert compute_ifs(v, gt) == 0.0


def test_ifs_unverified_facts_stay_faithful(gt_basic: GroundTruth) -> None:
    v = _verdict(
        fact_summary=FactSummary(
            faithful_facts=[
                FactStatement(
                    statement="距离1.2km", need_verify=True, verified_ok=None
                )
            ]
        )
    )
    assert compute_ifs(v, gt_basic) == 1.0


# ---------------------------------------------------------------------------
# IISR
# ---------------------------------------------------------------------------


def test_iisr_weighted_average(gt_basic: GroundTruth) -> None:
    v = _verdict(
        implicit_intent_judgments=[
            ImplicitJudgment(
                rubric_instruction="附近=步行15分钟或驾车10分钟",
                confidence_Wi=0.85,
                satisfaction_score_Ci=1.0,
                triggered=True,
            ),
            ImplicitJudgment(
                rubric_instruction="驾车导航必须推荐有停车场的餐厅",
                confidence_Wi=0.68,
                satisfaction_score_Ci=1.0,
                triggered=True,
            ),
        ]
    )
    # (0.85*1 + 0.68*1) / (0.85 + 0.68) = 1.0
    assert math.isclose(compute_iisr(v, gt_basic), 1.0, rel_tol=1e-6)


def test_iisr_hard_constraint_weighted_average(gt_basic: GroundTruth) -> None:
    """Hard-constraint violation no longer zeroes IISR; it participates in
    the weighted average normally (Ci=0 drags score down proportionally)."""
    v = _verdict(
        implicit_intent_judgments=[
            ImplicitJudgment(
                rubric_instruction="附近=步行15分钟或驾车10分钟",
                confidence_Wi=0.85,
                satisfaction_score_Ci=1.0,
                triggered=True,
            ),
            ImplicitJudgment(
                rubric_instruction="驾车导航必须推荐有停车场的餐厅",
                confidence_Wi=0.68,
                satisfaction_score_Ci=0.0,
                triggered=True,
                violates_hard_constraint=True,
            ),
        ]
    )
    # (0.85*1.0 + 0.68*0.0) / (0.85 + 0.68) = 0.85 / 1.53 ≈ 0.5556
    expected = 0.85 / (0.85 + 0.68)
    assert math.isclose(compute_iisr(v, gt_basic), expected, rel_tol=1e-4)


def test_iisr_no_explicit_intent_gate(gt_basic: GroundTruth) -> None:
    """ECR=0（所有 explicit intent 都失败）时，IISR 仍按 Σ(Wi·Ci)/Σ(Wi) 算，
    不再被强行置 0。该 gate 与 [[feedback_iisr_no_triggered_skip]] "每条 rubric
    必入分母" 原则冲突，且 AR = ECR · IISR 已经天然承担"explicit 全失败 → 整
    case 0 分"的语义，IISR 维度不该再被再次清零。

    Regression: 用户实测 case 1032da551... weighted_sum=0.7275, total_weight=
    1.97 → 应该 ≈0.3693，老版本因 ECR=0 报 0.0。
    """
    v = _verdict(
        # ECR 完全失败：success_count=0 但 total_count>0
        explicit_intent_summary=ExplicitIntentSummary(total_count=2, success_count=0),
        implicit_intent_judgments=[
            ImplicitJudgment(
                rubric_instruction="附近=步行15分钟或驾车10分钟",
                confidence_Wi=0.85,
                satisfaction_score_Ci=1.0,
                triggered=True,
            ),
            ImplicitJudgment(
                rubric_instruction="驾车导航必须推荐有停车场的餐厅",
                confidence_Wi=0.68,
                satisfaction_score_Ci=0.5,
                triggered=True,
            ),
        ],
    )
    expected = (0.85 * 1.0 + 0.68 * 0.5) / (0.85 + 0.68)
    assert math.isclose(compute_iisr(v, gt_basic), expected, rel_tol=1e-4)


def test_iisr_legacy_triggered_field_ignored(gt_basic: GroundTruth) -> None:
    """``triggered`` 字段被退役：每条 rubric 无条件入分母（GT 责任，不是
    判官的逃逸口）。即使 LLM 留下 legacy ``triggered=False``，第二条仍
    按 Wi=0.68 / Ci=0 计入分母，把整体 IISR 拉低。"""
    v = _verdict(
        implicit_intent_judgments=[
            ImplicitJudgment(
                rubric_instruction="附近=步行15分钟或驾车10分钟",
                confidence_Wi=0.85,
                satisfaction_score_Ci=0.5,
                triggered=True,
            ),
            ImplicitJudgment(
                rubric_instruction="驾车导航必须推荐有停车场的餐厅",
                confidence_Wi=0.68,
                satisfaction_score_Ci=0.0,
                triggered=False,  # legacy field — ignored now
            ),
        ]
    )
    expected = (0.85 * 0.5 + 0.68 * 0.0) / (0.85 + 0.68)
    assert math.isclose(compute_iisr(v, gt_basic), expected, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# Eff
# ---------------------------------------------------------------------------


def test_eff_matches_human_median(gt_basic: GroundTruth) -> None:
    # actual == median (10) → 10/(10+10) = 0.5（人工基线）
    v = _verdict(conversation_turn_count=10)
    assert math.isclose(compute_eff(v, gt_basic), 0.5, abs_tol=1e-9)


def test_eff_half_over_median(gt_basic: GroundTruth) -> None:
    # actual=15, median=10 → 10/(10+15) = 0.4（多用了 50% 轮次，低于基线）
    v = _verdict(conversation_turn_count=15)
    assert math.isclose(compute_eff(v, gt_basic), 0.4, rel_tol=1e-9)


def test_eff_double_median(gt_basic: GroundTruth) -> None:
    # actual = 2·median (20) → 10/(10+20) = 1/3 ≈ 0.3333（明显冗余）
    v = _verdict(conversation_turn_count=20)
    assert math.isclose(compute_eff(v, gt_basic), 1.0 / 3.0, rel_tol=1e-9)


def test_eff_faster_than_human(gt_basic: GroundTruth) -> None:
    # actual=5, median=10 → 10/(10+5) = 2/3 ≈ 0.6667；>0.5 表示优于人工
    v = _verdict(conversation_turn_count=5)
    assert math.isclose(compute_eff(v, gt_basic), 2.0 / 3.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Calculator (aggregation)
# ---------------------------------------------------------------------------


def test_calculator_perfect_case(gt_basic: GroundTruth) -> None:
    v = _verdict(
        conversation_turn_count=10,  # match default human_median → Eff=0.5
        # 至少一条 faithful 事实，避免 N_total=0 时 IFS 走"无证据"分支返回 0
        fact_summary=FactSummary(
            faithful_facts=[FactStatement(statement="poi_search 返回了川菜馆 X")]
        ),
        implicit_intent_judgments=[
            ImplicitJudgment(
                rubric_instruction="附近=步行15分钟或驾车10分钟",
                confidence_Wi=0.85,
                satisfaction_score_Ci=1.0,
                triggered=True,
            ),
            ImplicitJudgment(
                rubric_instruction="驾车导航必须推荐有停车场的餐厅",
                confidence_Wi=0.68,
                satisfaction_score_Ci=1.0,
                triggered=True,
            ),
        ]
    )
    scores = MetricCalculator().compute(v, gt_basic)
    assert scores.ECR == 1.0
    assert scores.TS == 1.0
    assert scores.IFS == 1.0
    assert scores.AR == 1.0  # AR = ECR(1.0) · IISR(1.0)
    assert scores.Eff == 0.5  # 10/(10+10) = 0.5（人工基线）
    assert scores.SES == 0.5  # AR(1.0) × Eff(0.5) = 0.5


def test_calculator_ifs_zero_is_isolated() -> None:
    """新版 IFS：rubric 行有 1 条 score=0 → IFS=0；其它 5 维独立保持 1.0。"""
    from app.core.evaluation.schema import (
        RubricElementEvidence,
        RubricRowJudgment,
    )

    # 自定义 GT：含 1 行 rubric + 2 条 implicit_intent
    gt = GroundTruth(
        explicit_intent=["POI类型为川菜馆", "提供路线规划方案"],
        implicit_intent=[
            ImplicitIntent(
                rubric_instruction="附近=步行15分钟或驾车10分钟",
                constraint_type="soft", evidence_confidence=0.85,
            ),
            ImplicitIntent(
                rubric_instruction="驾车导航必须推荐有停车场的餐厅",
                constraint_type="hard", evidence_confidence=0.68,
            ),
        ],
        clarification_policy=ClarificationPolicy(max_allowed=3),
        truth_trajectory=TruthTrajectory(
            tool_calls=ToolCallRubric(
                expected_tools=["poi_search", "driving_route"],
                factual_answer_rubric=["营业时间确认"],
            )
        ),
    )
    v = _verdict(
        conversation_turn_count=10,  # match default human_median → Eff=0.5
        rubric_row_judgments=[
            RubricRowJudgment(
                rubric_row="营业时间确认",
                elements=[
                    RubricElementEvidence(
                        element="营业时间",
                        grounded=False,
                        reason="no_tool_grounding",
                        explanation="编造营业时间，无 tool 支撑",
                    ),
                ],
                score=0,
                reasoning="营业时间要素无 tool 依据，行判 0",
            ),
        ],
        implicit_intent_judgments=[
            ImplicitJudgment(
                rubric_instruction="附近=步行15分钟或驾车10分钟",
                confidence_Wi=0.85, satisfaction_score_Ci=1.0, triggered=True,
            ),
            ImplicitJudgment(
                rubric_instruction="驾车导航必须推荐有停车场的餐厅",
                confidence_Wi=0.68, satisfaction_score_Ci=1.0, triggered=True,
            ),
        ],
    )
    scores = MetricCalculator().compute(v, gt)
    # IFS = 0/1 = 0; 其它维度独立保持预期值
    assert scores.IFS == 0.0
    assert scores.ECR == 1.0
    assert scores.TS == 1.0
    assert scores.AR == 1.0  # AR = ECR(1.0) · IISR(1.0)
    assert scores.Eff == 0.5  # 10/(10+10) = 0.5
    assert scores.SES == 0.5  # AR(1.0) × Eff(0.5) = 0.5
