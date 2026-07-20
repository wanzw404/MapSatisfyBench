"""Pydantic data contracts for the JudgeAgent evaluation pipeline.

Data flow:
    GroundTruth (input) + conversation_history (input)
      --LLM--> JudgeVerdict (intermediate, schema matches prompt output)
      --API--> JudgeVerdict (with verified_ok back-filled on facts)
      --CODE--> MetricScores (deterministic numbers)
    → EvalResult (envelope: verdict + scores + audit trail)
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Ground Truth
# ---------------------------------------------------------------------------


class ImplicitIntent(BaseModel):
    """One implicit-intent rubric from ground truth."""

    rubric_instruction: str
    constraint_type: Literal["hard", "soft"]
    evidence_confidence: float = Field(
        ge=0.0,
        description=(
            "Wi: weight of this rubric in IISR aggregation. "
            "标准范围 [0, 1]；上限完全放开兼容人工标注约定——标注者会用 1.1 / 1.2 "
            "甚至更高的值表示'超硬约束'（比满分还重要的关键 rubric）。"
            "IISR=Σ(Wi·Ci)/Σ(Wi) 是加权平均，Wi 任意正值都 ≤ max(Ci) ≤ 1，"
            "最终指标仍在 [0, 1]。"
        ),
    )
    evidence: str = ""
    evidence_source: str = ""


class ClarificationPolicy(BaseModel):
    """Budget for Agent's clarification turns."""

    max_allowed: int = Field(ge=0, description="Maximum clarification rounds allowed")
    evidence: str = ""


class ToolCallRubric(BaseModel):
    """Expected tool-call behaviour."""

    expected_tools: list[str] = Field(default_factory=list)
    parameter_rules: dict[str, Any] = Field(default_factory=dict)
    factual_answer_rubric: list[str] = Field(default_factory=list)


class TruthTrajectory(BaseModel):
    """Expected execution trajectory."""

    tool_calls: ToolCallRubric = Field(default_factory=ToolCallRubric)


class GroundTruth(BaseModel):
    """Complete ground-truth spec for a single evaluation case."""

    model_config = ConfigDict(extra="allow")  # keep any additional fields from dataset

    explicit_intent: list[str] = Field(default_factory=list)
    implicit_intent: list[ImplicitIntent] = Field(default_factory=list)
    clarification_policy: ClarificationPolicy = Field(
        default_factory=lambda: ClarificationPolicy(max_allowed=3)
    )
    truth_trajectory: TruthTrajectory = Field(default_factory=TruthTrajectory)


# ---------------------------------------------------------------------------
# Judge Verdict (LLM output schema + back-filled verification fields)
# ---------------------------------------------------------------------------


class ExplicitIntentSummary(BaseModel):
    total_count: int = 0
    success_count: int = 0
    success_intents: list[str] = Field(default_factory=list)
    failed_intents: list[str] = Field(default_factory=list)


class ToolCallDetail(BaseModel):
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class IncorrectToolCall(BaseModel):
    tool_name: str
    reason: str = ""


class ToolCallSummary(BaseModel):
    correct_calls: list[ToolCallDetail] = Field(default_factory=list)
    incorrect_calls: list[IncorrectToolCall] = Field(default_factory=list)
    extra_calls: list[str] = Field(default_factory=list)
    missed_calls: list[str] = Field(default_factory=list)


class FactStatement(BaseModel):
    """A single factual statement extracted from the Agent reply.

    `need_verify` is set by the LLM when it cannot fully confirm the fact
    from conversation_history alone (e.g. POI existence / live values).
    `verified_ok` is later back-filled by FactVerifier via the Amap API.
    """

    model_config = ConfigDict(extra="allow")

    statement: str
    source_tool: Optional[str] = None
    need_verify: bool = False
    verified_ok: Optional[bool] = None
    verify_reason: Optional[str] = None
    faithful: bool = True  # LLM initial judgement
    # Free-form: LLM 通常以分类码起头（fabricated / contradicted / verified_wrong），
    # 后接子类型与具体原因（"fabricated:(a) 工具未注册" / "contradicted: rubric X 与字段 Y"）。
    # 早期 schema 用 Literal 卡死了三选一，导致带说明的回答触发 ValidationError → 502；
    # 改为 str 后 LLM 原文得以保留下来排查，下游 verifier / meta_judge 仍按需写入纯码值。
    reason: Optional[str] = None


class FactSummary(BaseModel):
    """Fact-level judgement bucket, LLM outputs this, verifier augments it.

    旧版 IFS 走这一路（把 content 拆成单条事实再统计）。新版 IFS 走
    ``rubric_row_judgments``（按 factual_answer_rubric 行级判定），但本
    结构作为辅助 / 兼容老 verdict 保留。
    """

    model_config = ConfigDict(extra="allow")

    faithful_facts_count: int = 0
    faithful_facts: list[FactStatement] = Field(default_factory=list)
    unfaithful_facts: list[FactStatement] = Field(default_factory=list)


class RubricElementEvidence(BaseModel):
    """单个 rubric 要素的判定 + 证据。

    判定状态有三种：
      * ``grounded=True``：要素在 content 中有 tool 依据 / 通过外部验证
      * ``grounded=False``：要素缺失 / 无 tool 依据 / 与 tool 矛盾 / 外部验证否决
      * ``skipped=True``：需外部验证但**外部验证不可达**（API 挂 / 信息不足）
        → 该要素从 IFS 计算中**忽略**，既不算 grounded 也不算违反

    LLM 输出阶段：仅填 ``element / grounded / source_tool / content_quote /
    reason / explanation / need_external_verify``。``skipped`` 由代码层
    在 verifier 阶段决定，LLM 不需要也不该填。
    """

    model_config = ConfigDict(extra="allow")

    element: str
    grounded: bool = False
    source_tool: Optional[str] = None
    content_quote: str = ""
    reason: Optional[
        Literal[
            "absent",
            "no_tool_grounding",
            "contradicted",
            "external_verify_failed",
        ]
    ] = None
    explanation: str = ""

    # 外部验证相关（content 有声明但 tool 没字段支撑时由 LLM 标记）
    need_external_verify: bool = False
    external_verified_ok: Optional[bool] = None
    external_verify_reason: str = ""

    # 由 verifier 写入：外部验证不可达时该要素不参与 IFS 计算
    skipped: bool = False


class RubricRowJudgment(BaseModel):
    """factual_answer_rubric 单行的行级判定（IFS 新版打分单元）。

    判定逻辑（all-or-nothing，跳过项不参与）::

        非 skipped 要素全部 grounded=True → score = 1
        非 skipped 要素任一 grounded=False → score = 0
        所有要素都 skipped → 行级 skipped=True，整行从 IFS 中忽略
    """

    model_config = ConfigDict(extra="allow")

    rubric_row: str
    elements: list[RubricElementEvidence] = Field(default_factory=list)
    score: int = Field(default=0, ge=0, le=1)
    reasoning: str = ""

    # 由 verifier 写入：整行所有要素都不可验证时为 True，IFS 直接忽略此行
    skipped: bool = False


class ImplicitJudgment(BaseModel):
    model_config = ConfigDict(extra="allow")

    rubric_instruction: str
    # confidence_Wi 上限与 ImplicitIntent.evidence_confidence 一致：完全放开。
    # prompt 明确「Wi 直接从 evidence_confidence 复制，不再 re-judge」，输入
    # 1.5 / 2.0 时 LLM 原样回填，schema 不再做上限校验避免整组 IISR 被静默丢弃。
    confidence_Wi: float = Field(ge=0.0)
    satisfaction_score_Ci: float = Field(ge=0.0, le=1.0)
    triggered: bool = True
    reasoning: str = ""
    # Zero-tolerance switch: if True, IISR is forced to 0.0.
    violates_hard_constraint: bool = False


class JudgeVerdict(BaseModel):
    """Structured JSON produced by the LLM + back-filled verification marks."""

    model_config = ConfigDict(extra="allow")

    explicit_intent_summary: ExplicitIntentSummary = Field(
        default_factory=ExplicitIntentSummary
    )
    tool_call_summary: ToolCallSummary = Field(default_factory=ToolCallSummary)
    fact_summary: FactSummary = Field(default_factory=FactSummary)
    # 新版 IFS 用：factual_answer_rubric 每行 1 条判定（行级 all-or-nothing）。
    # 兼容老 verdict：缺失时 IFS 退回到 fact_summary 路径计算。
    rubric_row_judgments: list[RubricRowJudgment] = Field(default_factory=list)
    implicit_intent_judgments: list[ImplicitJudgment] = Field(default_factory=list)

    # Runtime-measured fields (may be set by LLM or back-filled by code):
    clarification_turns: int = 0
    conversation_turn_count: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Metric output envelope
# ---------------------------------------------------------------------------


class MetricScores(BaseModel):
    """Final deterministic scores, computed from JudgeVerdict + GroundTruth.

    Exactly the six per-case metrics — no weighted aggregation, no
    convenience roll-up. Cross-case statistics live in
    :mod:`app.core.evaluation.metrics_summary`.

    ``details`` carries the per-metric provenance (formula + numerator /
    denominator / auxiliary counters) so consumers can audit where each
    score came from without re-running the pipeline.
    """

    # 所有 float 指标统一保留 4 位小数（避免 Eff=0.6125467811865476 这种长尾）。
    # 通过下面的 field_validator 在赋值时自动 round 到 4。
    # 当对应 judge 重试后仍失败时，指标为 None（而非 schema 默认值）。
    ECR: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    TS: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    IFS: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    IISR: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description=(
            "Implicit-decision-factor Satisfaction Rate — weighted average of "
            "satisfaction_Ci over GT.implicit_intent rubrics. 1.0 when GT "
            "has no implicit rubric. None when IISRJudge fails after retry. "
            "Surfaced top-level for Excel/report consumption; AR = ECR · IISR "
            "still computed independently."
        ),
    )
    AR: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description=(
            "Accepted-response Probability = ECR · IISR. If ECR=0, AR=0. "
            "If GT has no implicit intent, IISR=1.0 so AR=ECR. "
            "None when either ECR or IISR is None (judge failure)."
        ),
    )
    Eff: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Interaction efficiency = S_median / (S_median + S_actual). "
            "0.5 = on baseline, >0.5 = faster than baseline, <0.5 = slower. "
            "Bounded in (0, 1). Set to null when the GT is invalid for Eff "
            "(e.g. clarification_policy.max_allowed=0); such cases are also "
            "excluded from batch Eff aggregation."
        ),
    )
    SES: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Satisfaction Efficiency Score = AR × Eff. Combines task completion "
            "(AR) with turn efficiency (Eff) into a single bounded score. "
            "0.5 = human baseline (AR=1, Eff=0.5), >0.5 = better than human, "
            "<0.5 = worse than human. Set to null when AR or Eff is null."
        ),
    )
    CEI: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Cost Efficiency Index = (SES / TotalTokens) × 10^6. "
            "每百万 Token 消耗所产出的综合效能，越高性价比越优。"
            "Set to null when SES is null or TotalTokens ≤ 0."
        ),
    )
    e2e_latency_ms: float = Field(
        default=0.0,
        description="Mean end-to-end latency (ms) across assistant turns",
    )

    input_tokens: int = Field(
        default=0,
        description="Session-level sum of prompt/input tokens (across all assistant turns)",
    )
    output_tokens: int = Field(
        default=0,
        description="Session-level sum of completion/output tokens (across all assistant turns)",
    )
    total_tokens: int = Field(
        default=0,
        description=(
            "Sum of input + output tokens for the session. 保留作向后兼容；"
            "新代码请优先用 input_tokens / output_tokens 两个分立字段。"
        ),
    )
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "ECR", "TS", "IFS", "IISR", "AR", "Eff", "SES", "CEI", "e2e_latency_ms",
        mode="after",
    )
    @classmethod
    def _round_to_4dp(cls, v: Optional[float]) -> Optional[float]:
        """所有 float 指标统一 round 到 4 位小数；None 透传（Eff 跳过场景）。"""
        if v is None:
            return None
        return round(float(v), 4)


class VerifyLogEntry(BaseModel):
    """One audit row from the FactVerifier."""

    model_config = ConfigDict(extra="allow")

    statement: str
    source_tool: Optional[str] = None
    ok: Optional[bool] = None
    reason: Optional[str] = None
    api_response: Any = None


# ---------------------------------------------------------------------------
# Meta-Judge audit (Stage 4.5) — pre-metric verdict patching
# ---------------------------------------------------------------------------


CorrectionOp = Literal["set", "add", "remove"]
AttackType = Literal[
    "contradiction",
    "weak_evidence",
    "double_standard",
    "omission_error",
    "hallucinated_verification",
]
Severity = Literal["critical", "major", "minor"]


class Correction(BaseModel):
    """One structured patch emitted by the Meta-Judge.

    The Patcher will only apply corrections whose ``target_path`` matches the
    allowlist defined in ``meta_judge.ALLOWED_PATHS``.
    """

    model_config = ConfigDict(extra="allow")

    correction_id: int = 0
    target_path: str
    operation: CorrectionOp = "set"
    new_value: Any = None
    attack_type: AttackType = "omission_error"
    severity: Severity = "minor"
    reason: str = ""
    evidence_from_source: str = ""

    # Filled by Patcher after attempting to apply.
    applied: bool = False
    apply_error: Optional[str] = None


class MetaJudgeReport(BaseModel):
    """Full Meta-Judge output, before/after Patcher application."""

    model_config = ConfigDict(extra="allow")

    summary: str = ""
    audit_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    corrections: list[Correction] = Field(default_factory=list)

    @property
    def applied_count(self) -> int:
        return sum(1 for c in self.corrections if c.applied)


IISRSource = Literal[
    "standalone",
    "main_judge_fallback",
    "skipped_empty_gt",
    "disabled",
]
IFSSource = Literal[
    "standalone",
    "main_judge_fallback",
    "disabled",
]


class EvalResult(BaseModel):
    """Envelope returned to the caller / persisted to report storage.

    When the Meta-Judge runs, ``verdict`` is the *patched* verdict that
    metrics were computed against (the first-pass / corrections are no
    longer surfaced — only the final verdict + scores).

    ``judge_status`` records per-judge outcome after the 5-way standalone
    fan-out (ecr / ts / ifs / iisr → ``"ok"`` / ``"failed"`` /
    ``"skipped_empty_gt"``) so batch consumers can compute coverage
    without re-parsing logs. Cases never fail because one judge failed —
    the corresponding verdict slot falls back to its schema default and
    metrics still compute (see judge_agent.py fallback table for the
    metric impact of each failure mode).

    ``iisr_source`` / ``ifs_source`` are **deprecated** — they predate
    the 5-way fan-out and are kept one release for downstream Excel
    columns. New consumers should read ``judge_status`` instead.
    """

    case_id: str
    verdict: JudgeVerdict
    scores: MetricScores

    judge_status: dict[str, str] = Field(default_factory=dict)

    iisr_source: IISRSource = "main_judge_fallback"
    ifs_source: IFSSource = "main_judge_fallback"
