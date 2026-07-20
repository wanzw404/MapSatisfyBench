"""Ensemble (multi-model **per-metric** consensus) judge.

Calls the same JudgeAgent pipeline through **N base models** independently.
Default N=3 with model IDs::

    gemini-2.5-pro-06-17
    claude-opus-4-7
    gpt-5.5-0424-global

(参见下面 ``MODEL_GEMINI`` / ``MODEL_CLAUDE`` / ``MODEL_GPT`` 常量及说明。)

聚合规则（**新版：逐指标聚合**）::

    Round 1 — 3 模型并发跑 → 收集每个模型的 6 维 metrics
        ↓
    对每一维（ECR / TS / IFS / AR / Eff）独立判定：
      • 找最大的"两两差 ≤ tolerance"集合（默认 0.1）
      • 集合大小 ≥ 2 → 取该集合数值的均值作为该维的 consensus
      • 集合大小 < 2 → 该维走 fallback_model 的值
        ↓
    Round 1 6 维都 cluster 成功 ⇒ 提前结束；否则进 Round 2 收集更多样本
        ↓
    Round 2 跑完后再做一次 6 维聚合（用 round1 + round2 的全部 6 个样本）
        ↓
    最终 scores = 6 维 consensus 拼成的 MetricScores
    最终 verdict = fallback_model 那次成功 run 的 verdict（用于细节 / details / reason）
    最终 reason = 用 fallback_model 跑 explainer 重新生成（基于上面的 consensus scores）

为什么这样改：旧的"6 维 signature 完全一致"过严——claude IFS=0.60 / gpt IFS=0.67
差距很小（<0.1）实质应判一致；但其中 AR 又彻底分歧。逐维聚合能把"能达成共识
的维度"和"真分歧的维度"分开处理，AR 退到 fallback 但 IFS 仍用 0.6+0.67 的均值。

Concurrency: 同一轮的 3 个模型用 ``asyncio.gather`` 并发执行；不同轮顺序执行
（前一轮已 6 维全 cluster 就提前结束，省一轮 LLM 调用）。

Caller 视角：
    >>> resp, audit = await ensemble_evaluate(req)
    # resp 跟普通 evaluate 输出格式完全相同（EvaluateResponse）
    # audit.consensus.per_metric_audit 是核心，能看到每一维是 cluster 均值
    # 还是 fallback，以及参与 cluster 的模型 + 它们的原始数值
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import HTTPException

from app.core.evaluation.explainer import generate_explanation
from app.core.evaluation.schema import EvalResult, GroundTruth, MetricScores
from app.schemas.evaluate_schemas import EvaluateRequest, EvaluateResponse
from app.services.evaluate_service import (
    _build_llm_provider,
    _build_scores,
)
from app.core.evaluation import evaluate_case as run_single_case

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 默认模型清单 + 兜底
# ---------------------------------------------------------------------------
#
# 模型 ID 来源：
#   * GPT-5.5 → `gpt-5.5-0424-global`        来自 .env 里的注释行（已验证）
#   * Claude  → `claude-opus-4-7`            Anthropic 官方 Opus 4.7 模型 ID
#   * Gemini  → `gemini-2.5-pro-06-17`       实际可用 ID（用户指定）
#
# 如果某个模型的真实 ID 跟这里不一致，运行时会从网关返回 404 /
# "model not found" 错误。三种修正方式：
#   1. 直接改本文件下面的 ``MODEL_*`` 常量；
#   2. 在 CLI 用 ``--models`` 覆盖：
#        scripts/run_ensemble_judge.py --models "<a>,<b>,<c>"
#   3. 调 ``ensemble_evaluate(..., models=("...",))`` 时显式传入。

MODEL_GEMINI = "gemini-2.5-pro-06-17"
MODEL_CLAUDE = "claude-opus-4-7"
MODEL_GPT    = "gpt-5.5-0424-global"

DEFAULT_MODELS: tuple[str, ...] = (
    MODEL_GEMINI,    # 兜底优先级最高（顺序也决定多数判定时的 winner 优先级）
    MODEL_CLAUDE,
    MODEL_GPT,
)
DEFAULT_FALLBACK_MODEL = MODEL_GEMINI
DEFAULT_ROUNDS = 2


# ---------------------------------------------------------------------------
# 内部数据结构
# ---------------------------------------------------------------------------


@dataclass
class _JudgeRun:
    """单个模型 / 单轮的判定输出。"""

    model: str
    round_idx: int
    result: Optional[EvalResult] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.result is not None


DEFAULT_TOLERANCE = 0.1
"""逐指标"两两差 ≤ tolerance"算同一 cluster。用户口径：差 < 0.1 视为一致。"""

METRIC_NAMES: tuple[str, ...] = ("ECR", "TS", "IFS", "AR", "Eff", "SES", "CEI")


def _cluster_values(values: list[float], tol: float) -> tuple[list[int], float]:
    """找最大的"互相在 tol 内"的子集；返回 (索引列表, 该子集均值)。

    实现：以每个值为锚，找所有跟它差 ≤ tol 的值，记下最大的那一组。
    O(N²)，N=3 / 6 完全够用。

    注意：这里"互相在 tol 内"是宽松定义——锚点 i 的邻居 j 满足 |i-j|≤tol，
    但 j 的邻居不一定都在锚点 i 的 tol 内。对小规模 N 这种简化够用，避免
    严格意义"图染色"的复杂度。
    """
    if not values:
        return [], 0.0
    n = len(values)
    best: list[int] = []
    for i in range(n):
        cluster = [i]
        for j in range(n):
            if i == j:
                continue
            if abs(float(values[i]) - float(values[j])) <= tol:
                cluster.append(j)
        if len(cluster) > len(best):
            best = cluster
    if not best:
        return [], 0.0
    mean_v = sum(float(values[i]) for i in best) / len(best)
    return best, mean_v


def _per_metric_consensus(
    runs: list["_JudgeRun"],
    tolerance: float,
    fallback_model: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], Optional["_JudgeRun"]]:
    """每个指标独立做 cluster + 均值；找不到就走 fallback 模型的值。

    返回 ``(consensus_dict, per_metric_audit, fallback_run)``：
      * ``consensus_dict[m]`` 是该指标的最终值（float 或 None）
      * ``per_metric_audit`` 是每个指标的审计列表（含 method / values / cluster）
      * ``fallback_run`` 是用来兜底的 _JudgeRun，未来用于 verdict / details
    """
    valid_runs = [r for r in runs if r.ok and r.result is not None]
    if not valid_runs:
        return {}, [], None

    # fallback_run：优先 fallback_model 最近一次成功的 run，否则第一个成功 run
    fallback_run: Optional[_JudgeRun] = None
    for r in reversed(valid_runs):
        if r.model == fallback_model:
            fallback_run = r
            break
    if fallback_run is None:
        fallback_run = valid_runs[0]

    consensus: dict[str, Any] = {}
    per_metric_audit: list[dict[str, Any]] = []

    for m in METRIC_NAMES:
        # 收集所有 run 的 (model, round, value)；None 单独保留以处理 Eff 跳过场景
        items = [
            {"model": r.model, "round": r.round_idx, "value": getattr(r.result.scores, m)}
            for r in valid_runs
        ]
        non_none = [it for it in items if it["value"] is not None]
        none_count = len(items) - len(non_none)

        # 全 None：consensus = None（典型：max_allowed=0 时 Eff）
        if not non_none:
            consensus[m] = None
            per_metric_audit.append({
                "metric": m, "method": "all_none",
                "values": items, "consensus": None,
            })
            continue

        # 多数 None：consensus 也判 None
        if none_count > len(non_none):
            consensus[m] = None
            per_metric_audit.append({
                "metric": m, "method": "majority_none",
                "values": items, "consensus": None,
            })
            continue

        # 数值聚类
        nums = [float(it["value"]) for it in non_none]
        cluster_idx, cluster_mean = _cluster_values(nums, tolerance)

        if len(cluster_idx) >= 2:
            # 走 cluster 均值
            cluster_members = [non_none[i] for i in cluster_idx]
            mean_v = round(cluster_mean, 4)
            consensus[m] = mean_v
            per_metric_audit.append({
                "metric": m, "method": "cluster_mean",
                "tolerance": tolerance,
                "values": items,
                "cluster_members": cluster_members,
                "cluster_size": len(cluster_idx),
                "consensus": mean_v,
            })
        else:
            # 找不到 cluster → fallback model 的值
            fb_val = None
            for it in reversed(non_none):
                if it["model"] == fallback_model:
                    fb_val = it["value"]
                    break
            if fb_val is None:
                fb_val = non_none[-1]["value"]   # fallback 也没成功值，取任意
            consensus[m] = (
                round(float(fb_val), 4) if isinstance(fb_val, (int, float)) else fb_val
            )
            per_metric_audit.append({
                "metric": m, "method": "fallback_no_cluster",
                "tolerance": tolerance,
                "values": items,
                "fallback_model": fallback_model,
                "consensus": consensus[m],
            })

    return consensus, per_metric_audit, fallback_run


def _all_metrics_clustered(per_metric_audit: list[dict[str, Any]]) -> bool:
    """6 个指标是不是全都达成 cluster（无 fallback / no_cluster）。

    用于决定 round 1 跑完后是否提前结束（不进 round 2）。
    None 类（all_none / majority_none）视为"达成共识"，因为它们没分歧。
    """
    accepted = {"cluster_mean", "all_none", "majority_none"}
    return all(a["method"] in accepted for a in per_metric_audit)


# ---------------------------------------------------------------------------
# 单模型一次评测——封装 _build_llm_provider + run_single_case
# ---------------------------------------------------------------------------


def _stamp() -> str:
    """简短时间戳，毫秒级，用于进度日志。"""
    return time.strftime("%H:%M:%S")


def _progress(msg: str) -> None:
    """统一进度打印——直接走 stdout 并 flush，避免被缓冲掉。"""
    print(f"[{_stamp()}] {msg}", flush=True)


async def _judge_with_model(
    model: str,
    round_idx: int,
    case_dict: dict[str, Any],
    case_id: str,
    *,
    language: str,
    enable_verification: bool,
    enable_meta_judge: bool,
) -> _JudgeRun:
    """跑单个模型的判定，捕获所有异常，返回 _JudgeRun。

    每个模型起跑/回来都打印进度，方便观察是哪个模型慢 / 哪个先回来。
    """
    short_cid = case_id[:12] if case_id else "?"
    _progress(f"[case={short_cid}][round={round_idx}][{model}] ▶ start")
    t0 = time.monotonic()
    try:
        provider = _build_llm_provider(model_override=model)
        result: EvalResult = await run_single_case(
            case_dict,
            llm_provider=provider,
            language=language,
            enable_verification=enable_verification,
            enable_meta_judge=enable_meta_judge,
        )
        elapsed = time.monotonic() - t0
        s = result.scores
        _progress(
            f"[case={short_cid}][round={round_idx}][{model}] ✓ done in {elapsed:.1f}s "
            f"| ECR={s.ECR} TS={s.TS} IFS={s.IFS} AR={s.AR} Eff={s.Eff} SES={s.SES} CEI={s.CEI}"
        )
        return _JudgeRun(model=model, round_idx=round_idx, result=result)
    except HTTPException as he:
        elapsed = time.monotonic() - t0
        msg = f"HTTP {he.status_code}: {he.detail}"
        _progress(
            f"[case={short_cid}][round={round_idx}][{model}] ✗ FAILED in {elapsed:.1f}s "
            f"| {msg[:200]}"
        )
        logger.warning(f"[ensemble][{model}][round={round_idx}] {msg}")
        return _JudgeRun(model=model, round_idx=round_idx, error=msg)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        _progress(
            f"[case={short_cid}][round={round_idx}][{model}] ✗ FAILED in {elapsed:.1f}s "
            f"| {type(exc).__name__}: {exc}"
        )
        logger.exception(f"[ensemble][{model}][round={round_idx}] failed")
        return _JudgeRun(
            model=model, round_idx=round_idx, error=f"{type(exc).__name__}: {exc}"
        )


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------


async def ensemble_evaluate(
    req: EvaluateRequest,
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    rounds: int = DEFAULT_ROUNDS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[EvaluateResponse, dict[str, Any]]:
    """3 模型 + 逐指标聚合的集成判官（**新版**：per-metric consensus）。

    返回 (canonical_response, audit_log)：
      * canonical_response — 跟普通 ``run_evaluation`` 同 schema 的
        ``EvaluateResponse``（``case_id`` / ``results`` / ``reason``）。
        ``results.metrics`` 里 6 维分数是逐维聚合后的 consensus。
      * audit_log — 含每轮每模型的原始分数 + per-metric consensus 决策过程。
    """
    # 校验 ground_truth（跟 evaluate_service.run_evaluation 同一套）
    try:
        gt = GroundTruth.model_validate(req.ground_truth)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"ground_truth validation failed: {exc}"
        ) from exc

    history = [m.model_dump() for m in req.conversation_history_messages]
    case_dict: dict[str, Any] = {
        "case_id": req.case_id,
        "query": req.query,
        "full_intent": req.full_intent,
        "current_time": req.current_time,
        "current_location": req.current_location,
        "conversation_history_messages": history,
        "ground_truth": gt,
        "tools_schema": req.tools_schema,
        "session_stats": req.session_stats.model_dump() if req.session_stats else None,
    }

    audit: dict[str, Any] = {
        "models": list(models),
        "fallback_model": fallback_model,
        "tolerance": tolerance,
        "rounds_planned": rounds,
        "rounds_executed": 0,
        "rounds": [],
        "consensus": None,        # 填到最后
        "verdict_source_model": None,
        "verdict_source_round": None,
    }

    all_runs: list[_JudgeRun] = []
    short_cid = req.case_id[:12] if req.case_id else "?"

    for round_idx in range(1, rounds + 1):
        _progress(
            f"[case={short_cid}] === Round {round_idx}/{rounds}：并发启动 "
            f"{len(models)} 个模型 ==="
        )
        round_t0 = time.monotonic()
        round_runs: list[_JudgeRun] = await asyncio.gather(*[
            _judge_with_model(
                m, round_idx, case_dict,
                req.case_id or "",
                language=req.language,
                enable_verification=req.enable_verification,
                enable_meta_judge=req.enable_meta_judge,
            )
            for m in models
        ])
        round_elapsed = time.monotonic() - round_t0
        all_runs.extend(round_runs)
        audit["rounds_executed"] = round_idx
        audit["rounds"].append({
            "round": round_idx,
            "elapsed_s": round(round_elapsed, 1),
            "scores_per_model": [
                {
                    "model": r.model,
                    "ok": r.ok,
                    "error": r.error,
                    "metrics": (
                        r.result.scores.model_dump(exclude={"details"})
                        if r.ok else None
                    ),
                }
                for r in round_runs
            ],
        })

        # 早停判断：仅根据本轮 3 个 run 做 per-metric 聚合，全 cluster 即结束
        per_round_consensus, per_round_audit, _ = _per_metric_consensus(
            round_runs, tolerance, fallback_model,
        )
        if _all_metrics_clustered(per_round_audit):
            _progress(
                f"[case={short_cid}] Round {round_idx} 结束 ({round_elapsed:.1f}s) → "
                f"✓ 6 维全部达成 cluster，提前结束"
            )
            break
        else:
            no_cluster_metrics = [
                a["metric"] for a in per_round_audit if a["method"] == "fallback_no_cluster"
            ]
            _progress(
                f"[case={short_cid}] Round {round_idx} 结束 ({round_elapsed:.1f}s) → "
                f"⚠ 未全 cluster（待解决：{no_cluster_metrics}）"
                + (f"，进 Round {round_idx+1}" if round_idx < rounds else "，已是最后一轮")
            )

    # 跨轮 / 最终聚合：用所有 round 的所有成功 run
    consensus, per_metric_audit, fallback_run = _per_metric_consensus(
        all_runs, tolerance, fallback_model,
    )

    if not consensus or fallback_run is None:
        _progress(f"[case={short_cid}] ✗✗ 全部模型全部轮次都失败")
        raise HTTPException(
            status_code=502,
            detail=(
                "All ensemble judges failed in all rounds. "
                f"Errors: {[r.error for r in all_runs if r.error]}"
            ),
        )

    # 用 fallback_run 的 verdict 作为 details / reason 的素材源；分数走 consensus。
    # IISR 不在本模块的 METRIC_NAMES（共识 6 维不含 IISR），直接用 fallback_run
    # 的 IISR 值兜底——否则 MetricScores 校验缺必填字段会炸。
    consensus_scores = MetricScores(
        ECR=consensus["ECR"],
        TS=consensus["TS"],
        IFS=consensus["IFS"],
        IISR=fallback_run.result.scores.IISR,
        AR=consensus["AR"],
        Eff=consensus["Eff"],
        SES=consensus["SES"],
        CEI=consensus["CEI"],
        e2e_latency_ms=fallback_run.result.scores.e2e_latency_ms,
        total_tokens=fallback_run.result.scores.total_tokens,
        details=fallback_run.result.scores.details,
    )

    # 各维 consensus 是哪种来源，做个简短摘要
    method_summary = {
        a["metric"]: a["method"] for a in per_metric_audit
    }
    audit["consensus"] = {
        "method": "per-metric clustering with tolerance",
        "tolerance": tolerance,
        "scores": consensus_scores.model_dump(exclude={"details"}),
        "method_per_metric": method_summary,
        "per_metric_audit": per_metric_audit,
    }
    audit["verdict_source_model"] = fallback_run.model
    audit["verdict_source_round"] = fallback_run.round_idx

    n_cluster = sum(1 for v in method_summary.values() if v == "cluster_mean")
    n_fallback = sum(1 for v in method_summary.values() if v == "fallback_no_cluster")
    n_none = sum(1 for v in method_summary.values() if v in ("all_none", "majority_none"))
    _progress(
        f"[case={short_cid}] 聚合完成：{n_cluster} 维 cluster 均值, "
        f"{n_fallback} 维走 fallback ({fallback_model}), {n_none} 维 None。"
        f" verdict 源 = {fallback_run.model}（round {fallback_run.round_idx}）"
    )

    # 用 fallback_model 调 explainer 生成中文 reason，scores 用 consensus
    _progress(
        f"[case={short_cid}] 生成 reason 文字（用 {fallback_run.model} 调 explainer）..."
    )
    expl_t0 = time.monotonic()
    try:
        explanation = await generate_explanation(
            consensus_scores, fallback_run.result.verdict, gt,
            _build_llm_provider(model_override=fallback_run.model),
        )
        _progress(
            f"[case={short_cid}] reason 生成完毕（{time.monotonic()-expl_t0:.1f}s, "
            f"{len(explanation)} 字符）"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[ensemble] explanation generation failed: {exc}")
        _progress(
            f"[case={short_cid}] ⚠ reason 生成失败 ({time.monotonic()-expl_t0:.1f}s)：{exc}"
        )
        explanation = ""

    response = EvaluateResponse(
        case_id=req.case_id,
        results=_build_scores(consensus_scores),
        reason=explanation,
    )
    return response, audit
