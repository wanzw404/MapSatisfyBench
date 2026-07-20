"""LLM-backed explanation for an evaluation result.

Given :class:`MetricScores` and :class:`JudgeVerdict`, ask the LLM to
produce a short, human-readable Chinese explanation answering:

  * 哪些工具 多选 / 漏选 / 不匹配？原因是什么？
  * 每个指标的分数是怎么来的？
  * 还有哪些关键失分点（显式意图、事实不一致、硬约束违反）？

Output is a plain string (markdown-formatted sections). On any LLM
failure we fall back to a deterministic render so the endpoint never
breaks because of the narrative step.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .schema import GroundTruth, JudgeVerdict, MetricScores

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "你是一名评估审核员，需要把一份结构化评估结果翻译成普通业务同学也能看懂的中文复盘。"
    "目标是让读者立刻理解“这条样本好不好、问题出在哪、为什么得分高/低”。\n\n"
    "硬性写作要求：\n"
    "1. 全文使用简体中文，自然流畅，避免学术腔。\n"
    "2. 严禁出现任何数学公式、字典/JSON 片段、变量名（如 N_faithful、Σ(Wi·Ci)、numerator、denominator 等）。\n"
    "3. 指标必须用”中文名（英文缩写）”形式称呼，例如”显式意图完成率（ECR）”，并用一句通俗的话解释这个分值为什么是这样，"
    "例如”3 条显式意图里 1 条都没完成，所以显式意图完成率（ECR）= 0.00”。\n"
    "4. 数字保留两位小数；不要罗列分子/分母原始数字，把含义讲出来即可。\n"
    "5. 没有内容的小节也要保留标题，并写“无”。\n"
    "6. 不复述输入 JSON，不编造证据，只能依据给出的字段。\n\n"
    "输出格式（严格按下面 markdown 结构）：\n"
    "### 总体结论\n"
    "用一句话总评：本次回答是否完成了用户诉求？主要亮点和主要缺陷各是什么？\n\n"
    "### 工具调用情况\n"
    "- 多选：列出多调用的工具，并说明为什么这些调用是多余的；没有则写“无”。\n"
    "- 漏选：列出缺失的工具，并说明少调它会导致哪个用户诉求落空；没有则写“无”。\n"
    "- 不匹配：列出参数错误或用法错误的工具，把错误点用中文讲清楚；没有则写“无”。\n\n"
    "### 各指标得分原因\n"
    "依次解释 显式意图完成率（ECR）、工具命中率（TS）、事实一致率（IFS）、综合达成率（AR = ECR·IISR）、效率（Eff）。"
    "每条**至少 1 行**总评（中文名+缩写+具体得分+一句话解释），其中 IFS 和 AR **必须额外展开子项**（用次级缩进的 bullet）：\n"
    "- 事实一致率（IFS）展开：\n"
    "  · 列出 `facts.faithful` 中的每条忠实事实，说明它对得上哪个 `source_tool` 的返回（写 “X —— 来自 工具Y”）；\n"
    "  · 列出 `facts.unfaithful` 中的每条不忠实事实，写明 statement 原文 + 是 fabricated（凭空）还是 contradicted（与工具矛盾），以及涉及的 `source_tool`（如有）；\n"
    "  · 如果某一类很多（>5 条），可分组聚合（例如“5 条与门店地址相关的捏造”），但**至少要点出代表性 1-2 条原文**，不要只给数字。\n"
    "- 综合达成率（AR）展开：\n"
    "  · 先说 ECR 部分（显式意图完成情况）；\n"
    "  · 再列 `implicit_intents` 中每一条隐式意图，**必须包含以下三层信息**：\n"
    "    a) 一句话总评：rubric 摘要 → 满意度档（完全/部分/弱/未满足/未触发） + 具体 Ci 分数；\n"
    "    b) **子要点拆分**：把 `reasoning` 字段里的 “拆分：- 子要点 A …” 这个三段式（拆分 / 聚合 / 调整）**原样翻译成中文 bullet 列出**——读者要能从你的输出里直接看到每个子要点是命中、方向沾边还是缺失，给了多少分，以及证据原文；\n"
    "    c) 聚合算式（一行）：写出 “(子要点A分 + 子要点B分 + …) / N ≈ Ci”，如果有 “调整” 也跟一句；\n"
    "  · 如果某条 rubric 的 `reasoning` 字段没遵守 “拆分/聚合/调整” 格式（罕见），就照该字段原文复述，并显式标注 “reasoning 未提供子要点拆分，无法解释 Ci 的具体来源”，提醒读者审计；\n"
    "  · 如果存在 `hard_constraint_violations`，单独点出这条把 AR 拉到 0 的原因，引用 reasoning 里的 “违反” 语句作证据。\n\n"
    "### 其他关键失分点\n"
    "把未完成的显式意图、与事实不一致的陈述、被违反的硬约束逐条写出来；没有就写“无”。\n"
)


def _compact_payload(
    scores: MetricScores, verdict: JudgeVerdict, gt: GroundTruth
) -> dict[str, Any]:
    """Distill the verdict/scores into a LLM-friendly compact dict."""
    tcs = verdict.tool_call_summary
    fs = verdict.fact_summary
    exp = verdict.explicit_intent_summary

    # Hard-constraint violations: implicit judgments flagged hard-fail.
    # NOTE: the schema field is ``violates_hard_constraint``; the old
    # ``hard_violation`` key never existed on ImplicitJudgment, so the
    # getattr default silently short-circuited the primary signal and left
    # the narrative relying solely on the Ci/Wi heuristic below.
    hard_violations: list[dict[str, Any]] = []
    for j in verdict.implicit_intent_judgments:
        if getattr(j, "violates_hard_constraint", False) or (
            getattr(j, "triggered", True)
            and getattr(j, "satisfaction_score_Ci", 1.0) < 0.5
            and getattr(j, "confidence_Wi", 0.0) >= 0.6
        ):
            hard_violations.append(
                {
                    "rubric": j.rubric_instruction,
                    "satisfaction": j.satisfaction_score_Ci,
                    "confidence": j.confidence_Wi,
                    "reason": getattr(j, "reasoning", "") or "",
                }
            )

    # Full per-intent snapshot so the narrative can pinpoint unsatisfied
    # implicit intents regardless of whether they're hard or soft.
    implicit_intents: list[dict[str, Any]] = [
        {
            "rubric": j.rubric_instruction,
            "satisfaction": float(getattr(j, "satisfaction_score_Ci", 0.0) or 0.0),
            "confidence": float(getattr(j, "confidence_Wi", 0.0) or 0.0),
            "triggered": bool(getattr(j, "triggered", True)),
            "hard": bool(getattr(j, "violates_hard_constraint", False)),
            "reason": getattr(j, "reasoning", "") or "",
        }
        for j in verdict.implicit_intent_judgments
    ]

    # Strip the raw "formula" key from each metric's details — the LLM is
    # explicitly forbidden from echoing formulas back, and removing it from
    # the input eliminates the temptation entirely.
    cleaned_details: dict[str, Any] = {}
    for key, info in (scores.details or {}).items():
        if isinstance(info, dict):
            cleaned_details[key] = {k: v for k, v in info.items() if k != "formula"}
        else:
            cleaned_details[key] = info

    return {
        "scores": {
            "ECR": scores.ECR,
            "TS": scores.TS,
            "IFS": scores.IFS,
            "AR": scores.AR,
            "Eff": scores.Eff,
            "SES": scores.SES,
            "CEI": scores.CEI,
            "details": cleaned_details,
        },
        "tool_calls": {
            "incorrect": [c.model_dump() for c in tcs.incorrect_calls],
            "extra": list(tcs.extra_calls),
            "missed": list(tcs.missed_calls),
            "correct": [c.tool_name for c in tcs.correct_calls],
            "gold_expected": list(gt.truth_trajectory.tool_calls.expected_tools),
        },
        "explicit_intent": {
            "failed": list(exp.failed_intents),
            "success": list(exp.success_intents),
            "total": exp.total_count,
            "succeeded": exp.success_count,
        },
        "facts": {
            "faithful": [
                {
                    "statement": f.statement,
                    "source_tool": getattr(f, "source_tool", "") or "",
                    "need_verify": bool(getattr(f, "need_verify", False)),
                }
                for f in fs.faithful_facts
            ],
            "unfaithful": [
                {
                    "statement": f.statement,
                    "reason": getattr(f, "reason", "") or "",
                    "verify_reason": getattr(f, "verify_reason", "") or "",
                    "source_tool": getattr(f, "source_tool", "") or "",
                }
                for f in fs.unfaithful_facts
            ],
            "faithful_count": len(fs.faithful_facts),
        },
        "hard_constraint_violations": hard_violations,
        "implicit_intents": implicit_intents,
        "conversation_turns": verdict.conversation_turn_count,
    }


_METRIC_LABEL = {
    "ECR": "显式意图完成率（ECR）",
    "TS": "工具命中率（TS）",
    "IFS": "事实一致率（IFS）",
    "AR": "综合达成率（AR = ECR · IISR）",
    "Eff": "效率（Eff）",
    "CEI": "成本效率指数（CEI）",
}


def _describe_metric(key: str, value: float, payload: dict[str, Any]) -> str:
    """One-line plain-Chinese reason for a single metric, no formulas/JSON."""
    label = _METRIC_LABEL.get(key, key)
    val_txt = f"{float(value):.2f}" if isinstance(value, (int, float)) else str(value)
    ei = payload.get("explicit_intent", {})
    tc = payload.get("tool_calls", {})
    facts = payload.get("facts", {})
    hard = payload.get("hard_constraint_violations", [])

    if key == "ECR":
        total = ei.get("total", 0)
        ok = ei.get("succeeded", 0)
        failed_list = ei.get("failed", []) or []
        if total <= 0:
            return f"- {label} = {val_txt}：本案例没有显式意图可考核。"
        if failed_list:
            names = "、".join(str(x)[:40] for x in failed_list if x)
            return (
                f"- {label} = {val_txt}：共 {total} 条显式意图，"
                f"其中 {len(failed_list)} 条未达成（{names or '详见 explicit_intent'}）。"
            )
        return (
            f"- {label} = {val_txt}：{total} 条显式意图全部达成。"
        )
    if key == "TS":
        gold = tc.get("gold_expected") or []
        correct = tc.get("correct") or []
        missed = tc.get("missed") or []
        extra = tc.get("extra") or []
        wrong = [c.get("tool_name", "") for c in (tc.get("incorrect") or [])]
        bits = []
        if missed:
            bits.append(f"漏掉 {'、'.join(missed)}")
        if wrong:
            bits.append(f"调用方式有误的有 {'、'.join(wrong)}")
        if extra:
            bits.append(f"多调了 {'、'.join(extra)}")
        if not bits:
            bits.append(
                f"命中 {len(correct)} 个，与期望工具 {('、'.join(gold)) or '无'} 完全吻合"
            )
        return f"- {label} = {val_txt}：" + "；".join(bits) + "。"
    if key == "IFS":
        un = facts.get("unfaithful", []) or []
        ok = facts.get("faithful_count", 0)
        total = len(un) + int(ok or 0)
        if total == 0:
            return f"- {label} = {val_txt}：本案例没有需要核实的事实陈述。"
        if not un:
            return f"- {label} = {val_txt}：所有 {ok} 条事实陈述都通过了核验。"
        # Build per-claim bullets: "陈述 —— 与工具返回不符的原因"
        bullets: list[str] = []
        _enum_label = {
            "fabricated": "凭空编造",
            "contradicted": "与工具返回矛盾",
            "verified_wrong": "外部核查判定为假",
        }
        for f in un:
            stmt = (f.get("statement") or "").strip()
            if not stmt:
                continue
            # Prefer the free-text explanation from the fact verifier; fall
            # back to the short reason enum; last resort is a generic line.
            why_txt = (f.get("verify_reason") or "").strip()
            if not why_txt:
                enum_code = (f.get("reason") or "").strip()
                why_txt = _enum_label.get(enum_code, enum_code) or "与工具返回结果不一致"
            src = (f.get("source_tool") or "").strip()
            src_tag = f"（来源工具：{src}）" if src else ""
            bullets.append(f"“{stmt[:80]}”{src_tag} —— {why_txt[:120]}")
        detail = "；".join(bullets) if bullets else "详见 facts.unfaithful"
        return (
            f"- {label} = {val_txt}：共 {total} 条事实陈述，"
            f"其中 {len(un)} 条与工具返回不一致（{detail}）。"
        )
    if key == "AR":
        intents = payload.get("implicit_intents", []) or []
        ar_details = (
            (payload.get("scores", {}).get("details", {}) or {}).get("AR", {})
            or {}
        )
        ecr_c = float(ar_details.get("ecr_component") or ar_details.get("icr_component") or 0.0)
        iisr_c = float(ar_details.get("iisr_component", 0.0))
        iisr_breakdown = ar_details.get("iisr_breakdown", {}) or {}
        # ECR=0 → AR=0（乘法直接归零）
        if ecr_c <= 0.0 or iisr_breakdown.get("explicit_intents_unmet"):
            return (
                f"- {label} = {val_txt}：显式意图未达成（ECR≈{ecr_c:.2f}），"
                "AR = ECR · IISR 被乘法归零。"
            )
        if hard or iisr_breakdown.get("hard_constraint_violated"):
            hints = "、".join(hv.get("rubric", "") for hv in hard if hv.get("rubric"))
            return (
                f"- {label} = {val_txt}：存在硬约束违反（{hints or '未具体列出'}），"
                "IISR 项归零，AR 跟着归零。"
            )
        if not intents:
            return (
                f"- {label} = {val_txt}：本案例没有隐式意图，IISR=1，"
                f"AR = ECR ≈ {ecr_c:.2f}。"
            )
        failed = [i for i in intents if i.get("satisfaction", 1.0) < 0.5]
        partial = [
            i for i in intents if 0.5 <= i.get("satisfaction", 1.0) < 1.0
        ]
        if failed:
            names = "、".join(
                (i.get("rubric") or "")[:40] for i in failed if i.get("rubric")
            )
            return (
                f"- {label} = {val_txt}：ECR≈{ecr_c:.2f}, IISR≈{iisr_c:.2f}；"
                f"{len(intents)} 条隐式意图中有 {len(failed)} 条未满足"
                f"（{names or '详见 implicit_intent_judgments'}）。"
            )
        if partial:
            return (
                f"- {label} = {val_txt}：ECR≈{ecr_c:.2f}, IISR≈{iisr_c:.2f}；"
                f"{len(intents)} 条隐式意图中有 {len(partial)} 条只部分满足。"
            )
        return (
            f"- {label} = {val_txt}：ECR≈{ecr_c:.2f}, IISR≈{iisr_c:.2f}；"
            f"{len(intents)} 条隐式意图全部较好满足。"
        )
    if key == "Eff":
        # Prefer the turn count from the metric details (authoritative), fall
        # back to the verdict-level count.
        eff_details = (payload.get("scores", {}).get("details", {}) or {}).get(
            "Eff", {}
        ) or {}
        turns = eff_details.get("actual_turns")
        if turns is None:
            turns = payload.get("conversation_turns", 0)
        median = eff_details.get("human_median")
        if median is not None:
            return (
                f"- {label} = {val_txt}：本次共进行 {turns} 轮对话，"
                f"人类基线约 {median} 轮；0 表示贴合基线，负数为提前完成，正数为多出轮次。"
            )
        return (
            f"- {label} = {val_txt}：本次共进行 {turns} 轮对话，"
            "0 表示贴合人类基线，负数为提前完成，正数为多出轮次。"
        )
    return f"- {label} = {val_txt}。"


def _fallback_render(payload: dict[str, Any]) -> str:
    """Deterministic, formula-free Chinese fallback if the LLM call fails."""
    tc = payload["tool_calls"]
    ei = payload["explicit_intent"]
    facts = payload["facts"]
    hard = payload["hard_constraint_violations"]
    scores = payload["scores"]

    # Overall conclusion (one-liner)
    overall = float(scores.get("ECR", 0.0))
    if ei.get("total", 0) > 0 and ei.get("succeeded", 0) == 0:
        verdict_line = "本次回答未能完成用户的核心诉求，整体表现较差。"
    elif overall >= 0.8 and not facts.get("unfaithful") and not hard:
        verdict_line = "本次回答总体达成了用户诉求，仅有少量瑕疵。"
    else:
        verdict_line = "本次回答部分达成了用户诉求，但仍存在明显问题。"

    lines: list[str] = ["### 总体结论", verdict_line, ""]

    lines.append("### 工具调用情况")
    lines.append(
        "- 多选：" + ("、".join(tc["extra"]) if tc["extra"] else "无")
    )
    lines.append(
        "- 漏选：" + ("、".join(tc["missed"]) if tc["missed"] else "无")
    )
    if tc["incorrect"]:
        for c in tc["incorrect"]:
            reason = c.get("reason", "") or "调用方式错误"
            lines.append(f"- 不匹配：{c['tool_name']} —— {reason}")
    else:
        lines.append("- 不匹配：无")

    lines.append("")
    lines.append("### 各指标得分原因")
    for key in ("ECR", "TS", "IFS", "AR", "Eff", "SES", "CEI"):
        lines.append(_describe_metric(key, scores.get(key, 0.0), payload))

    lines.append("")
    lines.append("### 其他关键失分点")
    extras: list[str] = []
    for f in ei.get("failed", []) or []:
        extras.append(f"- 未达成的显式意图：{f}")
    for f in facts.get("unfaithful", []) or []:
        statement = f.get("statement", "")
        why = (f.get("verify_reason") or "").strip()
        if not why:
            enum_code = (f.get("reason") or "").strip()
            why = {
                "fabricated": "凭空编造",
                "contradicted": "与工具返回矛盾",
                "verified_wrong": "外部核查判定为假",
            }.get(enum_code, enum_code) or "与工具返回结果不一致"
        src = (f.get("source_tool") or "").strip()
        src_tag = f"（来源工具：{src}）" if src else ""
        extras.append(f"- 事实不一致：{statement}{src_tag} —— {why}")
    for hv in hard or []:
        rubric = hv.get("rubric", "")
        reason = hv.get("reason", "") or "未满足硬性要求"
        extras.append(f"- 硬约束违反：{rubric} —— {reason}")
    if extras:
        lines.extend(extras)
    else:
        lines.append("- 无")

    return "\n".join(lines)


async def generate_explanation(
    scores: MetricScores,
    verdict: JudgeVerdict,
    gt: GroundTruth,
    llm_provider: Any,
    *,
    temperature: float | None = None,
) -> str:
    """Ask the LLM for a Chinese narrative explanation of the evaluation.

    Falls back to a deterministic render on any LLM error so callers
    never lose the explanation field.
    """
    payload = _compact_payload(scores, verdict, gt)
    user_msg = (
        "以下是一个评估案例的裁决与分数，请据此输出要求的解释：\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```"
    )

    try:
        resp = await llm_provider.achat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
        )
        content = (resp.content or "").strip()
        if not content:
            raise ValueError("empty LLM response")
        return content
    except Exception as exc:  # noqa: BLE001
        logger.warning("explanation LLM call failed, falling back: %s", exc)
        return _fallback_render(payload)
