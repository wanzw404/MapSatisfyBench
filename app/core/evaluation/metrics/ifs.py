"""IFS — Information Faithfulness Score (rubric-row based, all-or-nothing).

公式::

    IFS = N_faithful_rows / N_total_rows

  * N_total_rows = ``ground_truth.truth_trajectory.tool_calls.factual_answer_rubric``
                  的行数。
  * N_faithful_rows = "行内所有**被声称**的要素都有 tool 依据"的行数。
  * 行级判定（absent 要素被忽略后 all-or-nothing）：
      - 把 ``reason=absent`` 的要素（content 根本没声称这件事）+
        ``skipped=True`` 的要素（外部验证不可达）从该行 element 列表里**剔除**；
      - 剔完后剩余要素全部 ``grounded=True`` → 行 = 1；
      - 剔完后没剩任何要素（全 absent / 全 skipped）→ 行 = 1（空满足）；
      - 剔完后存在 ``grounded=False``（no_tool_grounding / contradicted /
        external_verify_failed）→ 行 = 0。

  本质：IFS 衡量"答案声称的事实是否有依据"。没声称的事实**不进入**
  faithfulness 检查——既不算分子也不算扣分项。这跟"用户没问的需求 ECR
  不扣分"是同一种 vacuous truth。

兼容老 verdict（无 ``rubric_row_judgments``）：自动回退到旧逻辑——把
``fact_summary.faithful_facts`` / ``unfaithful_facts`` 当成 IFS 的分子分母。

边界值：
  * factual_answer_rubric 为空（GT 没列任何行）→ IFS = 1.0
    （vacuously satisfied，跟 IISR 无 implicit_intent 时返回 1.0 一致）
  * GT 有行但 verdict.rubric_row_judgments 缺失 → 走 fallback。
"""

from __future__ import annotations

from typing import Any

from app.core.evaluation.schema import (
    FactStatement,
    GroundTruth,
    JudgeVerdict,
    RubricRowJudgment,
)


def _is_unfaithful(fact: FactStatement) -> bool:
    if fact.faithful is False:
        return True
    if fact.need_verify and fact.verified_ok is False:
        return True
    return False


def _row_is_faithful(row: RubricRowJudgment) -> bool:
    """行级判定：absent / skipped 要素先剔除，剩余全 grounded 才算 1。

    规则::

        过滤掉 reason=absent（content 未声称）+ skipped=True（验证不可达）
        ├─ 剔完为空            → 1（空满足，没有被声称的事实可查证）
        ├─ 剩余全 grounded=True → 1
        └─ 剩余存在 grounded=False（no_tool_grounding / contradicted /
            external_verify_failed）→ 0

    **关键**：absent 要素**不进** all-or-nothing 判定。IFS 衡量的是
    "声称的事实是否有依据"——没声称的事实没有可查证之处，应该忽略，
    不能用来卡 row 失败。

    **不读 LLM 写的 ``row.score``**——LLM 容易把"答非所问"（维度 1
    ECR 的扣分点）误算到 IFS：明明 element 标了 absent，行 score 还
    被写成 0。代码从 element 级证据 deterministic 推导能彻底堵掉这种
    跨维度污染。

    Legacy fallback：``row.elements`` 为空时退回信任 ``row.score``，
    保兼容老 verdict。
    """
    if not row.elements:
        return row.score == 1

    # 剔除"未声称"和"验证不可达"的要素；剩下的就是"声称了的事实"。
    relevant = [
        e for e in row.elements
        if not e.skipped and e.reason != "absent"
    ]
    if not relevant:
        return True  # 空满足：没有任何被声称的事实
    return all(e.grounded for e in relevant)


def _row_is_skipped(row: RubricRowJudgment) -> bool:
    """整行从 IFS 中忽略：行级 skipped=True，或行内所有要素都 skipped。"""
    if getattr(row, "skipped", False):
        return True
    if row.elements and all(e.skipped for e in row.elements):
        return True
    return False


def _has_rubric_judgments(verdict: JudgeVerdict) -> bool:
    """verdict 是否走了新版 rubric 行级判定。"""
    return bool(getattr(verdict, "rubric_row_judgments", None))


def compute_ifs(verdict: JudgeVerdict, gt: GroundTruth) -> float:
    # ── 新路径：按 factual_answer_rubric 行级判定 ──────────────────────
    gt_rows = gt.truth_trajectory.tool_calls.factual_answer_rubric or []
    if not gt_rows:
        # GT 完全没列 rubric 行 → 无要求 → 自然满足
        return 1.0

    if _has_rubric_judgments(verdict):
        rows = verdict.rubric_row_judgments
        # 跳过外部验证不可达的行（不进 IFS 分母分子）
        valid_rows = [r for r in rows if not _row_is_skipped(r)]
        n_total = len(valid_rows)
        if n_total == 0:
            # 全部行都 skipped（外部验证全挂）→ 视为无要求，给满分
            return 1.0
        n_faithful = sum(1 for r in valid_rows if _row_is_faithful(r))
        return n_faithful / n_total

    # ── 兼容路径：老 verdict 没填 rubric_row_judgments，退回 fact 统计 ─
    fs = verdict.fact_summary
    flipped = sum(1 for f in fs.faithful_facts if _is_unfaithful(f))
    unfaithful_count = len(fs.unfaithful_facts) + flipped
    faithful_count = max(0, len(fs.faithful_facts) - flipped)
    n_total = faithful_count + unfaithful_count
    if n_total == 0:
        return 0.0
    return faithful_count / n_total


def _fact_to_dict(f: FactStatement, force_unfaithful: bool = False) -> dict[str, Any]:
    """把 FactStatement 转成可序列化 dict，带审计需要的全部字段。"""
    d: dict[str, Any] = {
        "statement": f.statement,
        "source_tool": getattr(f, "source_tool", "") or "",
        "need_verify": bool(getattr(f, "need_verify", False)),
        "verified_ok": getattr(f, "verified_ok", None),
    }
    reason = getattr(f, "reason", None)
    if reason:
        d["reason"] = reason
    elif force_unfaithful:
        # 来自外部验证器把 faithful 翻成 unfaithful 的情况，没填 reason
        d["reason"] = "verifier_flipped"
    vr = getattr(f, "verify_reason", None)
    if vr:
        d["verify_reason"] = vr
    return d


def details_ifs(verdict: JudgeVerdict, gt: GroundTruth) -> dict[str, Any]:
    gt_rows = gt.truth_trajectory.tool_calls.factual_answer_rubric or []

    # ── 新路径：行级判定 ───────────────────────────────────────────────
    if _has_rubric_judgments(verdict):
        rows = verdict.rubric_row_judgments
        n_total_raw = len(rows)
        skipped_rows = [r for r in rows if _row_is_skipped(r)]
        valid_rows = [r for r in rows if not _row_is_skipped(r)]
        n_total = len(valid_rows)
        n_faithful = sum(1 for r in valid_rows if _row_is_faithful(r))
        n_unfaithful = n_total - n_faithful
        n_skipped = len(skipped_rows)

        rows_detail = []
        for r in rows:
            elements_dump = [e.model_dump() for e in r.elements]
            grounded = [
                e for e in elements_dump
                if e.get("grounded") and not e.get("skipped")
            ]
            absent = [
                e for e in elements_dump
                if (not e.get("grounded")) and (not e.get("skipped"))
                and e.get("reason") == "absent"
            ]
            unfaithful = [
                e for e in elements_dump
                if (not e.get("grounded")) and (not e.get("skipped"))
                and e.get("reason") != "absent"
            ]
            skipped_elems = [e for e in elements_dump if e.get("skipped")]
            row_skipped = _row_is_skipped(r)
            row_faithful = False if row_skipped else _row_is_faithful(r)
            # `score` 输出 deterministic 值（与 is_faithful 对齐），LLM 原始
            # `row.score` 另存 `llm_score_raw` 供审计跨维度污染。
            # absent / unfaithful 拆开方便审计：absent 是"答案没声称"，
            # 在 IFS 里被忽略；unfaithful 是"声称了但没依据"，才真正扣分。
            rows_detail.append({
                "rubric_row": r.rubric_row,
                "score": 1 if row_faithful else 0,
                "llm_score_raw": r.score,
                "skipped": row_skipped,
                "is_faithful": row_faithful,
                "n_elements": len(elements_dump),
                "n_grounded": len(grounded),
                "n_absent": len(absent),
                "n_unfaithful": len(unfaithful),
                "n_skipped": len(skipped_elems),
                "elements_grounded": grounded,
                "elements_absent_ignored": absent,
                "elements_unfaithful": unfaithful,
                "elements_skipped": skipped_elems,
                "reasoning": r.reasoning,
            })

        return {
            "formula": (
                "N_faithful_rows / N_total_rows  "
                "(per-row: drop absent+skipped elements first, then "
                "all-or-nothing on what remains; skipped rows excluded "
                "from both num & den)"
            ),
            "method": "rubric_row_judgment",
            "numerator": n_faithful,
            "denominator": n_total,
            "n_faithful_rows": n_faithful,
            "n_unfaithful_rows": n_unfaithful,
            "n_skipped_rows": n_skipped,
            "n_total_judged_rows": n_total_raw,
            "n_gt_rows": len(gt_rows),
            "rows_detail": rows_detail,
        }

    # ── 兼容路径：老 fact_summary 统计 ────────────────────────────────
    fs = verdict.fact_summary
    faithful_facts: list[dict[str, Any]] = []
    flipped_facts: list[dict[str, Any]] = []
    for f in fs.faithful_facts:
        if _is_unfaithful(f):
            flipped_facts.append(_fact_to_dict(f, force_unfaithful=True))
        else:
            faithful_facts.append(_fact_to_dict(f))

    unfaithful_facts: list[dict[str, Any]] = [
        _fact_to_dict(f, force_unfaithful=True) for f in fs.unfaithful_facts
    ]
    unfaithful_facts.extend(flipped_facts)

    faithful_count = len(faithful_facts)
    unfaithful_count = len(unfaithful_facts)
    n_total = faithful_count + unfaithful_count

    by_source: dict[str, dict[str, int]] = {}
    for d in faithful_facts:
        key = d["source_tool"] or "(no_source_tool)"
        by_source.setdefault(key, {"faithful": 0, "unfaithful": 0})["faithful"] += 1
    for d in unfaithful_facts:
        key = d["source_tool"] or "(no_source_tool)"
        by_source.setdefault(key, {"faithful": 0, "unfaithful": 0})["unfaithful"] += 1

    return {
        "formula": "N_faithful / N_total  (legacy, fact-level fallback)",
        "method": "legacy_fact_summary",
        "numerator": faithful_count,
        "denominator": n_total,
        "faithful_count": faithful_count,
        "unfaithful_count": unfaithful_count,
        "by_source_tool": by_source,
        "faithful_facts": faithful_facts,
        "unfaithful_facts": unfaithful_facts,
    }
