"""IISR - Implicit-decision-factor Satisfaction Rate (weighted average).

IISR = Σ(Wi * Ci) / Σ(Wi)

Where:
  * Wi = ``evidence_confidence`` (from ground truth)
  * Ci = ``satisfaction_score_Ci`` (LLM judgement, in [0, 1])

Hard-constraint rubrics participate in the weighted average like any other
rubric. A violated hard constraint contributes Wi×0.0 to the numerator,
naturally dragging the score down in proportion to its weight. There is no
binary kill-switch — the weighted formula handles it correctly.

No explicit-intent gate (2026-05-27):
  Earlier versions forced IISR=0 whenever ECR was fully unmet
  (``success_count == 0``). That was a runtime escape hatch that
  contradicted the "every rubric in GT enters the weighted sum
  unconditionally" principle and inflated divergence vs. human
  annotators (who compute IISR purely from the formula). The gate is
  removed: IISR is always Σ(Wi·Ci)/Σ(Wi). The "explicit intent failed
  drags the case to 0" semantic is still captured by AR = ECR · IISR.
  ``details_iisr`` still surfaces ``explicit_intents_unmet`` as an
  informational audit flag.

Triggering policy (2026-05): every rubric in GT enters the weighted sum
unconditionally. The legacy ``triggered`` field is retained on
``ImplicitJudgment`` for backward compatibility but is ignored by the
calculator — time/location feasibility is the GT author's responsibility,
not a runtime escape hatch.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from app.core.evaluation.schema import (
    GroundTruth,
    ImplicitIntent,
    ImplicitJudgment,
    JudgeVerdict,
)


# Whitespace + common CJK/ASCII punctuation that LLMs casually swap in or
# drop when echoing a rubric. The normalizer wipes all of these so two
# strings that differ only by stray spaces or "。" vs "" still match.
_PUNCT_RE = re.compile(
    r"[\s　.,;:!?\"'\(\)\[\]\{\}\-_/\\~`+*=<>|"
    r"。，；：！？、‘’“”（）【】《》「」『』〈〉—…·]+"
)


def _normalize(s: str) -> str:
    """Squash whitespace + punctuation so paraphrase-drift survives lookup."""
    return _PUNCT_RE.sub("", (s or "")).lower()


def _match_judgments(
    gt_rubrics: list[ImplicitIntent],
    judgments: list[ImplicitJudgment],
) -> tuple[list[tuple[Optional[ImplicitJudgment], str]], list[ImplicitJudgment]]:
    """Align LLM judgments to GT rubrics with three layered strategies.

    Returns
    -------
    per_rubric :
        Same length / order as ``gt_rubrics``. Each entry is
        ``(judgment_or_none, match_method)`` where ``match_method`` is one
        of ``"exact" | "normalized" | "positional" | "missing"``.
    extras :
        Judgments emitted by the LLM that did not bind to any GT rubric
        (preserved for audit; caller may surface them with ``[GT 未列出]``).

    Strategies (in order):
      1. **exact**      — byte-for-byte ``rubric_instruction`` match
      2. **normalized** — strip whitespace + CJK/ASCII punctuation, lowercase
      3. **positional** — when ``len(judgments) == len(gt_rubrics)``, the
         residual unmatched indices on both sides are zipped by position.
         Standalone prompts enumerate GT rubrics in order and the LLM
         almost always preserves that order; this rescues the common case
         where the LLM truncates / rewords each rubric just enough to
         break normalized lookup.
    """
    per_rubric: list[tuple[Optional[ImplicitJudgment], str]] = [
        (None, "missing") for _ in gt_rubrics
    ]
    used_j: set[int] = set()

    # Pass 1: exact text match.
    exact_idx: dict[str, int] = {}
    for ji, j in enumerate(judgments):
        # First writer wins; later collisions land in `extras` via the
        # `used_j` check below.
        exact_idx.setdefault(j.rubric_instruction, ji)
    for ri, rubric in enumerate(gt_rubrics):
        ji = exact_idx.get(rubric.rubric_instruction)
        if ji is not None and ji not in used_j:
            per_rubric[ri] = (judgments[ji], "exact")
            used_j.add(ji)

    # Pass 2: normalized text match for whatever pass 1 missed.
    norm_idx: dict[str, int] = {}
    for ji, j in enumerate(judgments):
        if ji in used_j:
            continue
        norm_idx.setdefault(_normalize(j.rubric_instruction), ji)
    for ri, rubric in enumerate(gt_rubrics):
        if per_rubric[ri][0] is not None:
            continue
        key = _normalize(rubric.rubric_instruction)
        ji = norm_idx.get(key)
        if ji is not None and ji not in used_j:
            per_rubric[ri] = (judgments[ji], "normalized")
            used_j.add(ji)

    # Pass 3: positional zip — only when the LLM emitted exactly one
    # judgment per GT rubric. The residual unmatched slots on both sides
    # get paired in original order. Skipping this when counts differ
    # avoids silently binding a stray extra/missing judgment to the wrong
    # rubric (extras land in the `extras` return instead).
    if len(judgments) == len(gt_rubrics):
        unmatched_ri = [ri for ri, (j, _) in enumerate(per_rubric) if j is None]
        unmatched_ji = [ji for ji in range(len(judgments)) if ji not in used_j]
        for ri, ji in zip(unmatched_ri, unmatched_ji):
            per_rubric[ri] = (judgments[ji], "positional")
            used_j.add(ji)

    extras = [judgments[ji] for ji in range(len(judgments)) if ji not in used_j]
    return per_rubric, extras


def _explicit_intents_unmet(verdict: JudgeVerdict, gt: GroundTruth) -> bool:
    """Return True iff there is at least one explicit intent in the ground
    truth and zero of them were marked as satisfied by the judge.
    """
    total = verdict.explicit_intent_summary.total_count or len(gt.explicit_intent)
    success = verdict.explicit_intent_summary.success_count
    return total > 0 and success == 0


def compute_iisr(verdict: JudgeVerdict, gt: GroundTruth) -> float:
    # No implicit-intent rubric in GT → no constraint to satisfy → 1.0.
    # This makes AR = ECR · IISR collapse to AR = ECR for cases without
    # any implicit intent (per AR spec).
    if not gt.implicit_intent:
        return 1.0

    # Wi is always GT's evidence_confidence — never trust LLM's echoed
    # `confidence_Wi` (it routinely drifts when copied, and a paraphrased
    # rubric used to bypass the loop entirely, double-deflating IISR).
    # Ci comes from the matched judgment (exact → normalized → positional);
    # unmatched rubrics still enter the denominator at Ci=0.
    per_rubric, _extras = _match_judgments(
        gt.implicit_intent, verdict.implicit_intent_judgments
    )

    total_w = 0.0
    weighted_sum = 0.0
    for rubric, (j, _method) in zip(gt.implicit_intent, per_rubric):
        # Wi 不再设上限：标注可用 >1 的权重表示"超硬约束"；
        # IISR 是加权平均，Wi 任意正值都让结果 ≤ max(Ci) ≤ 1，结果仍在 [0,1]。
        wi = max(0.0, rubric.evidence_confidence)
        ci = max(0.0, min(1.0, j.satisfaction_score_Ci)) if j is not None else 0.0
        total_w += wi
        weighted_sum += wi * ci

    if total_w <= 0:
        # GT has implicit intents but Σ(Wi)=0 → no evidence → 0.
        return 0.0
    return max(0.0, min(1.0, weighted_sum / total_w))


def _classify_satisfaction(ci: float) -> str:
    """把 Ci 数值翻译成可读的满足度档位，方便审计 / reason 文字直接引用。"""
    if ci >= 0.9:
        return "完全满足"
    if ci >= 0.5:
        return "部分满足"
    if ci > 0.0:
        return "弱满足"
    return "未满足"


def details_iisr(verdict: JudgeVerdict, gt: GroundTruth) -> dict[str, Any]:
    judgments = verdict.implicit_intent_judgments

    # Wi 一律取 GT 的 evidence_confidence；Ci 来自匹配上的 LLM 判定。
    # 同 compute_iisr 共用 _match_judgments，保证两边数字 100% 一致。
    per_rubric, extras = _match_judgments(gt.implicit_intent, judgments)

    total_w = 0.0
    weighted_sum = 0.0
    hard_violated = False
    rubrics_detail: list[dict[str, Any]] = []
    match_counts = {"exact": 0, "normalized": 0, "positional": 0, "missing": 0}

    for rubric, (j, method) in zip(gt.implicit_intent, per_rubric):
        match_counts[method] += 1
        # 同 compute_iisr：Wi 只 floor 不 cap，标注 1.1/1.2/... 原样进入加权
        wi = max(0.0, rubric.evidence_confidence)
        ctype = rubric.constraint_type

        if j is None:
            total_w += wi
            rubrics_detail.append({
                "rubric": rubric.rubric_instruction,
                "constraint_type": ctype,
                "Wi": round(wi, 4),
                "Ci": 0.0,
                "weighted_contribution": 0.0,
                "violates_hard_constraint": False,
                "satisfaction_label": "judge 未给判定",
                "match_method": method,
                "reasoning": "",
            })
            continue

        ci = max(0.0, min(1.0, j.satisfaction_score_Ci))
        is_hard_fail = (
            ctype == "hard"
            and (j.violates_hard_constraint or j.satisfaction_score_Ci <= 0.0)
        )
        if is_hard_fail:
            hard_violated = True

        total_w += wi
        weighted_sum += wi * ci

        rubrics_detail.append({
            "rubric": rubric.rubric_instruction,
            "constraint_type": ctype,
            "Wi": round(wi, 4),
            "Ci": round(ci, 4),
            "weighted_contribution": round(wi * ci, 4),
            "violates_hard_constraint": is_hard_fail,
            "satisfaction_label": _classify_satisfaction(ci),
            "match_method": method,
            "reasoning": (j.reasoning or "").strip(),
        })

    # LLM 多给的 rubric（GT 里没有，且未被任一 GT 槽位吃掉）保留供审计；
    # 不进入 IISR 分子/分母——权重要么来自 GT 要么不存在。
    for j in extras:
        ci = max(0.0, min(1.0, j.satisfaction_score_Ci))
        rubrics_detail.append({
            "rubric": j.rubric_instruction,
            "constraint_type": "soft",  # 未知，保守按 soft
            "Wi": 0.0,
            "Ci": round(ci, 4),
            "weighted_contribution": 0.0,
            "violates_hard_constraint": False,
            "satisfaction_label": _classify_satisfaction(ci),
            "match_method": "extra",
            "reasoning": (j.reasoning or "").strip() + " [GT 未列出此 rubric]",
        })

    explicit_unmet = _explicit_intents_unmet(verdict, gt)

    return {
        "formula": (
            "Σ(Wi·Ci) / Σ(Wi); Wi=GT.evidence_confidence; "
            "Ci from matched judgment (exact→normalized→positional fallback); "
            "1.0 if GT has no implicit intent. No explicit-intent gate — "
            "the 'ECR=0 → drag to 0' semantic is captured by AR = ECR · IISR."
        ),
        "weighted_sum": round(weighted_sum, 4),
        "total_weight": round(total_w, 4),
        "hard_constraint_violated": hard_violated,
        "explicit_intents_unmet": explicit_unmet,
        "n_rubrics_in_gt": len(gt.implicit_intent),
        "n_rubrics_scored": len(rubrics_detail),
        "match_counts": match_counts,
        "n_extra_judgments": len(extras),
        "rubrics_detail": rubrics_detail,
    }
