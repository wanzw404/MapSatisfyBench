"""TS-Acc — Tool Selection Accuracy.

Two computation paths:

* **Deterministic (preferred)** — when ``conversation_history`` is provided,
  TS is computed directly from the bundled history + ground truth, with no
  reliance on the LLM verdict. Both gold and predicted tool names are
  **deduplicated by name**: for predicted, the dedup keeps the call with
  the most filled args (see ``_dedup_calls_by_longest_args``). A predicted
  name is "correct" iff it appears in the gold tool list AND its surviving
  call includes every required parameter listed under
  ``ground_truth.truth_trajectory.tool_calls.parameter_rules[name].paramter``
  (typo "paramter" is the current data convention; "parameter" is also
  accepted).

* **Legacy** — when no history is passed, TS falls back to the existing
  LLM-verdict-derived breakdown so older callers / tests keep working.

Formula (both paths, each tool name appears in exactly one bucket)::

    TS = |correct| / (|correct| + |incorrect| + |extra| + |missed|)

Bucket disjointness (the key invariant):

    correct   ⊆ gold ∩ pred       (dim1 ∧ dim2 ∧ dim3 all pass)
    incorrect ⊆ gold ∩ pred       (dim1 pass, dim2 or dim3 fail)
    extra     = pred  - gold      (called but not in gold)
    missed    = gold - (correct ∪ incorrect)   (in gold but never called)

A gold tool that was called with wrong params lands in ``incorrect`` ONLY
— it is no longer also counted in ``missed`` (which would double-penalise
the same tool in the denominator).
"""

from __future__ import annotations

import json
from typing import Any

from app.core.evaluation.schema import (
    GroundTruth,
    JudgeVerdict,
    ToolCallDetail,
)


# ---------------------------------------------------------------------------
# Helpers shared by legacy path
# ---------------------------------------------------------------------------


def _canonical_params(params: dict[str, Any] | None) -> str:
    """Stable JSON serialisation of a tool-call argument dict for hashing."""
    if not params:
        return ""
    try:
        return json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(sorted(params.items()))


def _correct_key(c: ToolCallDetail) -> tuple[str, str]:
    return (c.tool_name, _canonical_params(c.parameters))


# ---------------------------------------------------------------------------
# Helpers for the new deterministic path
# ---------------------------------------------------------------------------


def _required_params(gt: GroundTruth, tool_name: str) -> list[str]:
    """Required-param names for ``tool_name`` per parameter_rules.

    Tolerates both ``paramter`` (current data typo) and ``parameter``.
    Returns an empty list if no rule is defined for the tool — in that case
    "correct" reduces to "name appears in gold".
    """
    pr = gt.truth_trajectory.tool_calls.parameter_rules or {}
    rule = pr.get(tool_name) or {}
    raw = rule.get("paramter")
    if raw is None:
        raw = rule.get("parameter")
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        # Tolerate {param_name: {"required": bool, ...}} — treat missing
        # ``required`` as truthy (assume required).
        return [
            k for k, v in raw.items()
            if not isinstance(v, dict) or v.get("required", True)
        ]
    return []


def _args_have_all(args: Any, required: list[str]) -> bool:
    """True iff every required name is present in args with a non-empty value.

    Accepts ``args`` as a dict or a JSON string (LLM tool-call arguments are
    sometimes strings). Empty string / None / [] / {} count as missing.
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (TypeError, ValueError):
            return not required  # opaque string — only OK if nothing required
    if not isinstance(args, dict):
        return not required
    for p in required:
        if p not in args:
            return False
        v = args[p]
        if v is None or v == "" or v == [] or v == {}:
            return False
    return True


def _extract_pred_calls(
    history: list[dict[str, Any]] | None,
) -> list[tuple[str, Any]]:
    """Return ``[(tool_name, args)]`` from every tool_call in history (no dedup).

    Handles both the bundled format (``{"name": ..., "args": {...}}``) and
    the OpenAI-style nested form (``{"function": {"name": ..., "arguments": "..."}}``).
    """
    out: list[tuple[str, Any]] = []
    for msg in history or []:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name") or (tc.get("function") or {}).get("name", "")
            args = tc.get("args")
            if args is None:
                args = tc.get("arguments")
            if args is None and isinstance(tc.get("function"), dict):
                args = tc["function"].get("arguments")
            if name:
                out.append((name, args))
    return out


def _args_len(args: Any) -> int:
    """Effective key count for dedup-by-longest-args.

    Returns the dict key count when ``args`` is a dict (or a JSON-parsable
    string yielding a dict). Anything else — None, opaque string, list,
    primitive — counts as 0. This is *only* a comparison key for picking
    the most-informative call when the same tool is invoked multiple
    times; it never affects TS scoring directly.
    """
    if isinstance(args, dict):
        return len(args)
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except (TypeError, ValueError):
            return 0
        return len(parsed) if isinstance(parsed, dict) else 0
    return 0


def _dedup_calls_by_longest_args(
    pred_calls: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Keep at most one entry per tool name: the one with the most filled args.

    Tie-breaker: first occurrence wins (strict ``>`` replacement). The
    surviving ``args`` is returned as-is — not re-serialised — so a JSON
    string stays a string; downstream code must tolerate both shapes (the
    same way ``_args_have_all`` already does).
    """
    by_name: dict[str, Any] = {}
    best_len: dict[str, int] = {}
    for name, args in pred_calls:
        n = _args_len(args)
        if name not in by_name or n > best_len[name]:
            by_name[name] = args
            best_len[name] = n
    return by_name


def _ts_breakdown_from_history(
    gt: GroundTruth,
    history: list[dict[str, Any]],
    dim3_pass: set[str] | None = None,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Return ``(correct, incorrect, extra, missed)`` name-sets, all dedup'd.

    Buckets are **disjoint** — every tool name lands in exactly one of the
    four sets, so it contributes at most 1 to the TS denominator. Concretely:

    * ``correct``   — gold ∩ pred, with dim2 (required params) ∧ dim3 (form)
                      both passing on the longest-args surviving call.
    * ``incorrect`` — gold ∩ pred, but dim2 or dim3 failed.
    * ``extra``     — pred − gold (called but not in gold).
    * ``missed``    — gold − pred (never called at all).

    A gold tool called with wrong params is **incorrect only**, NOT also
    missed; the older double-bucket scheme inflated the denominator by 1
    for every parameter-error case (~75% of real cases in practice).
    """
    gold = set(gt.truth_trajectory.tool_calls.expected_tools or [])
    deduped = _dedup_calls_by_longest_args(_extract_pred_calls(history))
    pred_names = set(deduped.keys())

    correct: set[str] = set()
    incorrect: set[str] = set()
    for name in pred_names & gold:
        dim2_ok = _args_have_all(deduped[name], _required_params(gt, name))
        dim3_ok = True if dim3_pass is None else (name in dim3_pass)
        if dim2_ok and dim3_ok:
            correct.add(name)
        else:
            incorrect.add(name)

    extra = pred_names - gold
    missed = gold - pred_names
    return correct, incorrect, extra, missed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_ts(
    verdict: JudgeVerdict,
    gt: GroundTruth,
    conversation_history: list[dict[str, Any]] | None = None,
    *,
    dim3_pass: set[str] | None = None,
) -> float:
    if conversation_history is not None:
        correct, incorrect, extra, missed = _ts_breakdown_from_history(
            gt, conversation_history, dim3_pass=dim3_pass
        )
        num = len(correct)
        den = len(correct) + len(incorrect) + len(extra) + len(missed)
        if den == 0:
            return 0.0
        return max(0.0, min(1.0, num / den))

    # ── Legacy path: derived from the LLM verdict's tool_call_summary ────
    s = verdict.tool_call_summary
    gold_names: set[str] = set(gt.truth_trajectory.tool_calls.expected_tools)
    correct_keys: set[tuple[str, str]] = {_correct_key(c) for c in s.correct_calls}
    correct_names: set[str] = {name for name, _ in correct_keys}
    incorrect_names: set[str] = {c.tool_name for c in s.incorrect_calls}
    extra_count = len(s.extra_calls)
    # 同 deterministic 路径：incorrect 桶里的 gold 工具不再算第二次进 missed。
    missed_names = gold_names - correct_names - incorrect_names
    intersection_keys = {k for k in correct_keys if k[0] in gold_names}
    numerator = len(intersection_keys)
    denominator = (
        len(correct_keys)
        + len(s.incorrect_calls)
        + extra_count
        + len(missed_names)
    )
    if denominator == 0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def details_ts(
    verdict: JudgeVerdict,
    gt: GroundTruth,
    conversation_history: list[dict[str, Any]] | None = None,
    *,
    dim3_pass: set[str] | None = None,
) -> dict[str, Any]:
    if conversation_history is not None:
        correct, incorrect, extra, missed = _ts_breakdown_from_history(
            gt, conversation_history, dim3_pass=dim3_pass
        )
        gold = set(gt.truth_trajectory.tool_calls.expected_tools or [])
        num = len(correct)
        den = len(correct) + len(incorrect) + len(extra) + len(missed)
        # 失败原因（dim2 缺参 / dim3 形态错）由 TSJudge._synthesise_summary
        # 写到 verdict.tool_call_summary.incorrect_calls[*].reason；details
        # 这里只放精简的 list[str]，需要 reason 的下游（meta_judge /
        # explainer）直接读 verdict slot 即可。
        return {
            "formula": (
                "|correct| / (|correct| + |incorrect| + |extra| + |missed|)  "
                "(deduped by name on both sides; buckets disjoint — a gold "
                "tool with wrong params is incorrect-only, not also missed)"
            ),
            "numerator": num,
            "denominator": den,
            "gold_tools": sorted(gold),
            "correct_tools": sorted(correct),
            "incorrect_tools": sorted(incorrect),
            "extra_tools": sorted(extra),
            "missed_tools": sorted(missed),
        }

    # ── Legacy details path ──────────────────────────────────────────────
    s = verdict.tool_call_summary
    gold_names: set[str] = set(gt.truth_trajectory.tool_calls.expected_tools)
    correct_keys: set[tuple[str, str]] = {_correct_key(c) for c in s.correct_calls}
    correct_names: set[str] = {name for name, _ in correct_keys}
    incorrect_names: set[str] = {c.tool_name for c in s.incorrect_calls}
    # 与 deterministic 路径保持一致：incorrect 桶里的 gold 工具不再算第二次进 missed。
    missed_names = gold_names - correct_names - incorrect_names
    intersection_keys = {k for k in correct_keys if k[0] in gold_names}

    denominator = (
        len(correct_keys)
        + len(s.incorrect_calls)
        + len(s.extra_calls)
        + len(missed_names)
    )

    return {
        "formula": (
            "|correct∩gold (by name)| / "
            "(|correct (name,params)| + |incorrect| + |extra| + |missed|)  "
            "(missed excludes names already in incorrect)"
        ),
        "numerator": len(intersection_keys),
        "denominator": denominator,
        "gold_tools": sorted(gold_names),
        "correct_keys": sorted(
            f"{name}{(':' + params) if params else ''}"
            for name, params in correct_keys
        ),
        "incorrect_count": len(s.incorrect_calls),
        "extra_count": len(s.extra_calls),
        "missed_tools": sorted(missed_names),
    }
