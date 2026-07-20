"""Centralised input preprocessing for the 4-way standalone judge fan-out.

All four LLM judges (ECR / TS / IFS / IISR) need overlapping but
distinct slices of the same source material. Rather than have each judge
re-derive normalisation / annotation / tool-call extraction, we run that
work **once** in :func:`prepare_judge_inputs` and hand every judge a
frozen :class:`JudgeInputs` snapshot.

The snapshot also pre-computes TS dim1 + dim2 (deterministic code
filters) so the TS judge LLM only sees candidates that have already
passed name-membership and required-param checks — keeping its job to
just dim3 (parameter shape + traceability).

The dataclass is ``frozen=True, slots=True`` so judges can read but not
mutate the shared state — each judge's contribution to the final
``JudgeVerdict`` is communicated via its own return value, not by
poking the input bag.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.core.evaluation.conversation_data_normalizer import (
    normalize_conversation_history,
)
from app.core.evaluation.metrics.ts import (
    _args_have_all,
    _dedup_calls_by_longest_args,
    _extract_pred_calls,
    _required_params,
)
from app.core.evaluation.schema import GroundTruth
from app.core.evaluation.tool_response_utils import annotate_tool_responses


@dataclass(frozen=True, slots=True)
class JudgeInputs:
    """Immutable bundle of everything the 5 standalone judges might need.

    Field groupings:

    * **Raw / context** — ``case_id``, ``query``, ``full_intent``,
      ``current_time``, ``current_location``, ``tools_schema``,
      ``raw_conversation_history`` (bundled format; TS/Eff want this),
      ``ground_truth`` (validated).
    * **Derived history** — ``normalized_history`` (flat OpenAI-style)
      and ``annotated_history`` (normalized + ``_classification`` on
      each tool response).
    * **Per-judge precomputes** —
      ``assistant_turns`` for IISR,
      ``factual_rubric`` for IFS,
      ``ts_pred_calls`` / ``ts_deduped_args`` / ``ts_dim12_pass`` /
      ``ts_candidates_for_llm`` for the TS code-side filters.
    """

    case_id: str
    query: str
    full_intent: str
    current_time: str
    current_location: str
    tools_schema: list[dict[str, Any]]
    raw_conversation_history: list[dict[str, Any]]
    ground_truth: GroundTruth

    normalized_history: list[dict[str, Any]]
    annotated_history: list[dict[str, Any]]
    assistant_turns: list[dict[str, Any]]
    factual_rubric: list[str]

    ts_pred_calls: list[tuple[str, Any]]
    ts_deduped_args: dict[str, Any]
    ts_dim12_pass: set[str]
    ts_candidates_for_llm: list[dict[str, Any]] = field(default_factory=list)


def _extract_assistant_turns(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Chronological list of assistant content turns (skip tool-only turns).

    Mirrors :meth:`IISRJudge._extract_assistant_turns` so both code paths
    produce the same list. Kept module-level here so non-IISR consumers
    (e.g. tests, future judges) can share it without importing IISRJudge.
    """
    out: list[dict[str, Any]] = []
    turn = 0
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not content:
            continue
        turn += 1
        out.append({"turn": turn, "content": content})
    return out


def _build_ts_candidates(
    gt: GroundTruth,
    deduped: dict[str, Any],
) -> tuple[set[str], list[dict[str, Any]]]:
    """Filter deduped calls through dim1 (gold) ∧ dim2 (required params).

    Returns:
      * ``dim12_pass``: tool names that pass both checks (used to clamp
        the LLM's dim3 verdict — anything the LLM "passes" that isn't in
        this set is treated as fabrication and dropped).
      * ``candidates``: prompt payload for the TS LLM, one entry per
        passing tool ``{tool_name, args, rules}`` — ``rules`` is the
        ``parameter_rules[name].rules`` dict (key → natural-language
        description) so the LLM can do shape/type matching without us
        re-flattening it.
    """
    gold = set(gt.truth_trajectory.tool_calls.expected_tools or [])
    param_rules = gt.truth_trajectory.tool_calls.parameter_rules or {}
    dim12_pass: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for name, args in deduped.items():
        if name not in gold:
            continue
        if not _args_have_all(args, _required_params(gt, name)):
            continue
        dim12_pass.add(name)
        block = param_rules.get(name) or {}
        rules = block.get("rules") if isinstance(block, dict) else None
        if not isinstance(rules, dict):
            rules = {}
        # If args came in as a JSON string, parse it for the LLM payload
        # so the prompt shows the actual key/value pairs rather than an
        # opaque blob. Dedup helper preserves the raw shape on purpose
        # (so downstream `_args_have_all` keeps working with strings).
        prompt_args: Any = args
        if isinstance(prompt_args, str):
            try:
                parsed = json.loads(prompt_args)
                if isinstance(parsed, dict):
                    prompt_args = parsed
            except (TypeError, ValueError):
                pass
        candidates.append(
            {"tool_name": name, "args": prompt_args, "rules": rules}
        )
    return dim12_pass, candidates


def prepare_judge_inputs(
    *,
    case_id: str,
    query: str,
    full_intent: str,
    current_time: str,
    current_location: str,
    raw_conversation_history: list[dict[str, Any]],
    ground_truth: GroundTruth | dict[str, Any],
    tools_schema: list[dict[str, Any]] | dict[str, Any] | None = None,
) -> JudgeInputs:
    """Run every shared preprocessing step once; return the frozen bundle.

    ``ground_truth`` may be a dict (will be validated) or an already-
    validated :class:`GroundTruth`. ``tools_schema`` is normalised to a
    list (the bundled judge accepted dict too).
    """
    gt: GroundTruth = (
        ground_truth
        if isinstance(ground_truth, GroundTruth)
        else GroundTruth.model_validate(ground_truth)
    )
    schema_list: list[dict[str, Any]]
    if tools_schema is None:
        schema_list = []
    elif isinstance(tools_schema, list):
        schema_list = tools_schema
    else:
        # dict shape (rare) — keep it as a single-entry list so prompts
        # that JSON-dump it still see something sensible.
        schema_list = [tools_schema]  # type: ignore[list-item]

    normalized = normalize_conversation_history(raw_conversation_history)
    annotated = annotate_tool_responses(normalized)
    assistant_turns = _extract_assistant_turns(annotated)

    factual_rubric = list(
        gt.truth_trajectory.tool_calls.factual_answer_rubric or []
    )

    ts_pred = _extract_pred_calls(raw_conversation_history)
    ts_deduped = _dedup_calls_by_longest_args(ts_pred)
    ts_dim12_pass, ts_candidates = _build_ts_candidates(gt, ts_deduped)

    return JudgeInputs(
        case_id=case_id,
        query=query,
        full_intent=full_intent,
        current_time=current_time,
        current_location=current_location,
        tools_schema=schema_list,
        raw_conversation_history=raw_conversation_history,
        ground_truth=gt,
        normalized_history=normalized,
        annotated_history=annotated,
        assistant_turns=assistant_turns,
        factual_rubric=factual_rubric,
        ts_pred_calls=ts_pred,
        ts_deduped_args=ts_deduped,
        ts_dim12_pass=ts_dim12_pass,
        ts_candidates_for_llm=ts_candidates,
    )
