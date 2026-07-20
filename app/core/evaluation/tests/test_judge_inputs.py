"""Unit tests for the shared judge-input preprocessor.

Covers:

* ``prepare_judge_inputs`` runs all 5 derivation steps without errors and
  produces well-typed outputs.
* TS dim1+dim2 candidate construction: only tools that are in
  ``expected_tools`` AND have all required params get into
  ``ts_candidates_for_llm`` / ``ts_dim12_pass``.
* ``_dedup_calls_by_longest_args`` keeps the call with the most args
  filled (so a "{poi:'xx'}" call wins over a "{}" call for the same
  tool name).
* Tool-only assistant turns are excluded from ``assistant_turns`` —
  IISR judges based on textual responses, not silent tool invocations.
"""

from __future__ import annotations

from app.core.evaluation.judge_inputs import (
    _extract_assistant_turns,
    prepare_judge_inputs,
)
from app.core.evaluation.metrics.ts import _dedup_calls_by_longest_args
from app.core.evaluation.schema import (
    ClarificationPolicy,
    GroundTruth,
    ToolCallRubric,
    TruthTrajectory,
)


def _gt() -> GroundTruth:
    return GroundTruth(
        explicit_intent=["找川菜馆"],
        implicit_intent=[],
        clarification_policy=ClarificationPolicy(max_allowed=3),
        truth_trajectory=TruthTrajectory(
            tool_calls=ToolCallRubric(
                expected_tools=["search_around_poi", "driving_route"],
                parameter_rules={
                    "search_around_poi": {
                        "parameter": ["keyword", "location"],
                        "rules": {
                            "keyword": "POI keyword",
                            "location": "lng,lat string",
                        },
                    },
                    "driving_route": {
                        "parameter": ["origin", "destination"],
                        "rules": {"origin": "...", "destination": "..."},
                    },
                },
                factual_answer_rubric=["最终回答需说明营业时间"],
            ),
        ),
    )


def _history_one_good_call() -> list[dict]:
    """Bundled-style history: one round-trip where search_around_poi has all required params."""
    return [
        {"role": "user", "content": "找附近川菜馆"},
        {
            "role": "assistant",
            "content": "好的",
            "tool_calls": [
                {
                    "name": "search_around_poi",
                    "args": {"keyword": "川菜", "location": "116.4,39.9"},
                    "response": "{\"pois\": [{\"name\": \"X 川菜\"}]}",
                }
            ],
        },
        {"role": "assistant", "content": "推荐 X 川菜"},
    ]


def test_prepare_judge_inputs_basic_shape():
    gt = _gt()
    inputs = prepare_judge_inputs(
        case_id="c1",
        query="找川菜馆",
        full_intent="找川菜馆",
        current_time="2026-05-26 12:00",
        current_location="116.4,39.9",
        raw_conversation_history=_history_one_good_call(),
        ground_truth=gt,
        tools_schema=None,
    )
    # ground_truth is normalised to a GroundTruth instance
    assert isinstance(inputs.ground_truth, GroundTruth)
    # factual_rubric pulled from GT, not from raw kwargs
    assert inputs.factual_rubric == ["最终回答需说明营业时间"]
    # tools_schema=None becomes []
    assert inputs.tools_schema == []
    # annotated_history is at least as long as normalized_history minus user turns
    assert len(inputs.annotated_history) >= 1
    # assistant_turns excludes the tool-only turn (no content): the
    # first assistant turn has content "好的", the second "推荐 X 川菜".
    # Both have content so both are included.
    assert len(inputs.assistant_turns) == 2
    assert all("turn" in t and "content" in t for t in inputs.assistant_turns)


def test_ts_dim12_pass_includes_valid_call():
    gt = _gt()
    inputs = prepare_judge_inputs(
        case_id="c1", query="", full_intent="", current_time="",
        current_location="",
        raw_conversation_history=_history_one_good_call(),
        ground_truth=gt,
    )
    # search_around_poi is in gold AND has both required params filled
    assert "search_around_poi" in inputs.ts_dim12_pass
    # driving_route was never called
    assert "driving_route" not in inputs.ts_dim12_pass
    # The LLM candidate carries args (parsed to dict) + rules block
    cand = next(c for c in inputs.ts_candidates_for_llm
                if c["tool_name"] == "search_around_poi")
    assert isinstance(cand["args"], dict)
    assert cand["args"]["keyword"] == "川菜"
    assert "keyword" in cand["rules"]


def test_ts_dim12_pass_drops_missing_required():
    """Tool name in gold but required params missing → dim2 fails → not in pass set."""
    gt = _gt()
    incomplete = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "search_around_poi",
                    "args": {"keyword": "川菜"},  # missing 'location'
                    "response": "{}",
                }
            ],
        },
    ]
    inputs = prepare_judge_inputs(
        case_id="c1", query="", full_intent="", current_time="",
        current_location="",
        raw_conversation_history=incomplete,
        ground_truth=gt,
    )
    assert "search_around_poi" not in inputs.ts_dim12_pass
    assert inputs.ts_candidates_for_llm == []


def test_dedup_keeps_longest_args():
    """Same tool called twice — once with full args, once with subset → keep full."""
    calls = [
        ("tool_a", {"k1": 1}),
        ("tool_a", {"k1": 1, "k2": 2}),
        ("tool_a", {}),
    ]
    deduped = _dedup_calls_by_longest_args(calls)
    assert deduped["tool_a"] == {"k1": 1, "k2": 2}


def test_extract_assistant_turns_skips_empty_content():
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "x"}]},
        {"role": "tool", "content": "{}"},
        {"role": "assistant", "content": "second"},
    ]
    turns = _extract_assistant_turns(history)
    assert [t["content"] for t in turns] == ["first", "second"]
    assert [t["turn"] for t in turns] == [1, 2]
