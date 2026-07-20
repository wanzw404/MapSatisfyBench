"""Integration test for JudgeAgent's 4-way standalone fan-out.

These tests use stub judges (no LLM, no network) to verify the
orchestration contract:

* When all 4 judges return successfully, every verdict slot is filled
  from the corresponding judge and ``judge_status`` is all "ok".
* When one judge returns ``None``, the verdict slot keeps its schema
  default and ``judge_status[name] == "failed"``. The case does NOT
  raise.
* When one judge raises, the same fallback path kicks in (gather
  return_exceptions=True), the others still run, and metrics still
  compute.
* TS ``dim3_pass`` is plumbed into the metric calculator (verified by
  observing that the synthesised tool_call_summary lands in the verdict).
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from app.core.evaluation.ifs_judge import IFSStandaloneResult
from app.core.evaluation.judge_agent import JudgeAgent
from app.core.evaluation.schema import (
    ClarificationPolicy,
    ExplicitIntentSummary,
    FactSummary,
    GroundTruth,
    ImplicitIntent,
    ImplicitJudgment,
    ToolCallDetail,
    ToolCallRubric,
    ToolCallSummary,
    TruthTrajectory,
)
from app.core.evaluation.ts_judge import TSStandaloneResult


class _StubJudge:
    """Bare-minimum judge stub: ``evaluate(inputs)`` returns a preset value
    or raises a preset exception. Tracks whether it was called."""

    def __init__(self, returns: Any = None, raises: Optional[BaseException] = None):
        self.returns = returns
        self.raises = raises
        self.calls = 0

    async def evaluate(self, inputs: Any) -> Any:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.returns


def _gt() -> GroundTruth:
    return GroundTruth(
        explicit_intent=["找川菜馆"],
        implicit_intent=[
            ImplicitIntent(
                rubric_instruction="附近",
                constraint_type="soft",
                evidence_confidence=0.8,
                evidence="x",
            ),
        ],
        clarification_policy=ClarificationPolicy(max_allowed=3),
        truth_trajectory=TruthTrajectory(
            tool_calls=ToolCallRubric(expected_tools=["search_around_poi"])
        ),
    )


def _agent_with_stubs(
    ecr_r=None, ts_r=None, ifs_r=None, iisr_r=None,
    ecr_raises=None, ts_raises=None, ifs_raises=None,
    iisr_raises=None,
) -> tuple[JudgeAgent, dict[str, _StubJudge]]:
    stubs = {
        "ecr":  _StubJudge(ecr_r, ecr_raises),
        "ts":   _StubJudge(ts_r, ts_raises),
        "ifs":  _StubJudge(ifs_r, ifs_raises),
        "iisr": _StubJudge(iisr_r, iisr_raises),
    }
    agent = JudgeAgent(
        llm_provider=None,  # type: ignore[arg-type]  unused — stubs short-circuit
        search_client=None,  # type: ignore[arg-type]
        ecr_judge=stubs["ecr"],  # type: ignore[arg-type]
        ts_judge=stubs["ts"],  # type: ignore[arg-type]
        ifs_judge=stubs["ifs"],  # type: ignore[arg-type]
        iisr_judge=stubs["iisr"],  # type: ignore[arg-type]
        enable_verification=False,
        enable_meta_judge=False,
    )
    return agent, stubs


def _ok_results():
    """A set of all-judges-succeed return values."""
    return dict(
        ecr_r=ExplicitIntentSummary(
            total_count=1, success_count=1, success_intents=["找川菜馆"],
        ),
        ts_r=TSStandaloneResult(
            dim3_pass={"search_around_poi"},
            tool_call_summary=ToolCallSummary(
                correct_calls=[
                    ToolCallDetail(tool_name="search_around_poi",
                                   parameters={"keyword": "川菜"})
                ],
            ),
            judgments=[{"tool_name": "search_around_poi", "dim3_pass": True}],
        ),
        ifs_r=IFSStandaloneResult(
            fact_summary=FactSummary(),
            rubric_row_judgments=[],
        ),
        iisr_r=[
            ImplicitJudgment(
                rubric_instruction="附近",
                satisfaction_score_Ci=1.0,
                confidence_Wi=0.8,
            )
        ],
    )


def test_all_judges_ok_fills_every_slot():
    agent, stubs = _agent_with_stubs(**_ok_results())
    result = asyncio.run(agent.evaluate(
        case_id="c1", query="q", full_intent="q",
        current_time="", current_location="",
        conversation_history=[{"role": "user", "content": "q"}],
        ground_truth=_gt(),
    ))
    # every judge ran exactly once
    assert all(s.calls == 1 for s in stubs.values())
    # judge_status all ok
    assert result.judge_status == {
        "ecr": "ok", "ts": "ok", "ifs": "ok",
        "iisr": "ok",
    }
    # each verdict slot populated from its corresponding judge
    v = result.verdict
    assert v.explicit_intent_summary.success_count == 1
    assert v.tool_call_summary.correct_calls[0].tool_name == "search_around_poi"
    assert v.implicit_intent_judgments[0].satisfaction_score_Ci == 1.0
    # IISR slot exists top-level on scores
    assert result.scores.IISR is not None
    assert 0.0 <= result.scores.IISR <= 1.0


def test_one_judge_returns_none_keeps_others_running():
    """ECR returning None should not abort TS/IFS/IISR."""
    ok = _ok_results()
    ok["ecr_r"] = None  # simulate ECR parse failure
    agent, stubs = _agent_with_stubs(**ok)
    result = asyncio.run(agent.evaluate(
        case_id="c1", query="q", full_intent="q",
        current_time="", current_location="",
        conversation_history=[{"role": "user", "content": "q"}],
        ground_truth=_gt(),
    ))
    # every judge attempted
    assert all(s.calls == 1 for s in stubs.values())
    # ECR failed; others ok
    assert result.judge_status["ecr"] == "failed"
    assert result.judge_status["ts"] == "ok"
    assert result.judge_status["ifs"] == "ok"
    assert result.judge_status["iisr"] == "ok"
    # ECR slot fell back to schema default (total_count=0)
    assert result.verdict.explicit_intent_summary.total_count == 0


def test_one_judge_raises_does_not_abort_batch():
    """IISR raising should be caught by gather(return_exceptions=True)."""
    ok = _ok_results()
    agent, stubs = _agent_with_stubs(
        **{k: v for k, v in ok.items() if k != "iisr_r"},
        iisr_raises=RuntimeError("boom"),
    )
    result = asyncio.run(agent.evaluate(
        case_id="c1", query="q", full_intent="q",
        current_time="", current_location="",
        conversation_history=[{"role": "user", "content": "q"}],
        ground_truth=_gt(),
    ))
    assert result.judge_status["iisr"] == "failed"
    # other slots still came through
    assert result.judge_status["ecr"] == "ok"
    assert result.verdict.tool_call_summary.correct_calls[0].tool_name == \
        "search_around_poi"


def test_iisr_failure_with_empty_gt_marked_as_skip_not_fail():
    """When GT has no implicit_intent and IISR returns None, status is
    'skipped_empty_gt' not 'failed' — vacuous, not an error."""
    gt = _gt()
    # Drop implicit_intent → vacuous
    gt.implicit_intent = []
    ok = _ok_results()
    ok["iisr_r"] = None
    agent, _ = _agent_with_stubs(**ok)
    result = asyncio.run(agent.evaluate(
        case_id="c1", query="q", full_intent="q",
        current_time="", current_location="",
        conversation_history=[{"role": "user", "content": "q"}],
        ground_truth=gt,
    ))
    assert result.judge_status["iisr"] == "skipped_empty_gt"


def test_ts_dim3_pass_drives_metric_correct_calls():
    """When TS returns dim3_pass={search_around_poi}, the gold tool ends
    up in scores.TS's numerator (via the calculator's ts_dim3_pass kwarg)."""
    ok = _ok_results()
    # history that actually has search_around_poi correctly called
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant", "content": "ok",
            "tool_calls": [
                {"name": "search_around_poi",
                 "args": {"keyword": "川菜"},
                 "response": "{}"}
            ],
        },
    ]
    agent, _ = _agent_with_stubs(**ok)
    result = asyncio.run(agent.evaluate(
        case_id="c1", query="q", full_intent="q",
        current_time="", current_location="",
        conversation_history=history,
        ground_truth=_gt(),
    ))
    # search_around_poi recognised → TS Jaccard = 1/1
    assert result.scores.TS == 1.0
