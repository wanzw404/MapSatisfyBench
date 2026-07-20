"""Standalone TS dim3 judge — physically-isolated input.

TS (Tool Selection) is split across code and LLM responsibilities:

* **dim1 — name membership** (code, in :func:`prepare_judge_inputs`):
  ``tool_name ∈ ground_truth.truth_trajectory.tool_calls.expected_tools``
* **dim2 — required params filled** (code, same place):
  ``_args_have_all(args, _required_params(gt, name))``
* **dim3 — parameter shape + traceability** (LLM, this judge):
  every value matches the natural-language ``rules`` description, and
  every value can be traced back to the conversation context (no
  fabricated coordinates / POI IDs / etc.).

Only candidates that have already passed dim1 ∧ dim2 are sent to the
LLM (``inputs.ts_candidates_for_llm``). The dim3 verdict is intersected
with ``inputs.ts_dim12_pass`` on return so any tool name the LLM
hallucinates can't leak into the downstream Jaccard.

This judge also **synthesises** a :class:`ToolCallSummary` for the
verdict — explainer / meta_judge / fact_verifier still read
``verdict.tool_call_summary`` and they should not need to learn about
the new dim3 plumbing. ``correct_calls[i].parameters`` carries the
dedup'd longest-args dict so downstream parameter audits see real
values, not empty placeholders.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from app.core.agent.llm.base import BaseLLMProvider
from app.core.evaluation.judge_inputs import JudgeInputs
from app.core.evaluation.metrics.ts import _required_params
from app.core.evaluation.prompt_loader import TSPromptLoader
from app.core.evaluation.schema import (
    IncorrectToolCall,
    ToolCallDetail,
    ToolCallSummary,
)

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class TSStandaloneResult:
    """What the TS judge hands back to ``JudgeAgent.evaluate``.

    * ``dim3_pass`` — the set passed to :func:`metrics.ts.compute_ts`
      via the ``ts_dim3_pass`` kwarg; controls which gold tools end up
      in TS's numerator.
    * ``tool_call_summary`` — the synthesised slot that lands on
      :class:`JudgeVerdict.tool_call_summary` so explainer /
      meta_judge / fact_verifier see a populated summary even after the
      bundled judge is gone.
    * ``judgments`` — raw per-tool LLM output (kept for audit /
      explainer hooks).
    """

    dim3_pass: set[str]
    tool_call_summary: ToolCallSummary
    judgments: list[dict[str, Any]]


class TSJudge:
    """Standalone TS dim3 judge with isolated input."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        *,
        prompt_loader: Optional[TSPromptLoader] = None,
        language: str = "chinese",
    ) -> None:
        self.llm = llm_provider
        self.prompt_loader = prompt_loader or TSPromptLoader(language=language)

    async def evaluate(
        self, inputs: JudgeInputs
    ) -> Optional[TSStandaloneResult]:
        candidates = inputs.ts_candidates_for_llm
        # No candidates passed dim1 ∧ dim2 — nothing for the LLM to look
        # at. Return immediately with an empty dim3_pass: code-side
        # filters have already condemned every actual call. (Note: this
        # is the *correct* semantic — the LLM doesn't get to rescue
        # tools that failed dim1/dim2.)
        if not candidates:
            return TSStandaloneResult(
                dim3_pass=set(),
                tool_call_summary=self._synthesise_summary(
                    inputs, dim3_pass=set(), llm_reasons={}
                ),
                judgments=[],
            )

        prompt = self.prompt_loader.render(
            current_time=inputs.current_time,
            current_location=inputs.current_location,
            conversation_history_json=json.dumps(
                inputs.annotated_history, ensure_ascii=False
            ),
            candidate_calls_json=json.dumps(candidates, ensure_ascii=False),
        )

        try:
            response = await self.llm.achat(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": "请按要求输出 dim3_judgments JSON。",
                    },
                ]
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("TSJudge LLM call failed: %s", exc)
            return None

        parsed = self._parse(response.content or "")
        if parsed is None:
            return None

        # Clamp the LLM's dim3_pass to the dim1+dim2 candidate set so a
        # hallucinated tool name can't sneak into the metric layer.
        dim3_pass = {
            name for name in parsed["dim3_pass"]
            if name in inputs.ts_dim12_pass
        }
        llm_reasons = parsed["reasons"]
        return TSStandaloneResult(
            dim3_pass=dim3_pass,
            tool_call_summary=self._synthesise_summary(
                inputs, dim3_pass=dim3_pass, llm_reasons=llm_reasons
            ),
            judgments=parsed["judgments"],
        )

    @staticmethod
    def _parse(raw: str) -> Optional[dict[str, Any]]:
        text = _FENCE_RE.sub("", (raw or "").strip()).strip()
        if not text.startswith("{"):
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("TSJudge: unparseable JSON (%s): %r", exc, raw[:200])
            return None
        if not isinstance(payload, dict):
            logger.warning("TSJudge: top-level JSON is not an object")
            return None

        items = payload.get("dim3_judgments")
        if not isinstance(items, list):
            logger.warning(
                "TSJudge: missing or non-list 'dim3_judgments'"
            )
            return None

        dim3_pass: set[str] = set()
        reasons: dict[str, str] = {}
        judgments: list[dict[str, Any]] = []
        for j in items:
            if not isinstance(j, dict):
                continue
            name = j.get("tool_name")
            if not isinstance(name, str) or not name:
                continue
            passed = bool(j.get("dim3_pass"))
            reason = str(j.get("reason") or "")
            judgments.append(
                {"tool_name": name, "dim3_pass": passed, "reason": reason}
            )
            if passed:
                dim3_pass.add(name)
            else:
                reasons[name] = reason
        return {
            "dim3_pass": dim3_pass,
            "reasons": reasons,
            "judgments": judgments,
        }

    # ------------------------------------------------------------------
    # tool_call_summary synthesis (for explainer / meta_judge / verifier)
    # ------------------------------------------------------------------

    @staticmethod
    def _synthesise_summary(
        inputs: JudgeInputs,
        *,
        dim3_pass: set[str],
        llm_reasons: dict[str, str],
    ) -> ToolCallSummary:
        """Build a ToolCallSummary from code + LLM signals combined.

        Rules:
        * ``correct_calls``: dim1 ∧ dim2 ∧ dim3 — emit ``ToolCallDetail``
          with the dedup'd longest-args dict (parsed if JSON-string).
        * ``incorrect_calls``: gold tool that was called but failed
          dim2 or dim3. ``reason`` records which check failed.
        * ``extra_calls``: predicted names not in gold.
        * ``missed_calls``: gold names that never produced a correct
          call (i.e. either never invoked, or every invocation failed
          dim2/dim3).
        """
        gt = inputs.ground_truth
        gold = set(gt.truth_trajectory.tool_calls.expected_tools or [])
        deduped = inputs.ts_deduped_args
        pred_names = set(deduped.keys())

        correct: list[ToolCallDetail] = []
        incorrect: list[IncorrectToolCall] = []
        correct_names: set[str] = set()

        for name in sorted(pred_names):
            if name not in gold:
                continue  # extra; handled below
            args = deduped[name]
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    if isinstance(parsed, dict):
                        args = parsed
                except (TypeError, ValueError):
                    pass
            args_dict = args if isinstance(args, dict) else {}

            dim2_ok = name in inputs.ts_dim12_pass
            dim3_ok = name in dim3_pass

            if dim2_ok and dim3_ok:
                correct.append(
                    ToolCallDetail(tool_name=name, parameters=args_dict)
                )
                correct_names.add(name)
                continue

            if not dim2_ok:
                missing = [
                    p for p in _required_params(gt, name)
                    if p not in args_dict
                ]
                reason = (
                    f"missing_required_params: {','.join(missing)}"
                    if missing else "dim2_required_params_invalid"
                )
            else:
                reason = (
                    f"dim3_param_shape_invalid: {llm_reasons.get(name, '')}"
                    .rstrip(": ")
                )
            incorrect.append(IncorrectToolCall(tool_name=name, reason=reason))

        extra = sorted(pred_names - gold)
        missed = sorted(gold - correct_names)

        return ToolCallSummary(
            correct_calls=correct,
            incorrect_calls=incorrect,
            extra_calls=extra,
            missed_calls=missed,
        )
