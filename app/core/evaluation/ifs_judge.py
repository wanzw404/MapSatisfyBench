"""Standalone IFS (dim3) judge — physically-isolated input.

Why this exists
---------------
The first-pass JudgeAgent prompt bundles all dimensions into one LLM
call. Although the bundled prompt declares "factual_answer_rubric only
serves dim3", the LLM still physically sees ``explicit_intent`` /
``implicit_intent`` / ``truth_trajectory.tool_calls.parameter_rules``
in the same prompt — cross-dimension contamination is suspected to
make IFS judging unstable (mirrors the IISR motivation in
:mod:`iisr_judge`).

This module provides a parallel, physically-isolated dim3 LLM call.
The prompt only carries:

* ``current_time`` / ``current_location``
* the **annotated** conversation history (each tool response carries
  an injected ``_classification = {type, is_empty, reason}`` produced
  by :func:`tool_response_utils.annotate_tool_responses` — IFS needs
  to see tool messages, unlike IISR which only sees assistant turns)
* ``factual_answer_rubric`` (the dim3-only rubric list)

Tools schema is intentionally **not** fed to IFS — every ``tool_calls``
entry in the conversation already carries its real ``response`` JSON,
so the schema (a "tool definition manual") is redundant for grounded /
absent / contradicted judgements. Schema is a TS / parameter-validation
concern, not an IFS concern.

On success the result overrides the batch verdict's ``fact_summary``
+ ``rubric_row_judgments`` slots. Failure mode mirrors IISRJudge: any
LLM / JSON / schema error returns ``None`` so the caller falls back to
the batch dim3 output. Critically, the override **must** happen before
:meth:`FactVerifier.verify_rubric_rows` runs, because the verifier
writes ``external_verified_ok`` / ``grounded`` / ``skipped`` in-place
on whichever ``RubricRowJudgment`` instances are in the verdict.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from app.core.agent.llm.base import BaseLLMProvider
from app.core.evaluation.prompt_loader import IFSPromptLoader
from app.core.evaluation.schema import FactSummary, RubricRowJudgment

if TYPE_CHECKING:
    from app.core.evaluation.judge_inputs import JudgeInputs

logger = logging.getLogger(__name__)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class IFSStandaloneResult:
    """Lightweight container for the two verdict slots the IFS judge fills.

    Returned by :meth:`IFSJudge.evaluate` on success; the caller copies
    these onto :class:`JudgeVerdict` (``fact_summary`` and
    ``rubric_row_judgments``) before the verifier runs.
    """

    fact_summary: FactSummary
    rubric_row_judgments: list[RubricRowJudgment]


class IFSJudge:
    """Standalone IFS (dim3) judge with isolated input.

    Only the conversation (with ``_classification`` annotations on each
    tool response), the ``factual_answer_rubric`` list, the tool schema,
    and the time/location context reach the prompt. The rest of the
    ground truth (explicit/implicit intents, parameter_rules, etc.) is
    never exposed to the dim3 LLM.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        *,
        prompt_loader: Optional[IFSPromptLoader] = None,
        language: str = "chinese",
    ) -> None:
        self.llm = llm_provider
        self.prompt_loader = prompt_loader or IFSPromptLoader(language=language)

    async def evaluate(
        self, inputs: "JudgeInputs"
    ) -> Optional[IFSStandaloneResult]:
        """Run the standalone IFS LLM call; return parsed dim3 output.

        Returns ``None`` on any LLM / JSON / schema error so the caller
        records ``judge_status["ifs"] = "failed"`` and falls back to
        schema defaults (empty rubric_row_judgments → IFS = 0/0 = 0;
        details should distinguish vacuous 0 from real 0).

        User messages are stripped from the prompt history: IFS scores
        assistant claims against tool returns, user turns aren't on
        either side of that comparison. Keeping them let the LLM
        misinterpret "AI 助手的最终回复" as "the assistant turn after
        the last user message" — when the trajectory ends on a user
        follow-up, that anchor doesn't exist and every rubric element
        gets marked ``absent`` → empty-satisfaction false-pass.
        Dropping user turns removes the ambiguous anchor entirely.
        Rubric semantics ("若最终答案给出X") + ``factual_answer_rubric``
        already encode what to look for, so user phrasing is not
        load-bearing for IFS.
        """
        ifs_history = [
            m for m in inputs.annotated_history if m.get("role") != "user"
        ]
        prompt = self.prompt_loader.render(
            current_time=inputs.current_time,
            current_location=inputs.current_location,
            conversation_history_json=json.dumps(
                ifs_history, ensure_ascii=False
            ),
            factual_answer_rubric_json=json.dumps(
                inputs.factual_rubric, ensure_ascii=False
            ),
        )

        try:
            response = await self.llm.achat(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": "请按要求输出 IFS JSON。",
                    },
                ]
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("IFSJudge LLM call failed: %s", exc)
            return None

        return self._parse(response.content or "")

    # ``RubricElementEvidence.reason`` 是 Literal[...]，LLM 偶发返回带不规范
    # 大小写（"no_tool_groundING"）/ 同义别名，会被 pydantic 拒掉整行→整个
    # IFS judge fail。规范成 schema 接受的字面量后再喂校验；无法对齐的
    # 写 None（语义：未给原因，不阻断流程）。
    _REASON_NORMALIZE = {
        "absent": "absent",
        "no_tool_grounding": "no_tool_grounding",
        "no_tool_ground": "no_tool_grounding",
        "no_grounding": "no_tool_grounding",
        "contradicted": "contradicted",
        "contradict": "contradicted",
        "external_verify_failed": "external_verify_failed",
        "external_verification_failed": "external_verify_failed",
        "verify_failed": "external_verify_failed",
    }

    @classmethod
    def _normalize_reasons(cls, rows_raw: list) -> None:
        """In-place 把每个 element.reason 规范化成 schema 允许的字面量。"""
        for row in rows_raw:
            if not isinstance(row, dict):
                continue
            for el in row.get("elements") or []:
                if not isinstance(el, dict):
                    continue
                r = el.get("reason")
                if not isinstance(r, str):
                    continue
                key = r.strip().lower()
                el["reason"] = cls._REASON_NORMALIZE.get(key)  # None if unknown

    @staticmethod
    def _parse(raw: str) -> Optional[IFSStandaloneResult]:
        """Parse the IFS standalone reply.

        Only ``rubric_row_judgments`` (or its alias ``rows_detail`` — the
        key name used by the current prompt template) is required; new
        IFS is purely row-level. ``fact_summary`` is legacy and treated
        as optional — when omitted we plug in an empty placeholder so
        the verdict slot stays well-typed. Returns ``None`` only when
        the JSON itself is unparseable, the rubric list is missing /
        non-list, or any row fails pydantic validation.
        """
        text = _FENCE_RE.sub("", (raw or "").strip()).strip()
        if not text.startswith("{"):
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "IFSJudge: unparseable JSON (%s): %r", exc, raw[:200]
            )
            return None

        # Accept both the canonical key and the prompt-template alias.
        rows_raw = payload.get("rubric_row_judgments")
        if rows_raw is None:
            rows_raw = payload.get("rows_detail")
        if not isinstance(rows_raw, list):
            logger.warning(
                "IFSJudge: missing or non-list 'rubric_row_judgments' / "
                "'rows_detail'"
            )
            return None

        IFSJudge._normalize_reasons(rows_raw)

        # fact_summary is optional — new IFS is row-level only. When the
        # prompt omits it we use an empty placeholder; the metric layer
        # ignores fact_summary whenever rubric_row_judgments is non-empty.
        fact_summary_raw = payload.get("fact_summary")
        try:
            fact_summary = (
                FactSummary.model_validate(fact_summary_raw)
                if isinstance(fact_summary_raw, dict)
                else FactSummary()
            )
            rubric_rows = [RubricRowJudgment.model_validate(r) for r in rows_raw]
        except Exception as exc:  # pydantic ValidationError
            logger.warning("IFSJudge: schema mismatch (%s)", exc)
            return None

        return IFSStandaloneResult(
            fact_summary=fact_summary,
            rubric_row_judgments=rubric_rows,
        )
