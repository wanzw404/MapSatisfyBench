"""Standalone IISR (dim4) judge — physically-isolated input.

Why this exists
---------------
The first-pass JudgeAgent prompt bundles all 5 dimensions into one LLM
call. Even though dim4 (Implicit Intent / IISR) declares "strict
isolation — only consult `implicit_intent`", the LLM physically sees
``factual_answer_rubric`` / ``explicit_intent`` / ``truth_trajectory``
in the same prompt, and cross-dimension contamination can lower IISR
scoring quality (observed: 0519eval IISR within_acc = 0.625 vs other
dims ≥ 0.95).

This module provides a parallel, physically-isolated dim4 LLM call:
the prompt only carries ``current_time`` / ``current_location`` /
extracted assistant turns / the ``implicit_intent`` rubric list — no
other GT fields are available even to look at. The result overrides the
batch's ``implicit_intent_judgments`` slot in :class:`JudgeVerdict`.

Failure mode mirrors :class:`MetaJudge.audit`: any LLM / parse error
returns ``None`` so the caller can fall back to the batch's dim4 output.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

from app.core.agent.llm.base import BaseLLMProvider
from app.core.evaluation.prompt_loader import IISRPromptLoader
from app.core.evaluation.schema import ImplicitJudgment

if TYPE_CHECKING:
    from app.core.evaluation.judge_inputs import JudgeInputs

logger = logging.getLogger(__name__)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class IISRJudge:
    """Standalone IISR (dim4) judge with isolated input.

    The prompt receives only ``current_time``, ``current_location``,
    a chronological list of assistant turn contents, and the
    ``implicit_intent`` rubric list — nothing else.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        *,
        prompt_loader: Optional[IISRPromptLoader] = None,
        language: str = "chinese",
    ) -> None:
        self.llm = llm_provider
        self.prompt_loader = prompt_loader or IISRPromptLoader(language=language)

    async def evaluate(
        self, inputs: "JudgeInputs"
    ) -> Optional[list[ImplicitJudgment]]:
        """Run the standalone IISR LLM call; return parsed judgments.

        Returns ``None`` on any LLM / JSON / schema error so the caller
        records ``judge_status["iisr"] = "failed"`` and falls back to
        the schema default ``implicit_intent_judgments=[]``.
        """
        implicit_intent = inputs.ground_truth.implicit_intent
        prompt = self.prompt_loader.render(
            current_time=inputs.current_time,
            current_location=inputs.current_location,
            assistant_turns_json=json.dumps(
                inputs.assistant_turns, ensure_ascii=False
            ),
            implicit_intent_json=json.dumps(
                [ii.model_dump() for ii in implicit_intent],
                ensure_ascii=False,
            ),
        )

        try:
            response = await self.llm.achat(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": "请按要求输出 implicit_intent_judgments JSON。",
                    },
                ]
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("IISRJudge LLM call failed: %s", exc)
            return None

        return self._parse_judgments(response.content or "")

    @staticmethod
    def _extract_assistant_turns(
        history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Pull out the chronological list of assistant ``content`` strings.

        Skips assistant messages that carry only ``tool_calls`` and no
        textual content — those don't constitute an answer to the user.
        """
        out: list[dict[str, Any]] = []
        turn = 0
        for msg in history:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not content:  # skip empty / tool-call-only turns
                continue
            turn += 1
            out.append({"turn": turn, "content": content})
        return out

    @staticmethod
    def _parse_judgments(raw: str) -> Optional[list[ImplicitJudgment]]:
        text = _FENCE_RE.sub("", (raw or "").strip()).strip()
        if not text.startswith("{"):
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "IISRJudge: unparseable JSON (%s): %r", exc, raw[:200]
            )
            return None

        items = payload.get("implicit_intent_judgments") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            logger.warning(
                "IISRJudge: missing or non-list 'implicit_intent_judgments'; "
                "raw=%r",
                raw[:300],
            )
            return None
        try:
            return [ImplicitJudgment.model_validate(j) for j in items]
        except Exception as exc:  # pydantic ValidationError
            logger.warning("IISRJudge: schema mismatch (%s)", exc)
            return None
