"""Standalone ECR (dim1) judge — physically-isolated input.

Why this exists
---------------
ECR (Explicit-decision-factor Completion Rate) only needs ``full_intent`` to
disambiguate intent terms, the ``explicit_intent`` rubric list, and the
assistant's textual replies. The bundled batch judge fed it tool
returns, parameter rules and the implicit rubric in the same prompt —
which pulled it toward penalising tool grounding / hidden preferences
(those belong to dim2/dim3/dim4).

This judge sees **only** those three slices and emits an
:class:`ExplicitIntentSummary` slot for the verdict. Failure mode
mirrors :class:`IISRJudge` / :class:`IFSJudge`: any LLM / JSON / schema
error returns ``None`` so the caller can record ``judge_status["ecr"]
= "failed"`` and fall back to the schema default (which is the
benefit-of-the-doubt 0/0 — see :class:`ExplicitIntentSummary` doc).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from app.core.agent.llm.base import BaseLLMProvider
from app.core.evaluation.judge_inputs import JudgeInputs
from app.core.evaluation.prompt_loader import ECRPromptLoader
from app.core.evaluation.schema import ExplicitIntentSummary

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ECRJudge:
    """Standalone ECR (dim1) judge with isolated input."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        *,
        prompt_loader: Optional[ECRPromptLoader] = None,
        language: str = "chinese",
    ) -> None:
        self.llm = llm_provider
        self.prompt_loader = prompt_loader or ECRPromptLoader(language=language)

    async def evaluate(
        self, inputs: JudgeInputs
    ) -> Optional[ExplicitIntentSummary]:
        gt = inputs.ground_truth
        # No explicit intent → metric layer treats as vacuous (ECR=1.0);
        # firing the LLM only burns tokens. Return a default summary so
        # the caller registers "ok".
        if not gt.explicit_intent:
            return ExplicitIntentSummary()

        assistant_contents = [
            m.get("content", "")
            for m in inputs.normalized_history
            if m.get("role") == "assistant" and m.get("content")
        ]

        prompt = self.prompt_loader.render(
            full_intent=inputs.full_intent,
            explicit_intent_json=json.dumps(
                list(gt.explicit_intent), ensure_ascii=False
            ),
            assistant_content_json=json.dumps(
                assistant_contents, ensure_ascii=False
            ),
        )

        try:
            response = await self.llm.achat(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": "请按要求输出 ECR 评测 JSON。",
                    },
                ]
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ECRJudge LLM call failed: %s", exc)
            return None

        return self._parse(response.content or "", gt.explicit_intent)

    @staticmethod
    def _parse(
        raw: str, gt_explicit: list[str]
    ) -> Optional[ExplicitIntentSummary]:
        text = _FENCE_RE.sub("", (raw or "").strip()).strip()
        if not text.startswith("{"):
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "ECRJudge: unparseable JSON (%s): %r", exc, raw[:200]
            )
            return None
        if not isinstance(payload, dict):
            logger.warning("ECRJudge: top-level JSON is not an object")
            return None

        # Prefer the explicit summary block; fall back to deriving from
        # evaluation_results if the LLM omits it (some templates drift).
        summary_raw = payload.get("summary")
        results_raw = payload.get("evaluation_results") or []

        try:
            if isinstance(summary_raw, dict):
                summary = ExplicitIntentSummary(
                    total_count=int(summary_raw.get("total_intents")
                                    or summary_raw.get("total_count") or 0),
                    success_count=int(summary_raw.get("success_count") or 0),
                    success_intents=list(summary_raw.get("success_intents") or []),
                    failed_intents=list(summary_raw.get("failed_intents") or []),
                )
            elif isinstance(results_raw, list):
                success_intents: list[str] = []
                failed_intents: list[str] = []
                for r in results_raw:
                    if not isinstance(r, dict):
                        continue
                    intent = str(r.get("intent", ""))
                    score = str(r.get("score", "")).lower()
                    if score == "success":
                        success_intents.append(intent)
                    else:
                        failed_intents.append(intent)
                summary = ExplicitIntentSummary(
                    total_count=len(success_intents) + len(failed_intents),
                    success_count=len(success_intents),
                    success_intents=success_intents,
                    failed_intents=failed_intents,
                )
            else:
                logger.warning(
                    "ECRJudge: missing both 'summary' and 'evaluation_results'"
                )
                return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ECRJudge: schema mismatch (%s)", exc)
            return None

        # Clamp total_count to at least the ground truth size so the
        # metric layer doesn't accidentally count "success_count > total"
        # when the LLM under-reports total.
        if summary.total_count < len(gt_explicit):
            summary.total_count = len(gt_explicit)
        return summary
