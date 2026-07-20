"""Meta-Judge — Stage 4.5 of the JudgeAgent pipeline.

Audits the first-pass JudgeVerdict and emits a *structured correction list*.
The corrections are then applied by ``VerdictPatcher`` against a strict
allowlist of paths, **before** the deterministic metric calculator runs.

This means corrections directly modify the numerator / denominator of
metrics (no post-hoc multiplicative weight), which is the explicit user
preference for this design.

Pipeline position::

    Stage 3: FactVerifier (web search) → fills verified_ok
    Stage 4: MetaJudge.audit() → list[Correction]            ◀── HERE
             VerdictPatcher.apply() → patched JudgeVerdict
    Stage 5: MetricCalculator.compute(patched) → MetricScores
"""

from __future__ import annotations

import copy  # noqa: F401  (kept for forward use; safe to drop later)
import json
import logging
import re
from typing import Any, Optional

from app.core.agent.llm.base import BaseLLMProvider
from app.core.evaluation.conversation_data_normalizer import normalize_conversation_history
from app.core.evaluation.prompt_loader import MetaJudgePromptLoader
from app.core.evaluation.schema import (
    Correction,
    FactStatement,
    GroundTruth,
    ImplicitJudgment,
    JudgeVerdict,
    MetaJudgeReport,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Allowlist — paths the Patcher is permitted to mutate.
# Each entry maps a regex pattern → tuple(allowed_ops).
# ---------------------------------------------------------------------------

ALLOWED_PATHS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    # fact flips on existing items
    (re.compile(r"^fact_summary\.faithful_facts\[(\d+)\]\.faithful$"), ("set",)),
    (re.compile(r"^fact_summary\.faithful_facts\[(\d+)\]\.verified_ok$"), ("set",)),
    (re.compile(r"^fact_summary\.unfaithful_facts\[(\d+)\]\.faithful$"), ("set",)),
    (re.compile(r"^fact_summary\.unfaithful_facts\[(\d+)\]\.verified_ok$"), ("set",)),
    # explicit intent list mutations
    (re.compile(r"^explicit_intent_summary\.success_intents$"), ("add", "remove")),
    (re.compile(r"^explicit_intent_summary\.failed_intents$"), ("add", "remove")),
    # tool call list mutations
    (re.compile(r"^tool_call_summary\.missed_calls$"), ("add", "remove")),
    (re.compile(r"^tool_call_summary\.extra_calls$"), ("add", "remove")),
    # implicit intent score / hard-constraint flag
    (
        re.compile(r"^implicit_intent_judgments\[(\d+)\]\.satisfaction_score_Ci$"),
        ("set",),
    ),
    (
        re.compile(r"^implicit_intent_judgments\[(\d+)\]\.violates_hard_constraint$"),
        ("set",),
    ),
    # clarification turns counter
    (re.compile(r"^clarification_turns$"), ("set",)),
]

# Hard cap — drop anything beyond this; protects against runaway audits.
MAX_CORRECTIONS = 8

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


# ---------------------------------------------------------------------------
# MetaJudge — LLM caller + report parser
# ---------------------------------------------------------------------------


class MetaJudge:
    """Runs the Devil's Advocate audit prompt and returns a parsed report."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        *,
        prompt_loader: Optional[MetaJudgePromptLoader] = None,
        language: str = "chinese",
    ) -> None:
        self.llm = llm_provider
        self.prompt_loader = prompt_loader or MetaJudgePromptLoader(language=language)

    async def audit(
        self,
        *,
        verdict: JudgeVerdict,
        ground_truth: GroundTruth,
        conversation_history: list[dict[str, Any]],
    ) -> MetaJudgeReport:
        """Call the LLM with the Devil's Advocate prompt; return parsed report.

        On any failure (LLM error, unparseable JSON) returns a neutral empty
        report so the caller can keep going with the un-audited verdict.
        """
        prompt = self.prompt_loader.render(
            ground_truth_json=ground_truth.model_dump_json(),
            conversation_history_json=json.dumps(
                normalize_conversation_history(conversation_history),
                ensure_ascii=False,
            ),
            judge_results_json=verdict.model_dump_json(),
            factual_answer_rubric_json=json.dumps(
                ground_truth.truth_trajectory.tool_calls.factual_answer_rubric,
                ensure_ascii=False,
            ),
        )

        try:
            response = await self.llm.achat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "请输出 corrections JSON。"},
                ]
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("MetaJudge LLM call failed: %s", exc)
            return MetaJudgeReport(summary=f"audit skipped: LLM error ({exc})")

        return self._parse_report(response.content or "")

    @staticmethod
    def _parse_report(raw: str) -> MetaJudgeReport:
        text = _FENCE_RE.sub("", (raw or "").strip()).strip()
        if not text.startswith("{"):
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("MetaJudge: unparseable JSON (%s): %r", exc, raw[:200])
            return MetaJudgeReport(summary=f"audit skipped: unparseable JSON ({exc})")

        try:
            report = MetaJudgeReport.model_validate(payload)
        except Exception as exc:  # pydantic ValidationError
            logger.warning("MetaJudge: schema mismatch (%s)", exc)
            return MetaJudgeReport(summary=f"audit skipped: schema mismatch ({exc})")

        # Cap correction list defensively.
        if len(report.corrections) > MAX_CORRECTIONS:
            severity_rank = {"critical": 0, "major": 1, "minor": 2}
            report.corrections.sort(
                key=lambda c: severity_rank.get(c.severity, 3)
            )
            report.corrections = report.corrections[:MAX_CORRECTIONS]
        return report


# ---------------------------------------------------------------------------
# VerdictPatcher — applies allowlisted corrections in-place on a copy
# ---------------------------------------------------------------------------


class VerdictPatcher:
    """Applies a list of ``Correction`` objects to a JudgeVerdict.

    Corrections that target a non-allowlisted path or use a non-allowed op
    are rejected (``correction.applied = False``, ``apply_error`` set) but
    do not raise — they get logged and skipped so a partial audit still
    helps.
    """

    def apply(
        self,
        verdict: JudgeVerdict,
        corrections: list[Correction],
    ) -> tuple[JudgeVerdict, list[Correction]]:
        """Return (patched_verdict, annotated_corrections).

        The original ``verdict`` is **not** mutated; a deep copy is patched.
        Each correction in the returned list has ``applied`` and (on
        rejection) ``apply_error`` populated.
        """
        patched = verdict.model_copy(deep=True)
        annotated: list[Correction] = []

        for original in corrections:
            corr = original.model_copy(deep=True)
            try:
                self._apply_one(patched, corr)
                corr.applied = True
            except _PatchRejected as exc:
                corr.applied = False
                corr.apply_error = str(exc)
                logger.info(
                    "VerdictPatcher rejected correction id=%s path=%r: %s",
                    corr.correction_id,
                    corr.target_path,
                    exc,
                )
            except Exception as exc:  # pragma: no cover - defensive
                corr.applied = False
                corr.apply_error = f"unexpected error: {exc}"
                logger.warning(
                    "VerdictPatcher unexpected error on path=%r: %s",
                    corr.target_path,
                    exc,
                )
            annotated.append(corr)

        # Re-sync derived counters that the prompt is forbidden to touch.
        self._resync_counters(patched)
        return patched, annotated

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _apply_one(self, verdict: JudgeVerdict, corr: Correction) -> None:
        op = corr.operation
        path = corr.target_path

        match = self._match_allowed(path, op)
        if match is None:
            raise _PatchRejected(
                f"path/op not in allowlist (path={path!r}, op={op!r})"
            )
        pattern, indices = match

        # Dispatch by path family
        if pattern.pattern.startswith("^fact_summary\\.faithful_facts"):
            self._patch_fact(
                verdict.fact_summary.faithful_facts, indices[0], path, corr
            )
        elif pattern.pattern.startswith("^fact_summary\\.unfaithful_facts"):
            self._patch_fact(
                verdict.fact_summary.unfaithful_facts, indices[0], path, corr
            )
        elif path == "explicit_intent_summary.success_intents":
            self._patch_string_list(
                verdict.explicit_intent_summary.success_intents, corr
            )
        elif path == "explicit_intent_summary.failed_intents":
            self._patch_string_list(
                verdict.explicit_intent_summary.failed_intents, corr
            )
        elif path == "tool_call_summary.missed_calls":
            self._patch_string_list(verdict.tool_call_summary.missed_calls, corr)
        elif path == "tool_call_summary.extra_calls":
            self._patch_string_list(verdict.tool_call_summary.extra_calls, corr)
        elif pattern.pattern.startswith("^implicit_intent_judgments"):
            self._patch_implicit(verdict.implicit_intent_judgments, indices[0], path, corr)
        elif path == "clarification_turns":
            value = corr.new_value
            if not isinstance(value, int) or value < 0:
                raise _PatchRejected(
                    f"clarification_turns expects non-negative int, got {value!r}"
                )
            verdict.clarification_turns = value
        else:  # pragma: no cover - defensive (allowlist already matched)
            raise _PatchRejected(f"no dispatcher for path {path!r}")

    @staticmethod
    def _match_allowed(
        path: str, op: str
    ) -> Optional[tuple[re.Pattern[str], tuple[str, ...]]]:
        for pattern, ops in ALLOWED_PATHS:
            m = pattern.match(path)
            if m:
                if op not in ops:
                    raise _PatchRejected(
                        f"op {op!r} not allowed for path {path!r} (allowed: {ops})"
                    )
                return pattern, m.groups()
        return None

    @staticmethod
    def _patch_fact(facts: list[FactStatement], index_str: str, path: str, corr: Correction) -> None:
        try:
            idx = int(index_str)
        except (TypeError, ValueError):
            raise _PatchRejected(f"non-integer index in path {path!r}")
        if idx < 0 or idx >= len(facts):
            raise _PatchRejected(
                f"fact index {idx} out of range (len={len(facts)})"
            )
        fact = facts[idx]
        if path.endswith(".faithful"):
            if not isinstance(corr.new_value, bool):
                raise _PatchRejected(f"faithful expects bool, got {corr.new_value!r}")
            fact.faithful = corr.new_value
            if corr.new_value is False and not fact.reason:
                fact.reason = "verified_wrong"
        elif path.endswith(".verified_ok"):
            value = corr.new_value
            if value is not None and not isinstance(value, bool):
                raise _PatchRejected(
                    f"verified_ok expects bool|null, got {value!r}"
                )
            fact.verified_ok = value
            if value is False:
                fact.faithful = False
                if not fact.reason:
                    fact.reason = "verified_wrong"
        else:  # pragma: no cover
            raise _PatchRejected(f"unsupported fact field in path {path!r}")
        # Annotate verify_reason so the audit trail is visible downstream.
        fact.verify_reason = (
            f"meta_judge: {corr.attack_type}/{corr.severity} — {corr.reason}"
        )

    @staticmethod
    def _patch_string_list(target: list[str], corr: Correction) -> None:
        value = corr.new_value
        if not isinstance(value, str) or not value.strip():
            raise _PatchRejected(
                f"list-mutation expects non-empty string value, got {value!r}"
            )
        value = value.strip()
        if corr.operation == "add":
            if value not in target:
                target.append(value)
        elif corr.operation == "remove":
            try:
                target.remove(value)
            except ValueError:
                raise _PatchRejected(f"value {value!r} not present, nothing to remove")
        else:  # pragma: no cover - allowlist guards op
            raise _PatchRejected(f"op {corr.operation!r} unsupported on string list")

    @staticmethod
    def _patch_implicit(
        judgments: list[ImplicitJudgment], index_str: str, path: str, corr: Correction
    ) -> None:
        try:
            idx = int(index_str)
        except (TypeError, ValueError):
            raise _PatchRejected(f"non-integer index in path {path!r}")
        if idx < 0 or idx >= len(judgments):
            raise _PatchRejected(
                f"implicit index {idx} out of range (len={len(judgments)})"
            )
        item = judgments[idx]
        if path.endswith(".satisfaction_score_Ci"):
            value = corr.new_value
            if not isinstance(value, (int, float)):
                raise _PatchRejected(
                    f"satisfaction_score_Ci expects float, got {value!r}"
                )
            value = max(0.0, min(1.0, float(value)))
            item.satisfaction_score_Ci = value
        elif path.endswith(".violates_hard_constraint"):
            if not isinstance(corr.new_value, bool):
                raise _PatchRejected(
                    f"violates_hard_constraint expects bool, got {corr.new_value!r}"
                )
            item.violates_hard_constraint = corr.new_value
        else:  # pragma: no cover
            raise _PatchRejected(f"unsupported implicit field in path {path!r}")

    @staticmethod
    def _resync_counters(verdict: JudgeVerdict) -> None:
        """Recompute counters that the LLM is forbidden to touch directly."""
        # explicit_intent_summary counters — must reflect post-patch state in
        # both directions. Using ``max`` here (previous behaviour) made the
        # denominator monotonic, so removing a hallucinated ``failed_intent``
        # left ECR's denominator stale and the score artificially low.
        s = verdict.explicit_intent_summary
        s.success_count = len(s.success_intents)
        s.total_count = s.success_count + len(s.failed_intents)

        # fact_summary.faithful_facts_count reflects current list length.
        verdict.fact_summary.faithful_facts_count = len(
            verdict.fact_summary.faithful_facts
        )


# ---------------------------------------------------------------------------
# Internal sentinel — soft rejection, do not propagate as exception
# ---------------------------------------------------------------------------


class _PatchRejected(Exception):
    """Raised internally to signal an allowlist / value-shape rejection."""
