"""Load judge prompt files from YAML with language / template support.

The evaluation pipeline runs 4 physically-isolated LLM judges in parallel
(ECR / TS / IFS / IISR). Each has its own segment-file directory
under ``app/core/agent/prompt/`` and its own placeholder allowlist:

* ``judge_ecr/`` — :class:`ECRPromptLoader`; placeholders
  :data:`ECR_PLACEHOLDERS`. Dim1 only, sees explicit_intent + final answer.
* ``judge_ts/`` — :class:`TSPromptLoader`; placeholders
  :data:`TS_PLACEHOLDERS`. Dim2 dim3 only — parameter shape / traceability
  for candidates that already passed code-side dim1+dim2 filters.
* ``judge_ifs/`` — :class:`IFSPromptLoader`; placeholders
  :data:`IFS_PLACEHOLDERS`. Dim3 only, sees annotated conversation +
  factual_answer_rubric.
* ``judge_iisr/`` — :class:`IISRPromptLoader`; placeholders
  :data:`IISR_PLACEHOLDERS`. Dim4 only, sees implicit_intent + assistant
  turn contents (no tools / no other GT).

Plus the Meta-Judge single-file prompt:

* ``check_jude_propmt.yaml`` — :class:`MetaJudgePromptLoader`; placeholders
  :data:`META_JUDGE_PLACEHOLDERS`.

All templates use a ``chinese: |`` / ``english: |`` block-scalar layout.
The loaders share placeholder-substitution logic via the internal
``_PromptRenderer`` base and segment-directory glob logic via
``_SegmentDirPromptLoader``. Each segment-directory loader globs
``[0-9]*.yaml`` in lexicographic order and concatenates per-language
blocks into a single prompt string — keep numeric prefixes gapped so
future segments slot in without renames.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any

import yaml

def _prompt_dir(name: str) -> str:
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "agent",
            "prompt",
            name,
        )
    )


DEFAULT_ECR_PROMPT_DIR = _prompt_dir("judge_ecr")
DEFAULT_TS_PROMPT_DIR = _prompt_dir("judge_ts")
DEFAULT_IFS_PROMPT_DIR = _prompt_dir("judge_ifs")
DEFAULT_IISR_PROMPT_DIR = _prompt_dir("judge_iisr")
META_JUDGE_PROMPT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "agent",
        "prompt",
        "check_jude_propmt.yaml",
    )
)

# Placeholders expected by the meta-judge (Devil's Advocate) prompt.
META_JUDGE_PLACEHOLDERS: tuple[str, ...] = (
    "ground_truth_json",
    "conversation_history_json",
    "judge_results_json",
    "factual_answer_rubric_json",
)

# Placeholders expected by the standalone ECR judge prompt (dim1 only).
# Only the assistant final-answer content list + explicit_intent rubric
# reach the LLM — no tool returns, no implicit intent, no rubric for
# factual answer (those belong to dims 2/3/4).
ECR_PLACEHOLDERS: tuple[str, ...] = (
    "full_intent",
    "explicit_intent_json",
    "assistant_content_json",
)

# Placeholders expected by the standalone TS judge prompt (dim3 only).
# dim1 (name membership) + dim2 (required params filled) are done by code
# upstream in ``judge_inputs.prepare_judge_inputs``; the LLM sees only
# candidates that already passed both, with their parameter_rules
# attached, and judges shape/type + traceability. ``conversation_history_json``
# is the annotated history so the LLM can do the traceability check.
TS_PLACEHOLDERS: tuple[str, ...] = (
    "current_time",
    "current_location",
    "conversation_history_json",
    "candidate_calls_json",
)

# Placeholders expected by the standalone IFS judge prompt (dim3 only).
# IFS needs the *annotated* conversation history (each tool response
# carries an injected ``_classification`` dict) so the prompt can reason
# about empty / non-empty tool results without re-parsing tool JSON.
#
# 不带 tools_schema_json：IFS 的判定单元是"声明是否被同轮 tool response
# 字段支撑"，conversation_history_json 里每条 tool_call 都已经带 response
# 真值，schema（工具定义说明书）是冗余信息。TS / 参数校验维度才需要 schema，
# 与 IFS 无关。
IFS_PLACEHOLDERS: tuple[str, ...] = (
    "current_time",
    "current_location",
    "conversation_history_json",
    "factual_answer_rubric_json",
)

# Placeholders expected by the standalone IISR judge prompt (dim4 only).
IISR_PLACEHOLDERS: tuple[str, ...] = (
    "current_time",
    "current_location",
    "assistant_turns_json",
    "implicit_intent_json",
)

class _PromptRenderer:
    """Shared placeholder-substitution layer; subclasses supply ``raw``."""

    def __init__(self, raw: str, placeholders: tuple[str, ...]) -> None:
        self._raw = raw
        self._placeholders = tuple(placeholders)
        self._placeholder_re = re.compile(
            r"\{(" + "|".join(self._placeholders) + r")\}"
        )

    def render(self, **context: Any) -> str:
        """Fill named placeholders without touching JSON braces.

        Unlike ``str.format``, this method only substitutes the keys listed
        in ``self._placeholders``. Unknown keys raise ``ValueError``;
        missing required keys default to an empty string (caller is
        responsible for providing all template variables).
        """
        unknown = [k for k in context if k not in self._placeholders]
        if unknown:
            raise ValueError(f"Unknown placeholder(s) passed to render: {unknown}")

        def _sub(match: re.Match[str]) -> str:
            name = match.group(1)
            value = context.get(name, "")
            return "" if value is None else str(value)

        return self._placeholder_re.sub(_sub, self._raw)


class _SegmentDirPromptLoader(_PromptRenderer):
    """Glob-and-concat base for segment-file judge prompts.

    Files matching ``[0-9]*.yaml`` are loaded in lexicographic order; each
    file's ``chinese`` / ``english`` block is concatenated into a single
    prompt. The numeric prefix (``00_``, ``10_``, …) is the only ordering
    mechanism — keep gaps between numbers so future segments can slot in
    without renames.

    Not used directly — each of the 5 standalone judges inherits from this
    base and supplies its own default directory + placeholder allowlist.
    """

    def __init__(
        self,
        prompt_dir: str,
        language: str,
        placeholders: tuple[str, ...],
    ) -> None:
        self.prompt_dir = prompt_dir
        self.language = language
        super().__init__(self._assemble(), placeholders)

    def _assemble(self) -> str:
        if not os.path.isdir(self.prompt_dir):
            raise FileNotFoundError(
                f"Judge prompt directory not found: {self.prompt_dir}"
            )
        files = sorted(
            glob.glob(os.path.join(self.prompt_dir, "[0-9]*.yaml"))
        )
        if not files:
            raise FileNotFoundError(
                f"No segment files matching '[0-9]*.yaml' under {self.prompt_dir}"
            )
        parts: list[str] = []
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError(
                    f"Judge prompt segment {path} must be a YAML mapping."
                )
            if self.language not in data:
                raise ValueError(
                    f"Language '{self.language}' missing in {path}. "
                    f"Available: {sorted(k for k in data if isinstance(data[k], str))}"
                )
            raw = data[self.language]
            if not isinstance(raw, str):
                raise ValueError(
                    f"Segment '{self.language}' in {path} must be a string."
                )
            parts.append(raw)
        return "".join(parts)


class ECRPromptLoader(_SegmentDirPromptLoader):
    """Standalone ECR (dim1) prompt loader.

    Defaults to the ``judge_ecr/`` directory and the dim1-only placeholder
    set (``full_intent`` + ``explicit_intent_json`` +
    ``assistant_content_json`` — no tool returns, no rubric).
    """

    def __init__(
        self,
        prompt_dir: str | None = None,
        language: str = "chinese",
        placeholders: tuple[str, ...] = ECR_PLACEHOLDERS,
    ) -> None:
        super().__init__(
            prompt_dir=prompt_dir or DEFAULT_ECR_PROMPT_DIR,
            language=language,
            placeholders=placeholders,
        )


class TSPromptLoader(_SegmentDirPromptLoader):
    """Standalone TS dim3 prompt loader.

    dim1 (name membership) + dim2 (required params) are done by code in
    ``prepare_judge_inputs``; the LLM only sees ``candidate_calls_json``
    (already filtered to dim1∧dim2 pass + deduped by longest-args) plus
    the annotated conversation history for traceability.
    """

    def __init__(
        self,
        prompt_dir: str | None = None,
        language: str = "chinese",
        placeholders: tuple[str, ...] = TS_PLACEHOLDERS,
    ) -> None:
        super().__init__(
            prompt_dir=prompt_dir or DEFAULT_TS_PROMPT_DIR,
            language=language,
            placeholders=placeholders,
        )


class IFSPromptLoader(_SegmentDirPromptLoader):
    """Standalone IFS (dim3) prompt loader.

    Defaults to the ``judge_ifs/`` directory and the dim3-only placeholder
    set.
    """

    def __init__(
        self,
        prompt_dir: str | None = None,
        language: str = "chinese",
        placeholders: tuple[str, ...] = IFS_PLACEHOLDERS,
    ) -> None:
        super().__init__(
            prompt_dir=prompt_dir or DEFAULT_IFS_PROMPT_DIR,
            language=language,
            placeholders=placeholders,
        )


class IISRPromptLoader(_SegmentDirPromptLoader):
    """Standalone IISR (dim4) prompt loader.

    Defaults to the ``judge_iisr/`` directory and the dim4-only placeholder
    set.
    """

    def __init__(
        self,
        prompt_dir: str | None = None,
        language: str = "chinese",
        placeholders: tuple[str, ...] = IISR_PLACEHOLDERS,
    ) -> None:
        super().__init__(
            prompt_dir=prompt_dir or DEFAULT_IISR_PROMPT_DIR,
            language=language,
            placeholders=placeholders,
        )


class MetaJudgePromptLoader(_PromptRenderer):
    """Single-file loader for the meta-judge prompt."""

    def __init__(
        self,
        prompt_path: str | None = None,
        language: str = "chinese",
    ) -> None:
        self.prompt_path = prompt_path or META_JUDGE_PROMPT_PATH
        self.language = language
        super().__init__(self._load(), META_JUDGE_PLACEHOLDERS)

    def _load(self) -> str:
        if not os.path.exists(self.prompt_path):
            raise FileNotFoundError(
                f"Meta-judge prompt file not found: {self.prompt_path}"
            )
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"Meta-judge prompt file {self.prompt_path} must be a YAML mapping."
            )
        if self.language not in data:
            raise ValueError(
                f"Language '{self.language}' missing in {self.prompt_path}. "
                f"Available: {sorted(k for k in data if isinstance(data[k], str))}"
            )
        raw = data[self.language]
        if not isinstance(raw, str):
            raise ValueError(
                f"Meta-judge prompt for language '{self.language}' must be a string."
            )
        return raw
