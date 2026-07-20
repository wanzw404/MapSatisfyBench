"""IFSJudge._parse regression tests — no LLM, no network."""

from __future__ import annotations

import json

from app.core.evaluation.ifs_judge import IFSJudge


def _payload(reasons: list) -> str:
    """Wrap a list of element-reason strings into a parseable IFS reply."""
    return json.dumps(
        {
            "rubric_row_judgments": [
                {
                    "rubric_row": "测试 rubric",
                    "elements": [
                        {"element": f"e{i}", "grounded": False, "reason": r}
                        for i, r in enumerate(reasons)
                    ],
                    "score": 0,
                    "reasoning": "",
                }
            ]
        },
        ensure_ascii=False,
    )


def test_parse_normalizes_camelcase_and_aliases() -> None:
    """LLM 返回大小写不规范 / 同义别名时 _parse 不应炸 schema。"""
    raw = _payload(
        [
            "no_tool_groundING",         # camelcase 反例（用户实测炸点）
            "NO_TOOL_GROUNDING",         # 全大写
            " absent ",                  # 带空格
            "Contradict",                # 同义别名
            "verify_failed",             # 同义别名
        ]
    )
    res = IFSJudge._parse(raw)
    assert res is not None, "归一化后不应再触发 schema mismatch"
    elements = res.rubric_row_judgments[0].elements
    assert [e.reason for e in elements] == [
        "no_tool_grounding",
        "no_tool_grounding",
        "absent",
        "contradicted",
        "external_verify_failed",
    ]


def test_parse_unknown_reason_becomes_none() -> None:
    """无法对齐到 Literal 的写 None，不阻断整个 case。"""
    raw = _payload(["totally_made_up_reason"])
    res = IFSJudge._parse(raw)
    assert res is not None
    assert res.rubric_row_judgments[0].elements[0].reason is None
