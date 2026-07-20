"""Mock evaluation runner — real pipeline, only GT + inference as input.

Each case here contains ONLY:
    * ground_truth                 (what the agent *should* produce)
    * conversation_history         (what the agent *did* produce — user/assistant/tool msgs)
    * minimal metadata             (query, persona, location, time)

It then runs every case through the REAL production pipeline via
`build_default_agent()`:

    Stage 2: LLM Judge         (live call to OpenAICompatProvider)
    Stage 3: Fact verification (live GoogleWebSearch MCP calls, optional)
    Stage 4: Meta-Judge audit  (live "Devil's Advocate" LLM call, optional)
    Stage 5: MetricCalculator  (deterministic)

No canned verdicts. No canned audits. The only things this script fakes are
the agent's tool responses inside `conversation_history`, because that's what
"mock inference result" means in the user's brief.

Prerequisites
-------------
    export AI_STUDIO_TOKEN=<your_token>

Usage
-----
    python3 scripts/mock_eval_cases.py                  # all cases, full pipeline
    python3 scripts/mock_eval_cases.py --case 2         # only case 2
    python3 scripts/mock_eval_cases.py --no-meta        # skip Meta-Judge
    python3 scripts/mock_eval_cases.py --no-verify      # skip Stage 3 (fact verify)
    python3 scripts/mock_eval_cases.py --dump outputs/mock_raw.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Allow running from project root without installing as a package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.core.agent.llm.openai_chat import OpenAICompatProvider  # noqa: E402
from app.core.evaluation import (  # noqa: E402
    BatchEvalReport,
    JudgeAgent,
    build_default_agent,
    evaluate_batch,
    evaluate_case,
    format_batch_report,
)
from app.core.evaluation.schema import EvalResult, GroundTruth  # noqa: E402


# =============================================================================
# CASE FIXTURES — ONLY ground_truth + conversation_history + metadata.
# No canned verdicts. No canned audits. The real pipeline will judge these.
# =============================================================================

SHARED_GT: dict[str, Any] = {
    "explicit_intent": [
        "POI类型为川菜馆",
        "在用户认为的'附近'范围内",
        "提供路线规划方案",
    ],
    "implicit_intent": [
        {
            "rubric_instruction": "附近的合理解释范围为步行15分钟(1km)到驾车10分钟(5km)",
            "constraint_type": "soft",
            "evidence_confidence": 0.85,
            "evidence": "近30天驾车出行占比82%，公交出行仅3次，结合出行方式偏好驾车，附近可扩展至3-5km",
            "evidence_source": "long_term",
        },
        {
            "rubric_instruction": "餐饮消费水平偏好人均60-100元的餐厅",
            "constraint_type": "soft",
            "evidence_confidence": 0.72,
            "evidence": "近3月餐饮消费中位数78元",
            "evidence_source": "long_term",
        },
        {
            "rubric_instruction": "若推荐驾车导航，必须推荐有停车场的餐厅",
            "constraint_type": "hard",
            "evidence_confidence": 0.68,
            "evidence": "近3月驾车到达的餐厅中85%有停车场",
            "evidence_source": "long_term",
        },
    ],
    "clarification_policy": {
        "max_allowed": 3,
        "evidence": "主动挖掘用户对于餐厅的偏好会增加1轮，推荐完询问用户选择增加1轮，出行方式偏好的挖掘",
    },
    "truth_trajectory": {
        "tool_calls": {
            "expected_tools": ["poi_search", "driving_route"],
            "parameter_rules": {
                "poi_search": {
                    "paramter": [],
                    "rules": {
                        "keyword": "搜索关键词；应填'川菜'或语义等价的菜系关键词；来源是query与full_intent",
                        "radius": "搜索半径；应填不超过5000米的数值；来源是full_intent中的距离偏好",
                        "location": "搜索中心点坐标；应使用context.user_current_loc的'lng,lat'字符串；来源是context",
                    },
                },
                "driving_route": {
                    "paramter": [],
                    "rules": {
                        "origin": "起点坐标；应使用context.user_current_loc的'lng,lat'字符串；来源是context",
                        "destination": "终点坐标；应来自上游poi_search返回的候选POI坐标，形如'lng,lat'；来源是上游工具返回",
                    },
                },
            },
            "factual_answer_rubric": [
                "餐厅名称必须与poi_search返回结果一致",
                "最终推荐的主餐厅名称必须是 poi_search 返回的 POI 之一",
                "导航时间/距离必须来自driving_route返回结果",
                "不得编造营业时间（若工具未返则回答中不得出现具体营业时间）",
                "不得虚构停车费用（可以说'停车信息不详'）",
            ],
        }
    },
}


# -----------------------------------------------------------------------------
# CASE 1 — Happy path: correct tool chain, faithful answer.
# Uses the BUNDLED format: `conversation_history_messages` with inline tool
# responses + turn_index per message. Normalized inside JudgeAgent.evaluate().
# -----------------------------------------------------------------------------
CASE_1 = {
    "case_id": "C001_happy_path",
    "query": "推荐附近的川菜馆并告诉我怎么过去",
    "full_intent": "附近的川菜馆，要驾车路线规划",
    "persona": "30岁男性，偏好驾车，餐饮预算60-100元，经常出差",
    "current_time": "2025-05-09 18:30",
    "current_location": "116.397,39.907",
    "ground_truth": SHARED_GT,
    "session_stats": {
        "total_input_tokens": 520,
        "total_output_tokens": 285,
    },
    "conversation_history_messages": [
        {
            "role": "user",
            "content": "推荐附近的川菜馆并告诉我怎么过去",
            "turn_index": 1,
        },
        {
            "role": "assistant",
            "content": "推荐**蜀香居川菜馆**，距离约2.8km，驾车约12分钟，人均85元，店内有停车场。",
            "turn_index": 1,
            "TTFT": 180,
            "tool_calls": [
                {
                    "id": "t1",
                    "name": "poi_search",
                    "args": {
                        "keyword": "川菜",
                        "radius": 3000,
                        "location": "116.397,39.907",
                    },
                    "response": {
                        "pois": [
                            {"id": "B001", "name": "蚵香居川菜馆", "distance": 2800,
                             "avg_price": 85, "has_parking": True,
                             "location": "116.412,39.910"},
                            {"id": "B002", "name": "老成都川菜", "distance": 2100,
                             "avg_price": 75, "has_parking": False,
                             "location": "116.402,39.895"},
                        ]
                    },
                },
                {
                    "id": "t2",
                    "name": "driving_route",
                    "args": {
                        "origin": "116.397,39.907",
                        "destination": "116.412,39.910",
                    },
                    "response": {
                        "duration_min": 12,
                        "distance_km": 2.8,
                        "steps": [{"instruction": "沿东长安街行驶..."}],
                    },
                },
            ],
        },
    ],
}


# -----------------------------------------------------------------------------
# CASE 2 — 显式意图不满足 (Explicit Intent Not Met)
# Agent misinterprets the cuisine: searches "附近餐厅" and recommends a 湘菜
# restaurant instead of 川菜. Tool chain is otherwise well-behaved.
# Expected signal: ECR drops (POI类型为川菜馆 = failed intent).
# -----------------------------------------------------------------------------
CASE_2 = {
    "case_id": "C002_explicit_intent_missed",
    "query": "推荐附近的川菜馆并告诉我怎么过去",
    "full_intent": "附近的川菜馆，要驾车路线规划",
    "persona": "30岁男性，偏好驾车，餐饮预算60-100元",
    "current_time": "2025-05-09 18:30",
    "current_location": "116.397,39.907",
    "ground_truth": SHARED_GT,
    "session_stats": {
        "total_input_tokens": 480,
        "total_output_tokens": 310,
    },
    "conversation_history_messages": [
        {
            "role": "user",
            "content": "推荐附近的川菜馆并告诉我怎么过去",
            "turn_index": 1,
        },
        {
            "role": "assistant",
            "content": "为您找到了**辣妹子湘菜馆**（湘菜），距离1.8km，驾车约7分钟，人均68元，店内有停车场。给您导航到店里。",
            "turn_index": 1,
            "TTFT": 220,
            "tool_calls": [
                {
                    "id": "t1",
                    "name": "poi_search",
                    "args": {
                        "keyword": "附近餐厅",
                        "radius": 3000,
                        "location": "116.397,39.907",
                    },
                    "response": {
                        "pois": [
                            {"id": "H001", "name": "辣妹子湘菜馆",
                             "cuisine": "湘菜", "distance": 1800,
                             "avg_price": 68, "has_parking": True,
                             "location": "116.408,39.901"},
                            {"id": "H002", "name": "蜀香居川菜馆",
                             "cuisine": "川菜", "distance": 2800,
                             "avg_price": 85, "has_parking": True,
                             "location": "116.412,39.910"},
                        ]
                    },
                },
                {
                    "id": "t2",
                    "name": "driving_route",
                    "args": {
                        "origin": "116.397,39.907",
                        "destination": "116.408,39.901",
                    },
                    "response": {"duration_min": 7, "distance_km": 1.8},
                },
            ],
        },
    ],
}


# -----------------------------------------------------------------------------
# CASE 3 — 隐式意图不满足 (Implicit Intent / Hard Constraint Violation)
# Agent drives the user to 老成都川菜 — a 川菜馆 (explicit OK) that has NO
# parking. Violates the hard rubric "若推荐驾车导航，必须推荐有停车场的餐厅".
# Expected signal: IISR = 0 (hard-constraint zero-tolerance).
# -----------------------------------------------------------------------------
CASE_3 = {
    "case_id": "C003_implicit_intent_violated",
    "query": "推荐附近的川菜馆并告诉我怎么过去",
    "full_intent": "附近的川菜馆，要驾车路线规划",
    "persona": "30岁男性，偏好驾车，餐饮预算60-100元",
    "current_time": "2025-05-09 18:30",
    "current_location": "116.397,39.907",
    "ground_truth": SHARED_GT,
    "session_stats": {
        "total_input_tokens": 510,
        "total_output_tokens": 260,
    },
    "conversation_history_messages": [
        {
            "role": "user",
            "content": "推荐附近的川菜馆并告诉我怎么过去",
            "turn_index": 1,
        },
        {
            "role": "assistant",
            "content": "推荐**老成都川菜**，距离2.1km，驾车约8分钟，人均75元。为您开启导航。",
            "turn_index": 1,
            "TTFT": 195,
            "tool_calls": [
                {
                    "id": "t1",
                    "name": "poi_search",
                    "args": {
                        "keyword": "川菜",
                        "radius": 3000,
                        "location": "116.397,39.907",
                    },
                    "response": {
                        "pois": [
                            {"id": "B001", "name": "蜀香居川菜馆",
                             "cuisine": "川菜", "distance": 2800,
                             "avg_price": 85, "has_parking": True,
                             "location": "116.412,39.910"},
                            {"id": "B002", "name": "老成都川菜",
                             "cuisine": "川菜", "distance": 2100,
                             "avg_price": 75, "has_parking": False,
                             "location": "116.402,39.895"},
                        ]
                    },
                },
                {
                    "id": "t2",
                    "name": "driving_route",
                    "args": {
                        "origin": "116.397,39.907",
                        "destination": "116.402,39.895",
                    },
                    "response": {"duration_min": 8, "distance_km": 2.1},
                },
            ],
        },
    ],
}


# -----------------------------------------------------------------------------
# CASE 4 — 工具调用名称错误 (Wrong Tool Name)
# Expected tools are [poi_search, driving_route]. Agent substitutes
# `search_restaurant` — a tool NOT in the expected set — for poi_search.
# Expected signal: TS drops (missed=poi_search, extra=search_restaurant).
# -----------------------------------------------------------------------------
CASE_4 = {
    "case_id": "C004_wrong_tool_name",
    "query": "推荐附近的川菜馆并告诉我怎么过去",
    "full_intent": "附近的川菜馆，要驾车路线规划",
    "persona": "30岁男性，偏好驾车，餐饮预算60-100元",
    "current_time": "2025-05-09 18:30",
    "current_location": "116.397,39.907",
    "ground_truth": SHARED_GT,
    "session_stats": {
        "total_input_tokens": 450,
        "total_output_tokens": 240,
    },
    "conversation_history_messages": [
        {
            "role": "user",
            "content": "推荐附近的川菜馆并告诉我怎么过去",
            "turn_index": 1,
        },
        {
            "role": "assistant",
            "content": "推荐**蜀香居川菜馆**，距离约2.5km，驾车10分钟，人均80元，店内有停车场。",
            "turn_index": 1,
            "TTFT": 250,
            "tool_calls": [
                {
                    "id": "t1",
                    "name": "search_restaurant",
                    "args": {"cuisine": "川菜", "within_km": 3},
                    "response": {
                        "results": [
                            {"id": "B001", "name": "蜀香居川菜馆",
                             "cuisine": "川菜", "distance": 2500,
                             "avg_price": 80, "has_parking": True,
                             "location": "116.412,39.910"},
                        ]
                    },
                },
                {
                    "id": "t2",
                    "name": "driving_route",
                    "args": {
                        "origin": "116.397,39.907",
                        "destination": "116.412,39.910",
                    },
                    "response": {"duration_min": 10, "distance_km": 2.5},
                },
            ],
        },
    ],
}


# -----------------------------------------------------------------------------
# CASE 5 — 工具调用参数错误 (Wrong Tool Parameters)
# Tool names are correct (poi_search + driving_route), but parameters violate
# the GT parameter_rules:
#   - radius=15000 (> 5000 limit)
#   - location="120.220,30.253" (Hangzhou — does NOT match user_current_loc)
#   - driving_route.origin uses the same bad coord
# Returned POI is geographically irrelevant.
# Expected signal: judge should mark these as incorrect_calls → TS penalty.
# -----------------------------------------------------------------------------
CASE_5 = {
    "case_id": "C005_wrong_tool_params",
    "query": "推荐附近的川菜馆并告诉我怎么过去",
    "full_intent": "附近的川菜馆，要驾车路线规划",
    "persona": "30岁男性，偏好驾车，餐饮预算60-100元",
    "current_time": "2025-05-09 18:30",
    "current_location": "116.397,39.907",
    "ground_truth": SHARED_GT,
    "session_stats": {
        "total_input_tokens": 490,
        "total_output_tokens": 300,
    },
    "conversation_history_messages": [
        {
            "role": "user",
            "content": "推荐附近的川菜馆并告诉我怎么过去",
            "turn_index": 1,
        },
        {
            "role": "assistant",
            "content": "推荐**成都老宅川菜（朝阳门店）**，距离约9.2km，驾车约22分钟，人均90元，有停车场。已为您规划路线。",
            "turn_index": 1,
            "TTFT": 310,
            "tool_calls": [
                {
                    "id": "t1",
                    "name": "poi_search",
                    "args": {
                        "keyword": "川菜",
                        "radius": 15000,
                        "location": "120.220,30.253",
                    },
                    "response": {
                        "pois": [
                            {"id": "B101", "name": "成都老宅川菜（朝阳门店）",
                             "cuisine": "川菜", "distance": 9200,
                             "avg_price": 90, "has_parking": True,
                             "location": "116.490,39.935"},
                        ]
                    },
                },
                {
                    "id": "t2",
                    "name": "driving_route",
                    "args": {
                        "origin": "120.220,30.253",
                        "destination": "116.490,39.935",
                    },
                    "response": {"duration_min": 22, "distance_km": 9.2},
                },
            ],
        },
    ],
}


# -----------------------------------------------------------------------------
# CASE 6 — 工具调用结果与关键事实陈述点不同 (Claim vs Tool-Result Mismatch)
# Tool names and parameters are perfect. Tool responses are truthful. But the
# agent's natural-language answer contradicts them on FOUR specific facts:
#   • distance: claims 1.2km — tool says 2.8km
#   • duration: claims 5min   — tool says 12min
#   • avg_price: claims 50元  — tool says 85元
#   • store name detail: claims "非常划算" — not supported by tool output
# Expected signal: IFS drops (multiple unfaithful_facts). IISR unaffected
# (the recommended POI still has parking → hard constraint OK).
# -----------------------------------------------------------------------------
CASE_6 = {
    "case_id": "C006_claim_vs_tool_mismatch",
    "query": "推荐附近的川菜馆并告诉我怎么过去",
    "full_intent": "附近的川菜馆，要驾车路线规划",
    "persona": "30岁男性，偏好驾车，餐饮预算60-100元",
    "current_time": "2025-05-09 18:30",
    "current_location": "116.397,39.907",
    "ground_truth": SHARED_GT,
    "session_stats": {
        "total_input_tokens": 530,
        "total_output_tokens": 275,
    },
    "conversation_history_messages": [
        {
            "role": "user",
            "content": "推荐附近的川菜馆并告诉我怎么过去",
            "turn_index": 1,
        },
        {
            "role": "assistant",
            "content": "推荐**蜀香居川菜馆**，距离只有1.2km，驾车约5分钟即达，人均仅50元，非常划算！店内停车方便。",
            "turn_index": 1,
            "TTFT": 165,
            "tool_calls": [
                {
                    "id": "t1",
                    "name": "poi_search",
                    "args": {
                        "keyword": "川菜",
                        "radius": 3000,
                        "location": "116.397,39.907",
                    },
                    "response": {
                        "pois": [
                            {"id": "B001", "name": "蜀香居川菜馆",
                             "cuisine": "川菜", "distance": 2800,
                             "avg_price": 85, "has_parking": True,
                             "location": "116.412,39.910"},
                        ]
                    },
                },
                {
                    "id": "t2",
                    "name": "driving_route",
                    "args": {
                        "origin": "116.397,39.907",
                        "destination": "116.412,39.910",
                    },
                    "response": {"duration_min": 12, "distance_km": 2.8},
                },
            ],
        },
    ],
}


CASES: list[dict[str, Any]] = [CASE_1, CASE_2, CASE_3, CASE_4, CASE_5, CASE_6]


# =============================================================================
# Real LLM provider builder (mirrors app/api/routes/evaluate.py:_build_llm_provider)
# =============================================================================


def build_llm_provider(model_override: str = "") -> OpenAICompatProvider:
    """Instantiate the LLM provider from env / settings."""
    from app.config import JUDGE_MODEL

    base_url = settings.BASE_URL
    api_key = os.environ.get("AI_STUDIO_TOKEN", "") or settings.AI_STUDIO_TOKEN
    if api_key in ("", "0", "your-api-key"):
        raise RuntimeError(
            "No valid API key. Set AI_STUDIO_TOKEN in .env"
        )
    # 这是 judge fixture 验证脚本，与生产 evaluate_service 同样**强制锁定**
    # 到 JUDGE_MODEL，避免开发本地输出与线上不一致。--model CLI 仍可调试
    # 用，但会打 warning。
    if model_override and model_override != JUDGE_MODEL:
        print(
            f"[mock_eval_cases] WARNING: model_override={model_override!r} 与 "
            f"JUDGE_MODEL={JUDGE_MODEL!r} 不一致；按 override 跑（仅本脚本）"
        )
        model = model_override
    else:
        model = JUDGE_MODEL
    return OpenAICompatProvider(
        base_url=base_url,
        api_key=api_key,
        model=model,
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY") or None,
        whale_api_key=os.environ.get("WHALE_API_KEY") or None,
        session_id=os.environ.get("LLM_SESSION_ID") or None,
    )


# =============================================================================
# Runner
# =============================================================================


async def run_case(
    case: dict[str, Any],
    *,
    agent: JudgeAgent,
) -> EvalResult:
    """Drive a single case through the facade :func:`evaluate_case`.

    The agent is built once by the caller and reused across every case in
    the run so we do not repeatedly instantiate LLM / search clients.
    """
    return await evaluate_case(case, agent=agent)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _fmt_scores(scores: Any) -> str:
    """Per-case metric breakdown (six independent metrics + runtime stats)."""
    rows = [
        ("ECR  (Explicit-decision-factor Completion Rate)", scores.ECR),
        ("TS   (Tool-call Success, Jaccard)", scores.TS),
        ("IFS  (Information Faithfulness Score)", scores.IFS),
        ("AR   (Accepted-response Probability = ECR · IISR)", scores.AR),
        ("Eff  (Interaction Efficiency)", scores.Eff),
    ]
    out = ["    metric                              score",
           "    " + "-" * 44]
    for name, value in rows:
        formatted = f"{value:5.3f}" if value is not None else "  N/A"
        out.append(f"    {name:<36}  {formatted}")
    out.append(f"    {'avg_ttft_ms':<36}  {scores.avg_ttft_ms:>7.2f}")
    out.append(f"    {'total_tokens':<36}  {scores.total_tokens:>5d}")
    return "\n".join(out)


def _print_case_header(case: dict[str, Any]) -> None:
    print("=" * 88)
    print(f"CASE  {case['case_id']}")
    print(f"QUERY {case['query']}")
    print(f"GT explicit_intents: {case['ground_truth']['explicit_intent']}")
    history = (
        case.get("conversation_history_messages")
        or case.get("conversation_history")
        or []
    )
    # Bundled format: tool_calls carry a top-level "name". Flat format: each
    # tool_call is {"id", "function": {"name": ...}}.
    tool_calls: list[str] = []
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = tc.get("name") or (tc.get("function") or {}).get("name", "")
            if name:
                tool_calls.append(name)
    print(f"Agent tool calls actually made: {tool_calls}")
    final = next(
        (m.get("content", "") for m in reversed(history)
         if m.get("role") == "assistant" and m.get("content")),
        "",
    )
    print(f"Agent final reply: {final}")
    print("-" * 88)


def _print_aggregate(
    report: BatchEvalReport,
    *,
    ttft_ms_per_case: list[float] | None = None,
    tokens_per_case: list[int] | None = None,
) -> None:
    """Delegate to :class:`BatchEvalReport.format` from the facade.

    The batch statistic for each metric is the per-case arithmetic mean
    (plus cheap dispersion diagnostics). No pass/fail threshold is applied.
    """
    if report.n_cases == 0:
        print("  (no cases to aggregate)")
        return

    print(report.format())

    # Per-case raw values table
    print()
    print(
        f"  {'case_id':<36} {'ECR':>5} {'TS':>5} {'IFS':>5} "
        f"{'AR':>5} {'Eff':>5} {'TTFT':>7} {'Tokens':>6}"
    )
    print("  " + "-" * 80)
    _ttft = ttft_ms_per_case or []
    _tok = tokens_per_case or []
    for i, r in enumerate(report.results):
        s = r.scores
        ttft = _ttft[i] if i < len(_ttft) else 0.0
        tok = _tok[i] if i < len(_tok) else 0
        print(
            f"  {r.case_id:<36} "
            f"{s.ECR:>5.2f} {s.TS:>5.2f} {s.IFS:>5.2f} "
            f"{s.AR:>5.2f} {(f'{s.Eff:>5.2f}' if s.Eff is not None else ' N/A '):>5} "
            f"{ttft:>7.1f} {tok:>6d}"
        )


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=int, default=0, help="run only case N (1-based); 0=all")
    ap.add_argument("--no-meta", action="store_true", help="disable Stage 4 Meta-Judge")
    ap.add_argument("--no-verify", action="store_true", help="disable Stage 3 fact verify")
    ap.add_argument("--language", default="chinese", choices=["chinese", "english"])
    ap.add_argument("--model", default="", help="LLM model override")
    ap.add_argument("--dump", type=str, default="",
                    help="dump full EvalResult JSON to this path")
    args = ap.parse_args()

    llm_provider = build_llm_provider(model_override=args.model)
    print(f"LLM provider: {type(llm_provider).__name__}  model={llm_provider.model}")
    print(f"Stages:  judge=ON  verify={'OFF' if args.no_verify else 'ON'}  "
          f"meta_judge={'OFF' if args.no_meta else 'ON'}  "
          f"search=GoogleWebSearch(MCP)")
    print()

    selected = [CASES[args.case - 1]] if 1 <= args.case <= len(CASES) else CASES

    # Build the JudgeAgent once and reuse it for every case — this mirrors
    # how the facade :func:`evaluate_batch` operates internally.
    agent: JudgeAgent = build_default_agent(
        llm_provider,
        language=args.language,
        enable_meta_judge=not args.no_meta,
    )
    agent.enable_verification = not args.no_verify

    dumped: list[dict[str, Any]] = []
    # results_or_none 与 selected 同长，失败位 None — 让下面的均值口径与
    # 生产路径 (evaluate_batch) 完全一致：分母 = 总 case 数，失败按 0 计入。
    results_or_none: list[Optional[EvalResult]] = []
    failures: list[tuple[str, str]] = []

    for case in selected:
        _print_case_header(case)
        try:
            result = await run_case(case, agent=agent)
        except Exception as exc:
            print(f"  !! pipeline failed: {type(exc).__name__}: {exc}")
            print()
            results_or_none.append(None)
            failures.append((str(case.get("case_id") or "?"), f"{type(exc).__name__}: {exc}"))
            continue

        print("Scores:")
        print(_fmt_scores(result.scores))
        print()

        results_or_none.append(result)
        if args.dump:
            dumped.append(json.loads(result.model_dump_json()))

    if args.dump and dumped:
        out = Path(args.dump)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dumped, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Dumped {len(dumped)} EvalResult(s) to {out}")

    # Build a BatchEvalReport using the same denominator policy as
    # `evaluate_batch`: failures injected as zero-score placeholders, but
    # runtime stats (TTFT / tokens) stay survivor-only.
    from app.core.evaluation.metrics_summary import aggregate_batch, zero_metric_scores
    successful = [r for r in results_or_none if r is not None]
    scores_for_stats = [
        (r.scores if r is not None else zero_metric_scores())
        for r in results_or_none
    ]
    verdicts_for_stats = [
        (r.verdict if r is not None else None) for r in results_or_none
    ]
    ttft_ms_per_case = [r.scores.avg_ttft_ms for r in successful]
    tokens_per_case = [r.scores.total_tokens for r in successful]
    batch_stats = aggregate_batch(
        scores=scores_for_stats,
        verdicts=verdicts_for_stats,
        ttft_ms_per_case=ttft_ms_per_case,
        tokens_per_case=tokens_per_case,
    )
    batch_report = BatchEvalReport(
        results=successful, stats=batch_stats, failures=failures,
    )

    print("=" * 88)
    print(" AGGREGATE SUMMARY")
    print("=" * 88)
    _print_aggregate(batch_report, ttft_ms_per_case=ttft_ms_per_case, tokens_per_case=tokens_per_case)
    print()
    print("=" * 88)
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
