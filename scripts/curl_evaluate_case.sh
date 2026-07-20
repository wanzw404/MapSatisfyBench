#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Single-case evaluation via POST /api/v1/evaluate/case
#
# Payload mirrors SHARED_GT + CASE_1 from scripts/mock_eval_cases.py, so the
# result should be the "happy path" scores (correct tool chain, faithful
# answer, high ICR/TS/IFS/IISR/ASCI).
#
# Prerequisites:
#   1. Start the eval service:
#        uvicorn app.main:app --host 0.0.0.0 --port 8080
#   2. Configure an LLM key on the server side (either AI_STUDIO_TOKEN in the
#      shell env or API_KEY in .env).
#
# Usage:
#   bash scripts/curl_evaluate_case.sh
#   EVAL_SERVICE_HOST=http://remote:8080 bash scripts/curl_evaluate_case.sh
#
# Tips:
#   - Set USE_MOCK_SEARCH=false to hit the real GoogleWebSearch MCP gateway.
#   - Set ENABLE_META_JUDGE=true to additionally run Stage 4 Meta-Judge.
#   - The response is pretty-printed with UTF-8 preserved (no \uXXXX escapes).
# -----------------------------------------------------------------------------
set -euo pipefail

HOST="${EVAL_SERVICE_HOST:-http://localhost:8080}"
URL="${HOST}/api/v1/evaluate/case"

ENABLE_VERIFICATION="${ENABLE_VERIFICATION:-false}"
ENABLE_META_JUDGE="${ENABLE_META_JUDGE:-false}"
LANGUAGE="${LANGUAGE:-chinese}"

echo ">>> POST ${URL}"
echo ">>> enable_verification=${ENABLE_VERIFICATION}  enable_meta_judge=${ENABLE_META_JUDGE}"
echo

curl -sS -X POST "${URL}" \
  -H "Content-Type: application/json" \
  --data @- <<JSON | python3 -m json.tool --no-ensure-ascii
{
  "case_id": "C001_happy_path",
  "query": "推荐附近的川菜馆并告诉我怎么过去",
  "full_intent": "附近的川菜馆，要驾车路线规划",
  "persona": "30岁男性，偏好驾车，餐饮预算60-100元，经常出差",
  "current_time": "2025-05-09 18:30",
  "current_location": "116.397,39.907",
  "language": "${LANGUAGE}",
  "enable_verification": ${ENABLE_VERIFICATION},
  "enable_meta_judge": ${ENABLE_META_JUDGE},
  "ground_truth": {
    "explicit_intent": [
      "POI类型为川菜馆",
      "在用户认为的'附近'范围内",
      "提供路线规划方案"
    ],
    "implicit_intent": [
      {
        "rubric_instruction": "附近的合理解释范围为步行15分钟(1km)到驾车10分钟(5km)",
        "constraint_type": "soft",
        "evidence_confidence": 0.85,
        "evidence": "近30天驾车出行占比82%，公交出行仅3次，结合出行方式偏好驾车，附近可扩展至3-5km",
        "evidence_source": "long_term"
      },
      {
        "rubric_instruction": "餐饮消费水平偏好人均60-100元的餐厅",
        "constraint_type": "soft",
        "evidence_confidence": 0.72,
        "evidence": "近3月餐饮消费中位数78元",
        "evidence_source": "long_term"
      },
      {
        "rubric_instruction": "若推荐驾车导航，必须推荐有停车场的餐厅",
        "constraint_type": "hard",
        "evidence_confidence": 0.68,
        "evidence": "近3月驾车到达的餐厅中85%有停车场",
        "evidence_source": "long_term"
      }
    ],
    "clarification_policy": {
      "max_allowed": 3,
      "evidence": "主动挖掘用户对于餐厅的偏好会增加1轮，推荐完询问用户选择增加1轮，出行方式偏好的挖掘"
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
              "location": "搜索中心点坐标；应使用context.user_current_loc的'lng,lat'字符串；来源是context"
            }
          },
          "driving_route": {
            "paramter": [],
            "rules": {
              "origin": "起点坐标；应使用context.user_current_loc的'lng,lat'字符串；来源是context",
              "destination": "终点坐标；应来自上游poi_search返回的候选POI坐标，形如'lng,lat'；来源是上游工具返回"
            }
          }
        },
        "factual_answer_rubric": [
          "餐厅属性必须与poi_search返回结果一致",
          "最终推荐的主餐厅名称必须是 poi_search 返回的 POI 之一",
          "导航时间/距离必须来自driving_route返回结果",
          "不得编造营业时间（若工具未返则回答中不得出现具体营业时间）",
          "不得虚构停车费用（可以说'停车信息不详'）"
        ]
      }
    }
  },
  "session_stats": {
    "total_input_tokens": 520,
    "total_output_tokens": 285
  },
  "conversation_history_messages": [
    {
      "role": "user",
      "content": "推荐附近的川菜馆并告诉我怎么过去",
      "turn_index": 1
    },
    {
      "role": "assistant",
      "content": "推荐**山东海鲜饺子馆**，距离约10.8km，驾车约102分钟，人均85元，店内没有停车场。",
      "turn_index": 1,
      "TTFT": 205,
      "tool_calls": [
        {
          "id": "t1",
          "name": "poi_search",
          "args": {
            "keyword": "鲁菜"
          },
          "response": {
            "pois": [
              {"id": "B001", "name": "山东海鲜饺子馆", "distance": 10800,
               "avg_price": 85, "has_parking": false,
               "location": "116.412,39.910"},
              {"id": "B002", "name": "粤菜", "distance": 21000,
               "avg_price": 75, "has_parking": false,
               "location": "116.402,39.895"}
            ]
          }
        },
        {
          "id": "t2",
          "name": "driving_route",
          "args": {
            "origin": "116.397,39.907",
            "destination": "116.412,39.910"
          },
          "response": {
            "duration_min": 12,
            "distance_km": 2.8,
            "steps": [{"instruction": "沿东长安街行驶..."}]
          }
        }
      ]
    }
  ]
}
JSON
