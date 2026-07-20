"""Tool response classification helpers — IFS / FactVerifier 共用。

`conversation_history_messages` 中的 `tool_calls[*].response` 字段在我们
的数据格式里是 **字符串形式**（已 JSON-stringified）。形态有几种：

  1. 空 ``tool_calls`` 列表 — assistant 直接答没调工具::

       "tool_calls": []

  2. ``response`` 字段缺失 / 为空字符串 — 调用未返回。

  3. ``response`` 是 JSON 字符串但表达"空 / 错误"语义::

       {"status": "0", "info": "USER_DAILY_QUERY_OVER_LIMIT"}     # API 失败
       {"status": "1", "count": "0", "info": "OK", "pois": []}    # 调成功但无结果
       {"results": []}                                             # 网络搜索无结果
       {"buslines": []} / {"routes": []} / {"districts": []}      # 其它列表字段为空

  4. ``response`` 是描述性错误字符串（非 JSON）::

       "get_navigation 需要提供 start_lon, start_lat, end_lon, end_lat 参数"
       "无mock数据"

  5. ``response`` 是 JSON 且含真实数据 — **唯一不算"空"的情况**::

       {"status": "1", "count": "1", "pois": [{"name": "...", ...}]}
       {"results": [{"title": "...", "content": "..."}]}

本模块两个核心函数：

* :func:`is_tool_response_empty` — 一个 bool，给最常用的"工具是否给出有效
  数据"判定用。
* :func:`classify_tool_response` — 详细分类（含 type / reason / parsed），
  给 audit / 调试 / verifier 决策用。
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 高德 / 网络搜索 / Whale 常见的"业务数据列表"键名。空数组即视为"无结果"。
_LIST_FIELDS: tuple[str, ...] = (
    "pois",
    "results",
    "routes",
    "buslines",
    "districts",
    "regeocodes",
    "geocodes",
    "tips",
    "items",
    "data",
)

# 表示工具自身报失败的 status 取值（高德 / OpenAI 兼容层常见）。
_FAIL_STATUS_VALUES: frozenset = frozenset(
    {"0", 0, 0.0, "false", "False", "error", "fail", "failed"}
)

# 文本里出现这些关键词，视为"工具自报错误"（而非真正返回数据的描述性文本）。
_ERROR_KEYWORDS: tuple[str, ...] = (
    "需要提供",
    "无mock数据",
    "Error",
    "error",
    "失败",
    "未找到",
    "缺少必要参数",
    "参数错误",
    "invalid",
    "Invalid",
    "not found",
    "Not Found",
)

# 字段值表示"无信息"（用在判定关键字段是否填充）。
_EMPTY_VALUES = (None, "", "无", "暂无", "N/A", "n/a", "null", "None")


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _try_parse_json(s: Any) -> tuple[bool, Any]:
    """尝试把字符串解析成 JSON。返回 (是否成功, 结果)。"""
    if not isinstance(s, str):
        return False, s
    text = s.strip()
    if not text:
        return False, None
    if not (text.startswith("{") or text.startswith("[")):
        return False, text
    try:
        return True, json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False, text


def _is_text_an_error(text: str) -> bool:
    """非 JSON 字符串里是否含明显错误关键词。"""
    if not text:
        return False
    return any(kw in text for kw in _ERROR_KEYWORDS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_tool_response_empty(response_str: Any) -> bool:
    """判断 ``response`` 是否表达"空 / 错误"语义。

    True 的情况（即"无工具依据"）：
      - 缺失 / None / 空字符串
      - 非 JSON 但含错误关键词（"需要提供"、"无mock数据" 等）
      - JSON 但 status 失败 / count=0 / 列表字段为空 / 全字段都是空值
    False 的情况：
      - JSON 含真实数据（pois/results 非空数组等）
      - 非 JSON 但是普通业务文本（无错误关键词）
    """
    if response_str is None:
        return True
    if not isinstance(response_str, str):
        # 非字符串视为有内容（比如已是 dict 的少见情况）
        return False
    text = response_str.strip()
    if not text or text.lower() in ("null", "none", "undefined"):
        return True

    parsed_ok, obj = _try_parse_json(text)

    # ── 非 JSON 字符串 ──────────────────────────────────────────────
    if not parsed_ok:
        return _is_text_an_error(text)

    # ── JSON dict 形态 ─────────────────────────────────────────────
    if isinstance(obj, dict):
        # 1. status 表示失败
        if obj.get("status") in _FAIL_STATUS_VALUES:
            return True
        # 2. count 是 0
        if obj.get("count") in (0, "0", 0.0):
            return True
        # 3. 任一列表字段是空数组
        for key in _LIST_FIELDS:
            v = obj.get(key)
            if isinstance(v, list) and len(v) == 0:
                return True
        # 4. 全字段值都是空（status/count/info 都没给有效信息时）
        if obj and all(v in _EMPTY_VALUES for v in obj.values()):
            return True
        return False

    # ── JSON list 形态 ─────────────────────────────────────────────
    if isinstance(obj, list):
        return len(obj) == 0

    # 其它 JSON 标量（罕见）— 保守判非空
    return False


def classify_tool_response(response_str: Any) -> dict[str, Any]:
    """详细分类，给 audit / verifier 决策用。

    返回 dict::

      {
        "type":   "missing" | "empty_string" | "api_failure" | "empty_results"
                | "error_string" | "text_data" | "has_data" | "empty_list"
                | "non_string" | "unknown",
        "reason": str,        # 一句话理由，便于审计
        "parsed": Any,        # 成功 parse 后的对象；否则原始字符串
        "is_empty": bool,     # 等价于 is_tool_response_empty(response_str)
      }
    """
    if response_str is None:
        return {
            "type": "missing", "reason": "无 response 字段",
            "parsed": None, "is_empty": True,
        }
    if not isinstance(response_str, str):
        return {
            "type": "non_string",
            "reason": f"response 类型为 {type(response_str).__name__}",
            "parsed": response_str, "is_empty": False,
        }

    text = response_str.strip()
    if not text or text.lower() in ("null", "none", "undefined"):
        return {
            "type": "empty_string", "reason": "response 为空 / null",
            "parsed": None, "is_empty": True,
        }

    parsed_ok, obj = _try_parse_json(text)

    if not parsed_ok:
        if _is_text_an_error(text):
            return {
                "type": "error_string",
                "reason": text[:200],
                "parsed": text, "is_empty": True,
            }
        return {
            "type": "text_data",
            "reason": "非 JSON 文本（无错误关键词）",
            "parsed": text, "is_empty": False,
        }

    if isinstance(obj, dict):
        if obj.get("status") in _FAIL_STATUS_VALUES:
            return {
                "type": "api_failure",
                "reason": f"status={obj.get('status')}, info={obj.get('info', '')}",
                "parsed": obj, "is_empty": True,
            }
        if obj.get("count") in (0, "0", 0.0):
            return {
                "type": "empty_results",
                "reason": f"count=0, info={obj.get('info', '')}",
                "parsed": obj, "is_empty": True,
            }
        for key in _LIST_FIELDS:
            v = obj.get(key)
            if isinstance(v, list) and len(v) == 0:
                return {
                    "type": "empty_results",
                    "reason": f"{key}=[]",
                    "parsed": obj, "is_empty": True,
                }
        if obj and all(v in _EMPTY_VALUES for v in obj.values()):
            return {
                "type": "empty_results",
                "reason": "all values empty",
                "parsed": obj, "is_empty": True,
            }
        return {
            "type": "has_data", "reason": "JSON dict with data",
            "parsed": obj, "is_empty": False,
        }

    if isinstance(obj, list):
        if not obj:
            return {
                "type": "empty_list", "reason": "[] 顶层数组",
                "parsed": obj, "is_empty": True,
            }
        return {
            "type": "has_data", "reason": f"non-empty list, len={len(obj)}",
            "parsed": obj, "is_empty": False,
        }

    return {
        "type": "unknown", "reason": "unexpected JSON shape",
        "parsed": obj, "is_empty": False,
    }


def extract_tool_responses(message: dict[str, Any]) -> list[dict[str, Any]]:
    """从一条 ``role="assistant"`` 消息里抽出所有 tool_calls 的响应分类。

    返回每个 tool_call 的 [{name, args, response, classification}, ...]。
    没有 tool_calls / 列表为空时返回空 list，调用方据此可判定"该轮没调工具"。
    """
    out: list[dict[str, Any]] = []
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        # 兼容 bundled 格式（name/args 顶层）+ OpenAI 格式（function.{name,arguments}）
        name = tc.get("name") or (tc.get("function") or {}).get("name", "")
        args = tc.get("args")
        if args is None and isinstance(tc.get("function"), dict):
            args = tc["function"].get("arguments")
        response = tc.get("response")
        classification = classify_tool_response(response)
        out.append({
            "name": name,
            "args": args,
            "response": response,
            "classification": classification,
        })
    return out


def turn_has_grounding_for(
    message: dict[str, Any]
) -> bool:
    """该轮 assistant 是否有【至少一个】tool_call 返回了真实数据。

    用于 IFS 判定时的快速过滤：True 表示该轮有 tool 数据可佐证 content；
    False 表示该轮所有 tool 调用都为空 / 失败 → content 的事实需要外部
    验证。
    """
    extracted = extract_tool_responses(message)
    if not extracted:
        return False
    return any(not item["classification"]["is_empty"] for item in extracted)


def annotate_tool_responses(
    history: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """给 conversation_history 中每条工具响应注入 ``_classification`` 字段。

    LLM 看到这个字段就知道该 tool 调用的响应"是不是空 / 哪种空"，**不需要
    自己 parse JSON 字符串**——彻底消除 LLM 在判 empty_results / api_failure /
    error_string 等场景上的主观抖动。

    支持两种格式：
      * normalize 后的 OpenAI flat 格式：在 ``role="tool"`` 消息上加注解
      * bundled 格式：在 ``assistant.tool_calls[*]`` 上原地加注解

    返回新的 list（浅拷贝每条 message + 改动注解的位置；原 history 不变动）。
    """
    annotated: list[dict[str, Any]] = []
    for msg in history or []:
        if not isinstance(msg, dict):
            annotated.append(msg)
            continue

        role = msg.get("role")

        # ── OpenAI flat 格式：tool 消息 ──────────────────────────────
        if role == "tool":
            new_msg = dict(msg)
            cls = classify_tool_response(msg.get("content"))
            new_msg["_classification"] = {
                "type": cls["type"],
                "is_empty": cls["is_empty"],
                "reason": cls["reason"],
            }
            annotated.append(new_msg)
            continue

        # ── bundled 格式：assistant.tool_calls 顶层有 response ──────
        if role == "assistant" and isinstance(msg.get("tool_calls"), list):
            new_msg = dict(msg)
            new_tcs = []
            for tc in msg["tool_calls"]:
                if not isinstance(tc, dict):
                    new_tcs.append(tc)
                    continue
                new_tc = dict(tc)
                # bundled 格式 response 在 tc 顶层；OpenAI 格式没有 response
                if "response" in new_tc:
                    cls = classify_tool_response(new_tc.get("response"))
                    new_tc["_classification"] = {
                        "type": cls["type"],
                        "is_empty": cls["is_empty"],
                        "reason": cls["reason"],
                    }
                new_tcs.append(new_tc)
            new_msg["tool_calls"] = new_tcs
            annotated.append(new_msg)
            continue

        annotated.append(msg)

    return annotated
