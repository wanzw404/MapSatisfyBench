"""Conversation-data normalizer (Stage 0 pre-processing).

The JudgeAgent and MetaJudge prompts both consume conversation_history as a
JSON-dumped flat OpenAI-style list:

    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [{"id": "...",
        "type": "function", "function": {"name": "...", "arguments": "<json>"}}]}
    {"role": "tool", "tool_call_id": "...", "content": "<json of tool response>"}

This module also accepts a more compact "bundled" format where each
assistant turn carries its own tool calls AND their responses, together with
a ``turn_index`` field::

    {"role": "user", "content": "...", "turn_index": 1}
    {"role": "assistant", "content": "...", "turn_index": 1, "tool_calls": [
        {"id": "t1", "name": "poi_search", "args": {...}, "response": {...}},
        {"id": "t2", "name": "driving_route", "args": {...}, "response": {...}}
    ]}

The bundled tool-call dict accepts either ``args`` (current canonical name
emitted by the inference pipeline) or ``arguments`` (legacy alias) for the
argument payload — both are normalised to ``function.arguments`` in the
output.

`normalize_conversation_history` detects the bundled format and expands it
back to the flat OpenAI-style list so downstream prompts need no changes.
The old format is returned unchanged.
"""

from __future__ import annotations

import json
from typing import Any


def _is_bundled_format(history: list[dict[str, Any]]) -> bool:
    """Return True if any assistant message uses the bundled tool-call schema.

    Signals that indicate the bundled format:
      * any message carries ``turn_index``;
      * any assistant ``tool_calls`` entry has a top-level ``response``;
      * any assistant ``tool_calls`` entry has a top-level ``args`` field
        (current canonical bundled key);
      * any assistant ``tool_calls`` entry has a top-level ``name`` (instead
        of ``function.name``) — this is the flat schema used in the spec.
    """
    for msg in history:
        if not isinstance(msg, dict):
            continue
        if "turn_index" in msg:
            return True
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                if "response" in tc:
                    return True
                if "args" in tc:
                    return True
                if "name" in tc and "function" not in tc:
                    return True
    return False


def _serialize_args(args: Any) -> str:
    if isinstance(args, str):
        return args
    return json.dumps(args or {}, ensure_ascii=False)


def _serialize_response(resp: Any) -> str:
    if isinstance(resp, str):
        return resp
    return json.dumps(resp, ensure_ascii=False)


def normalize_conversation_history(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand bundled history into OpenAI-style flat messages.

    If ``history`` is already flat, it is returned unchanged. Otherwise each
    assistant turn is split into:
        1. an ``assistant`` message carrying ``content`` + ``tool_calls`` in
           OpenAI schema (``{"id", "type": "function", "function": {...}}``),
        2. one ``role="tool"`` message per tool call, with
           ``tool_call_id`` linking back to the assistant call.

    The ``turn_index`` field is preserved on every emitted message so the
    judge prompt can still use it for reasoning if desired.
    """
    if not history:
        return []
    if not _is_bundled_format(history):
        return history

    out: list[dict[str, Any]] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")

        if role == "user":
            user_msg: dict[str, Any] = {
                "role": "user",
                "content": msg.get("content", ""),
            }
            if "turn_index" in msg:
                user_msg["turn_index"] = msg["turn_index"]
            out.append(user_msg)
            continue

        if role == "assistant":
            oai_tcs: list[dict[str, Any]] = []
            tool_msgs: list[dict[str, Any]] = []
            for i, tc in enumerate(msg.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id") or f"auto_{len(out)}_{i}"
                name = tc.get("name") or (tc.get("function") or {}).get("name", "")
                # Accept the new canonical key ``args`` first, then the legacy
                # ``arguments`` alias, then OpenAI's nested ``function.arguments``.
                args = tc.get("args")
                if args is None:
                    args = tc.get("arguments")
                if args is None and isinstance(tc.get("function"), dict):
                    args = tc["function"].get("arguments")
                oai_tcs.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": _serialize_args(args),
                    },
                })
                if "response" in tc:
                    tool_msgs.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": _serialize_response(tc["response"]),
                    })

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": msg.get("content", ""),
            }
            if oai_tcs:
                assistant_msg["tool_calls"] = oai_tcs
            if "turn_index" in msg:
                assistant_msg["turn_index"] = msg["turn_index"]
            out.append(assistant_msg)
            out.extend(tool_msgs)
            continue

        if role == "tool":
            # Already expanded — keep verbatim.
            out.append(msg)
            continue

        # Unknown role — keep verbatim so we do not silently drop data.
        out.append(msg)

    return out
