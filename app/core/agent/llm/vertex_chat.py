"""Vertex AI Gemini Chat Model (LangChain BaseChatModel 子类)。

通过 vertex 原生协议路径 ``/api/vertex/v1beta/models/{model}:
generateContent`` 调用 gemini 模型，**仅供隐式 React (BaseSimulationAgent)
对 gemini 系模型使用**——显式 React 通过 ``_USE_VERTEX_FOR_GEMINI=False``
仍走 ChatOpenAI 兼容协议。

为什么单独实现而不复用 ChatOpenAI：
  1. OpenAI 兼容协议在多轮 function calling 时会丢 ``thoughtSignature``
     字段（vertex 思考模型给每次工具调用配的签名），下一轮请求被 vertex
     后端拒绝，报 ``Function call is missing thought_signature``
  2. vertex 原生协议是 gemini 的 first-class 协议，function calling、
     thinking、usage_metadata 字段都更完整

最小可用版本（Phase 1）：
  * 仅支持 generateContent（非流式）。streamGenerateContent 留 TODO。
  * 支持 bind_tools (function calling)
  * thoughtSignature 双向透传（解析响应抽出来挂 additional_kwargs；构造下
    一轮请求时按 tool_call_id 反查写回 vertex part）
  * usageMetadata 映射：promptTokenCount/candidatesTokenCount/totalTokenCount
    + thoughtsTokenCount 单独挂到 additional_kwargs['reasoning_tokens']（不
    并入 output_tokens，与 OpenAI 兼容路径口径区分）
  * part.thought 文本作为 reasoning_content 放入 additional_kwargs（兼容
    现有 _merge_reasoning_into_content_if_empty 兜底）
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Optional, Sequence

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, SecretStr

logger = logging.getLogger(__name__)


# vertex finishReason → 与 OpenAI finish_reason 对齐（便于上层统一日志）
class VertexHTTPError(httpx.HTTPStatusError):
    """vertex 服务端 4xx/5xx 专用异常，**携带原始 response body**。

    标准 ``httpx.HTTPStatusError`` 的 message 只有
    ``"Client error '400 ' for url ..."``，丢失 vertex 返回的具体错误原因
    （如 "the number of function response parts is equal to..."）。本子类：

      * 自定义 message 包含完整 body，``str(e)`` 自然带详情
      * 额外属性 ``body_text`` 保存 vertex body，下游 ``dialogue_simulator``
        在 except 块用 ``getattr(e, 'body_text', None)`` 取出写到
        ``DialogueTurn.empty_response_dump`` 落 csv，事后可单字段查看
        完整服务端返回，无需翻日志

    上层 ``BaseSimulationAgent._ainvoke_with_retry`` 的 retryable 判定走
    ``isinstance(e, openai.APIConnectionError ...)`` 不命中 httpx 类型，
    所以本异常**不进 retry 链**——4xx 是语义错误，重试也是同样结果，应该
    暴露给业务侧诊断。
    """

    def __init__(self, message: str, *, request, response, body_text: str) -> None:
        super().__init__(message, request=request, response=response)
        self.body_text = body_text


_VERTEX_FINISH_REASON_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "MALFORMED_FUNCTION_CALL": "tool_call_error",
    "OTHER": "stop",
}

# OpenAI JSON-Schema 里 vertex 不接受 / 不需要的元字段；递归清理 parameters
_SCHEMA_KEYS_TO_DROP: frozenset[str] = frozenset(
    {"$schema", "$ref", "$id", "additionalProperties", "title", "examples"}
)


def _clean_schema(schema: Any) -> Any:
    """递归清理 OpenAI JSON-Schema 里 vertex 不支持的元字段。

    vertex functionDeclarations.parameters 是 OpenAPI 3.0 Schema 子集；
    OpenAI 习惯生成的 ``additionalProperties=false`` / ``title`` /
    ``examples`` 等会让 vertex 返 400。
    """
    if isinstance(schema, dict):
        return {
            k: _clean_schema(v)
            for k, v in schema.items()
            if k not in _SCHEMA_KEYS_TO_DROP
        }
    if isinstance(schema, list):
        return [_clean_schema(x) for x in schema]
    return schema


def _tools_to_vertex(tools: Sequence[Any]) -> list[dict]:
    """已 bind 的 LangChain tools (OpenAI schema) → vertex
    ``[{functionDeclarations: [...]}]``。

    BaseChatModel.bind_tools 把 tools 转 ``{type:"function", function:{...}}``
    存进 kwargs.tools。这里只抽 ``function`` 段、清理 schema。
    """
    declarations: list[dict] = []
    for t in tools:
        if isinstance(t, dict) and "function" in t:
            fn = t["function"]
        elif isinstance(t, dict) and "name" in t:
            fn = t
        else:
            fn = convert_to_openai_tool(t)["function"]
        declarations.append(
            {
                "name": fn.get("name") or "",
                "description": fn.get("description") or "",
                "parameters": _clean_schema(fn.get("parameters") or {}),
            }
        )
    return [{"functionDeclarations": declarations}]


def _messages_to_vertex(
    messages: Sequence[BaseMessage],
) -> tuple[list[dict], Optional[dict]]:
    """LangChain messages → vertex ``contents`` + ``systemInstruction``。

    role 映射::

        SystemMessage → 顶层 systemInstruction.parts[].text（vertex 单独字段）
        HumanMessage → {role:"user", parts:[{text:content}]}
        AIMessage(text only) → {role:"model", parts:[{text:content}]}
        AIMessage(with tool_calls) → {role:"model", parts:[
            {functionCall:{name, args}, thoughtSignature?:<from additional_kwargs>}
        ]}
        ToolMessage → {role:"user", parts:[{functionResponse:{name, response}}]}

    ToolMessage 通过 tool_call_id 反查前面 AIMessage 的 tool_calls 找到
    name（vertex functionResponse.name 必填，且 vertex 没有 tool_call_id
    概念，只能靠位置和 name 匹配）。

    **vertex 协议约束**（构造时显式保证，否则 400 INVALID_ARGUMENT）：
      1. **同 model turn 内的多个 functionCall 必须对应同一个 user turn
         里数量相等的 functionResponse**。LangChain 把并行工具调用拆成多
         条 ToolMessage，此处会把连续的 ToolMessage 合并到同一个 user
         content 的 parts 数组里。
      2. **role 严格交替**（user/model/user/model...）。连续 AIMessage 或
         连续 HumanMessage 会被合并到同一个 entry 防止 vertex 拒绝。
      3. **数量补齐**：若 model 的 functionCall 数 > 紧跟 user 的
         functionResponse 数（某条 tool 调用没产 ToolMessage，例如 tool
         异常被吞），按 name 配对补齐错误占位 functionResponse，让数量
         相等。否则 vertex 拒。
    """
    contents: list[dict] = []
    system_parts: list[dict] = []

    # 第一遍扫：建立 tool_call_id → tool_name 映射，供 ToolMessage 反查
    tool_call_id_to_name: dict[str, str] = {}
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                tcid = tc.get("id") or ""
                if tcid:
                    tool_call_id_to_name[tcid] = tc.get("name") or ""

    def _append_part(role: str, part: dict) -> None:
        """把 part 加到末尾 same-role entry 的 parts 里，否则新建 entry。

        保证 contents 序列相邻 entries 的 role 严格交替——LangChain message
        是平坦序列，多条 ToolMessage / 多条 AIMessage 之间可能不交替，靠这里
        合并避免 vertex 拒绝（fix #3）；同时把连续的 functionResponse 合并
        到同一 user content，解决 fix #1（vertex 要求同 model turn 的多个
        functionCall 与下一个 user turn 的多个 functionResponse 在同一 turn
        内一一对应）。
        """
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append(part)
        else:
            contents.append({"role": role, "parts": [part]})

    for m in messages:
        if isinstance(m, SystemMessage):
            system_parts.append({"text": str(m.content or "")})
            continue

        if isinstance(m, HumanMessage):
            _append_part("user", {"text": str(m.content or "")})
            continue

        if isinstance(m, AIMessage):
            parts_to_add: list[dict] = []
            if m.content:
                parts_to_add.append({"text": str(m.content)})
            # tool_calls → functionCall parts；thoughtSignature 透传
            sigs = (m.additional_kwargs or {}).get("thought_signatures") or {}
            for tc in m.tool_calls or []:
                part: dict[str, Any] = {
                    "functionCall": {
                        "name": tc.get("name") or "",
                        "args": tc.get("args") or {},
                    }
                }
                sig = sigs.get(tc.get("id") or "")
                if sig:
                    part["thoughtSignature"] = sig
                parts_to_add.append(part)
            if not parts_to_add:
                # 空 AIMessage → 给个空 text，vertex 拒绝空 parts
                parts_to_add.append({"text": ""})
            for p in parts_to_add:
                _append_part("model", p)
            continue

        if isinstance(m, ToolMessage):
            tcid = m.tool_call_id or ""
            tname = tool_call_id_to_name.get(tcid, "")
            if not tname:
                logger.warning(
                    "[VertexChat] ToolMessage tool_call_id=%r 找不到对应的 "
                    "AIMessage.tool_calls.name；functionResponse.name 留空可能被 "
                    "vertex 拒绝，检查 message 顺序",
                    tcid,
                )
            # ToolMessage.content 是 str；vertex functionResponse.response 要 dict
            response_obj: Any
            raw = m.content
            if isinstance(raw, dict):
                response_obj = raw
            else:
                try:
                    parsed = json.loads(str(raw))
                    response_obj = (
                        parsed if isinstance(parsed, dict) else {"result": parsed}
                    )
                except (json.JSONDecodeError, TypeError):
                    response_obj = {"result": str(raw or "")}

            _append_part(
                "user",
                {
                    "functionResponse": {
                        "name": tname,
                        "response": response_obj,
                    }
                },
            )
            continue

        # 未识别类型兜底
        logger.warning(
            "[VertexChat] 未识别 message 类型 %s，按 user text 兜底",
            type(m).__name__,
        )
        _append_part(
            "user",
            {"text": str(getattr(m, "content", "") or "")},
        )

    # fix #2 数量补齐：扫描相邻 (model, user) 对，若 model 含 N 个
    # functionCall、user 只含 M < N 个 functionResponse，按 name 配对补齐
    # 错误占位让数量相等。否则 vertex 拒：
    #   "the number of function response parts is equal to the number of
    #    function call parts of the function call turn"
    _pad_function_responses(contents)

    system_instruction = {"parts": system_parts} if system_parts else None
    return contents, system_instruction


def _pad_function_responses(contents: list[dict]) -> None:
    """vertex 严格要求 functionCall 数 == 紧随 user turn 的 functionResponse 数。

    对每个 model entry：
      * 数其 functionCall 数 N，找下一个 user entry 数其 functionResponse 数 M
      * M == N → OK
      * M < N → 按 name 配对补齐 N - M 个占位 functionResponse（response 体
        填错误说明）
      * M > N → 打 WARN（不主动 drop，请求大概率被 vertex 拒）

    in-place 修改 contents。
    """
    from collections import Counter

    for i, entry in enumerate(contents):
        if entry.get("role") != "model":
            continue
        fc_parts = [p for p in entry["parts"] if "functionCall" in p]
        if not fc_parts:
            continue
        if i + 1 >= len(contents):
            # 流末尾的 model entry：后面没 user turn 是正常的（等下次 inference
            # 来产 functionResponse），不补
            continue
        nxt = contents[i + 1]
        if nxt.get("role") != "user":
            continue

        fr_count = sum(1 for p in nxt["parts"] if "functionResponse" in p)
        fc_count = len(fc_parts)
        if fr_count == fc_count:
            continue
        if fr_count > fc_count:
            logger.warning(
                "[VertexChat] functionResponse 数 (%d) > functionCall 数 (%d)，"
                "vertex 可能拒；不主动 drop 保留原状",
                fr_count, fc_count,
            )
            continue

        # M < N：按 name multiset 差集找出缺失的 name，逐个补占位
        fc_names = [p["functionCall"].get("name") or "" for p in fc_parts]
        existing_fr_names = [
            p["functionResponse"].get("name") or ""
            for p in nxt["parts"]
            if "functionResponse" in p
        ]
        missing = Counter(fc_names) - Counter(existing_fr_names)
        added = 0
        for name in list(missing.elements()):
            nxt["parts"].append(
                {
                    "functionResponse": {
                        "name": name,
                        "response": {
                            "error": "tool execution missing (auto-filled)"
                        },
                    }
                }
            )
            added += 1
        if added:
            logger.warning(
                "[VertexChat] model turn 有 %d 个 functionCall 但 user turn "
                "只有 %d 个 functionResponse，自动补齐 %d 个错误占位 (names=%s)",
                fc_count, fr_count, added, list(missing.elements()),
            )


def _parse_vertex_response(payload: dict) -> AIMessage:
    """vertex generateContent 响应 → AIMessage。

    解析约定::

        content                = candidates[0].content.parts[].text 累加
        tool_calls             = candidates[0].content.parts[].functionCall 收集
                                 （自生成 ``vertex-<uuid8>`` 作 tool_call_id）
        additional_kwargs:
            reasoning_content    = candidates[0].content.parts[].thought 累加
                                   （part.thought 是 string 时；与现有
                                   ``_merge_reasoning_into_content_if_empty``
                                   兼容）
            thought_signatures   = {tool_call_id: sig}  ← 关键，下一轮请求
                                   会按这个 map 写回 vertex part
            thought_signature_text = 纯 text 响应也可能带 signature；兜底单挂
            reasoning_tokens     = usageMetadata.thoughtsTokenCount（不并入
                                   output_tokens，单列存放）
            finish_reason        = 归一化后的字符串
            finish_reason_raw    = vertex 原值
        usage_metadata         = input/output/total tokens（与 LangChain 标准
                                 字段对齐，下游 token 累计逻辑不需要改）
        response_metadata      = {model_name, finish_reason}
        id                     = 自生成 ``vertex-<uuid8>``（vertex 响应无 native id）
    """
    candidates = payload.get("candidates") or []
    if not candidates:
        # 空响应：保持 AIMessage 形态让上游 _stash_empty_dump_if_needed 接管
        return AIMessage(
            content="",
            additional_kwargs={"finish_reason": "no_candidates"},
            id=f"vertex-{uuid.uuid4().hex[:8]}",
        )

    cand0 = candidates[0]
    parts = (cand0.get("content") or {}).get("parts") or []

    content_text: list[str] = []
    tool_calls: list[dict] = []
    reasoning_text: list[str] = []
    thought_signatures: dict[str, str] = {}
    last_signature: Optional[str] = None

    for part in parts:
        text = part.get("text")
        if text:
            content_text.append(str(text))

        # part.thought 是 reasoning 文本（str）；也见过为 bool/None 的，过滤
        thought = part.get("thought")
        if isinstance(thought, str) and thought.strip():
            reasoning_text.append(thought)

        fc = part.get("functionCall")
        if isinstance(fc, dict):
            tc_id = f"vertex-{uuid.uuid4().hex[:8]}"
            tool_calls.append(
                {
                    "name": fc.get("name") or "",
                    "args": fc.get("args") or {},
                    "id": tc_id,
                    "type": "tool_call",
                }
            )
            sig = part.get("thoughtSignature")
            if sig:
                thought_signatures[tc_id] = sig
                last_signature = sig
        else:
            # 纯 text part 也可能带签名（如你示例里的纯文本响应）；记到 last
            sig = part.get("thoughtSignature")
            if sig:
                last_signature = sig

    additional_kwargs: dict[str, Any] = {}
    if reasoning_text:
        additional_kwargs["reasoning_content"] = "\n".join(reasoning_text)
    if thought_signatures:
        additional_kwargs["thought_signatures"] = thought_signatures
    if last_signature and not thought_signatures:
        # 没 functionCall 但仍有 signature → 单独存，未来如果发现 vertex 在
        # 纯 text 后续请求里也要回传，可在 _messages_to_vertex 加逻辑
        additional_kwargs["thought_signature_text"] = last_signature

    fr_raw = cand0.get("finishReason") or "STOP"
    fr_norm = _VERTEX_FINISH_REASON_MAP.get(fr_raw, str(fr_raw).lower())
    additional_kwargs["finish_reason"] = fr_norm
    additional_kwargs["finish_reason_raw"] = fr_raw

    # vertex 用 camelCase usageMetadata，部分版本同时给 snake_case usage_metadata
    usage_md = payload.get("usageMetadata") or payload.get("usage_metadata") or {}
    usage = {
        "input_tokens": int(usage_md.get("promptTokenCount") or 0),
        "output_tokens": int(usage_md.get("candidatesTokenCount") or 0),
        "total_tokens": int(usage_md.get("totalTokenCount") or 0),
    }
    reasoning_tokens = usage_md.get("thoughtsTokenCount")
    if reasoning_tokens is not None:
        additional_kwargs["reasoning_tokens"] = int(reasoning_tokens or 0)

    return AIMessage(
        content="".join(content_text),
        additional_kwargs=additional_kwargs,
        tool_calls=tool_calls,
        usage_metadata=usage,
        response_metadata={
            "model_name": payload.get("modelVersion") or "",
            "finish_reason": fr_norm,
        },
        id=f"vertex-{uuid.uuid4().hex[:8]}",
    )


class VertexChat(BaseChatModel):
    """通过 Vertex AI 原生 Gemini 协议的 Chat Model。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True
    )

    model: str = Field(..., description="gemini 模型名，如 gemini-3.1-pro-preview")
    base_url: str = Field(
        default="",  # 请自行配置 Vertex AI 兼容服务地址
        description="Vertex AI 兼容服务根路径（含 v1beta 段）",
    )
    api_key: SecretStr = Field(
        ..., description="Bearer token（与 OpenAI 兼容路径同 token）"
    )
    timeout: float = Field(default=300.0, description="单次 HTTP 请求超时（秒）")
    temperature: Optional[float] = Field(default=None)
    # gemini 默认 maxOutputTokens 通常仅 8192；thinking 模型 reasoning 部分
    # 容易吃满 → content 被截断或全是 thought 没正文输出，触发"真空响应"。
    # 拉到接近上限让 thinking + 最终回复都有充足预算。
    # ⚠️ 上限来自 vertex 服务端硬性约束，且**不同模型上限不同**：
    #   - gemini-3.1-pro-preview: "supported range is from 1 (inclusive) to
    #     65537 (exclusive)" → 最大 65536
    #   - gemini-2.5-flash-lite-07-22: "supported range is from 1 (inclusive)
    #     to 65536 (exclusive)" → 最大 65535
    # 取下确界 65535 兼容所有 gemini 模型；少 1 token 对生成质量无感。
    # 超过会被拒 400 INVALID_ARGUMENT。
    max_output_tokens: Optional[int] = Field(default=65535)
    # 是否启用 gemini thinking 模式。默认 False —— vertex 后端原本默认开启
    # thinking（即使我们不传 thinkingConfig），通过 ``thinkingConfig.thinkingBudget=0``
    # **显式关闭**。实测 gemini-3.1-pro-preview 关闭后 ``thoughtsTokenCount``
    # 由 ~900 降到 0、totalTokenCount 由 929 降到 24，省 ~97% token。
    # 与 BaseSimulationAgent.thinking 语义对齐（False=不思考，True=思考）。
    thinking: bool = Field(default=False)

    @property
    def _llm_type(self) -> str:
        return "vertex-chat"

    def _build_payload(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]],
        kwargs: dict,
    ) -> tuple[str, dict]:
        contents, system_instruction = _messages_to_vertex(messages)
        gen_config: dict[str, Any] = {"responseModalities": ["TEXT"]}
        if self.temperature is not None:
            gen_config["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            gen_config["maxOutputTokens"] = self.max_output_tokens
        if not self.thinking:
            # 显式关闭 thinking。vertex gemini 不传 thinkingConfig 时默认开启
            # thinking、产生大量 thoughtsTokenCount（实测 gemini-3.1-pro-preview
            # 单次问答 ~900 thought tokens）。thinkingBudget=0 是关闭它的官方
            # 方式。仅当 ``thinking=True`` 时才不传 thinkingConfig 让 vertex 走
            # 默认（开 thinking）。
            gen_config["thinkingConfig"] = {"thinkingBudget": 0}
        if stop:
            gen_config["stopSequences"] = list(stop)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": gen_config,
            "_originalFormat": "vertex-ai",
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        # bind_tools 把 OpenAI schema 塞进 kwargs.tools；这里转 vertex 格式
        bound_tools = kwargs.get("tools")
        if bound_tools:
            payload["tools"] = _tools_to_vertex(bound_tools)

        url = f"{self.base_url.rstrip('/')}/models/{self.model}:generateContent"
        return url, payload

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
        }

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        url, payload = self._build_payload(messages, stop, kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=self._headers(), json=payload)
        if resp.status_code >= 400:
            body_text = resp.text or ""
            logger.warning(
                "[VertexChat] HTTP %d body=%r",
                resp.status_code,
                body_text[:500],
            )
            # 抛 VertexHTTPError 而非 raise_for_status：携带完整 body 让
            # dialogue_simulator 异常兜底能拿到 vertex 真实错误原因写到 csv。
            short_message = (
                f"vertex HTTP {resp.status_code} for "
                f"{resp.request.url}: {body_text[:200]}"
            )
            raise VertexHTTPError(
                short_message,
                request=resp.request,
                response=resp,
                body_text=body_text,
            )
        ai_msg = _parse_vertex_response(resp.json())
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 隐式 React 路径全 async；sync 走 asyncio.run 兜底（仅测试 / REPL 用）
        return asyncio.run(self._agenerate(messages, stop, None, **kwargs))

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        """转 OpenAI schema 后走 BaseChatModel.bind，下游 _agenerate 在
        kwargs.tools 里拿出来再转 vertex 格式。"""
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, **kwargs)
