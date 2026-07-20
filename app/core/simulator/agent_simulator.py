import asyncio
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Annotated, Optional, Sequence, TypedDict

import yaml
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.core.tools.base import set_tool_sandbox, set_tool_trace_id

logger = logging.getLogger(__name__)

# ── 加载 prompt YAML ──
_PROMPT_FILE = (
    Path(__file__).resolve().parent.parent
    / "agent"
    / "prompt"
    / "agent_simulator_prompt.yaml"
)


def _load_prompt_template() -> str | None:
    """读取 agent_simulator_prompt.yaml，返回 chinese 模板文本。"""
    if not _PROMPT_FILE.exists():
        logger.warning("Prompt file not found: %s", _PROMPT_FILE)
        return None
    try:
        with open(_PROMPT_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("chinese")
    except Exception as e:
        logger.error("Failed to load prompt YAML: %s", e)
        return None


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # graph_input 注入：每轮 dialogue_simulator 透传，渲染 system prompt 用。
    # persona / full_intent 故意不放——评测里这两项是用户内心信息，不应让被测 agent 偷看。
    query: str
    context: str
    time: str
    location: str


class BaseSimulationAgent:
    """LangGraph React Agent 仿真器：agent 节点 + tools 节点的标准两节点循环。"""

    # 类级开关：model 名以 ``gemini`` 开头时是否切到 Vertex 原生协议
    # （而非 OpenAI 兼容协议）。Vertex 原生协议能正确透传 thoughtSignature，
    # 解决多轮 function calling 报 "Function call is missing
    # thought_signature" 的问题。
    _USE_VERTEX_FOR_GEMINI: bool = True

    # 单条 tool 在日志中截断的字符上限，避免长结果刷屏
    _TOOL_RESULT_LOG_LIMIT = 500

    # 单条 ToolMessage content 的字符上限。超出后追加截断提示，防止单次工具
    # 调用就把上下文打满。32KB ≈ 8K tokens，给 search_poi_along_route 这类
    # 已做字段过滤后 median ~24KB 的工具留足空间，避免再次触发头部截断丢失尾部
    # POI 数据（之前"工具调用循环" bug 的根因）。
    _TOOL_RESULT_MAX_CHARS = 32000

    # 单次 LLM 请求的超时（传给 openai/httpx）。卡死的 socket 会在这个时间被
    # 切断并抛 APITimeoutError，让 _ainvoke_with_retry 进入重试。
    _LLM_REQUEST_TIMEOUT = 300.0

    # 所有 agent LLM 调用统一 temperature。1.0 兼容所有已验证模型：
    #   * GPT-5 系（gpt-5-mini-0807-global / gpt-5.3-chat-0303-global 等）
    #     上游硬性约束 "Only default (1) value is supported"
    #   * GPT-4 / claude / qwen / gemini / deepseek 都能接受 1.0
    # 设为常量便于跨模型统一对比，需要按模型差异调整时往
    # _MODEL_SPECIFIC_TEMPERATURE 加 substring 即可。
    _DEFAULT_TEMPERATURE = 1.0

    # 模型特异 temperature 覆盖（小写 substring 匹配）：当某模型对
    # _DEFAULT_TEMPERATURE 不接受时，在此指定专属值。当前默认 1.0 已兼容
    # 所有验证过的模型，本表暂为空，保留作为未来扩展点。
    _MODEL_SPECIFIC_TEMPERATURE: dict[str, float] = {}
    # 重试次数（不含初次调用）。三次总尝试在 AGENT_TURN_TIMEOUT=950s 内可控
    # （3×300 + 1+2 backoff ≈ 903s）。
    _LLM_MAX_RETRIES = 2
    # 指数退避基数：1s, 2s
    _LLM_BACKOFF_BASE = 1.0

    # 模型名 substring（小写匹配）→ 强制 streaming 值。
    # 这是**网关侧的硬性约束**而非用户偏好，所有调用方进 __init__ 都会被
    # _resolve_streaming 自动校正，不论 streaming 入参是什么。
    #
    #   "claude":         OpenAI-compat 适配层多 tool_use chunk
    #                     index 错乱，流式会出 search_around_poisearch_poi
    #                     这种拼接 bug，必须非流式。
    #   "qwen3-30b-a3b":  thinking 模型；DashScope 在非流式下报错
    #                     "enable_thinking only support stream call"，
    #                     必须流式。
    #   "qwen3-4b":       同 30b-a3b，DashScope 对 qwen3-4b 开 thinking
    #                     时也强制流式（_THINKING_POLICY 子串 "qwen3" 已
    #                     命中并注入 enable_thinking=True，但本表的子串
    #                     原先只有 30b-a3b → 4b 走非流式触发 400）。
    #
    # 未匹配的 model 走 _DEFAULT_STREAMING_FOR_UNLISTED（非流式）——避免
    # 在未经验证的模型上意外开流式踩 chunk reducer / token usage 缺失等坑。
    # 新增 strict 模型加一行 substring 即可。
    _STREAMING_POLICY: dict[str, bool] = {
        # claude 移除：旧记录 False 是为了兜底 OpenAI-compat 适配层
        # 流式 tool_use chunk index bug；现在 claude 走 Anthropic 原生
        # 协议（无 chunk bug），与其它模型一样默认非流式 + 用户 --streaming opt-in
        "qwen3-30b-a3b": True,
        "qwen3-4b": True,
        "qwen3-8b": True,
    }
    _DEFAULT_STREAMING_FOR_UNLISTED = False

    # 模型名 substring（小写匹配）→ thinking on/off 双键策略。
    # 每个 entry 的 on / off 字段语义：
    #   * dict — 该状态需要注入的 ChatOpenAI kwargs（一般是 extra_body）
    #   * None — 该状态无需注入（默认行为已对齐，或服务端无法切换该状态）
    # 两个键都必须显式写出（None 也要写），避免漏配置。
    #
    # 决策来源（详见 _resolve_thinking_kwargs）：
    #   * thinking=True  + on  is None → raise（该模型不支持开 thinking，如非
    #                                          reasoning 模型 gpt-4.1 / gpt-5.x-chat）
    #   * thinking=True  + on  is dict → 返 on
    #   * thinking=False + off is None → 返 {}（默认即 off）
    #   * thinking=False + off is dict → 返 off（**强制关**：服务端默认 on 的模型
    #                                            必须显式发关闭载荷，如 DashScope 系
    #                                            的 qwen3 / deepseek-v4-pro）
    #
    # **顺序敏感**：dict 按插入顺序遍历，**长 substring 必须排在短 substring 之前**
    # （如 "qwen3.6-plus" 在 "qwen3" 之前）；首个命中即返。
    #
    # gemini-* 不进本表：__init__ 的 use_vertex 分支已分流到 VertexChat，
    # 由其内部 thinkingConfig.thinkingBudget=0 控制开/关，与 extra_body 体系无关。
    _THINKING_POLICY: dict[str, dict[str, Optional[dict]]] = {
        # ── OpenAI 非 reasoning 模型：不存在 thinking 概念 ────────────────
        "gpt-5.3-chat": {"on": None, "off": None},
        "gpt-41":       {"on": None, "off": None},

        # ⚠️ claude 系不在本表里：claude 走 Anthropic 原生协议
        # （详见 app/core/agent/llm/anthropic_chat.py），thinking 由
        # ChatAnthropic 的顶层 ``thinking`` 参数驱动，不经 extra_body。
        # __init__ 在 use_anthropic 分支提前接管，根本不会调用本函数。

        # ── Qwen 系列（DashScope/百炼）：服务端默认 ON，强制关 ────
        # 长 substring 在前。qwen3.6-plus 与 qwen3 载荷相同但分行写便于审计。
        "qwen3.6-plus": {
            "on":  {"extra_body": {"enable_thinking": True}},
            "off": {"extra_body": {"enable_thinking": False}},
        },
        "qwen3": {  # 命中 qwen3-4b / Qwen3-30B-A3B / 其它 qwen3.* 变体
            "on":  {"extra_body": {"enable_thinking": True}},
            "off": {"extra_body": {"enable_thinking": False}},
        },
        "qwen-plus": {  # 旧版 qwen-plus 变体（非 3.x）
            "on":  {"extra_body": {"enable_thinking": True}},
            "off": {"extra_body": {"enable_thinking": False}},
        },

        # ── DeepSeek 系列（百炼/DashScope）：默认 ON，强制关 ──────
        "deepseek-v4-pro": {
            "on":  {"extra_body": {"enable_thinking": True}},
            "off": {"extra_body": {"enable_thinking": False}},
        },
        "deepseek-v3.2": {  # V3.2-Exp 支持 hybrid；保守显式关
            "on":  {"extra_body": {"enable_thinking": True}},
            "off": {"extra_body": {"enable_thinking": False}},
        },
    }

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "",
        tools: list | None = None,
        sandbox: bool = False,
        streaming: bool = False,
        thinking: bool = False,
    ) -> None:
        """
        Args:
            streaming: 启用后 ``call_llm`` 内部走 ``astream`` 累加 chunks，
                可拿到真实首 chunk 时间作为 TTFT；LangGraph 上层与非流式行为完全一致。
                **注意**：实际生效值由 ``_STREAMING_POLICY`` 根据 model 名最终决定，
                入参可能被 ``_resolve_streaming`` 覆盖（见类常量注释）。
            thinking: 是否启用 thinking / reasoning 模式。默认 False。
                启用时根据 ``_THINKING_POLICY`` 给 model 注入对应 extra_body：
                qwen 系 → ``enable_thinking=True``；claude → ``thinking={...}``；
                gpt-o1/o3 → ``reasoning_effort=high``。
                **严格白名单**：传 True 但 model 未在策略表中 → ``__init__``
                直接 raise ``ValueError``，让用户立刻知道该模型不支持。
        """
        # 网关行为决定的 streaming 策略，比入参优先。
        streaming = self._resolve_streaming(model, streaming)

        # 模型路径分发：claude → Anthropic 原生协议；
        #              gemini（隐式 React）→ Vertex 原生协议；
        #              其它 → ChatOpenAI 兼容协议
        use_anthropic = (model or "").lower().startswith("claude")
        use_vertex = (
            self._USE_VERTEX_FOR_GEMINI
            and (model or "").lower().startswith("gemini")
        )

        # _resolve_thinking_kwargs 只对 ChatOpenAI 路径有意义——anthropic /
        # vertex 路径用各自原生 client 的 thinking 顶层参数控制，跳过严格
        # 白名单校验避免误 raise。
        if use_anthropic or use_vertex:
            thinking_kwargs: dict = {}
        else:
            thinking_kwargs = self._resolve_thinking_kwargs(model, thinking)
        if use_vertex and streaming:
            logger.warning(
                "[Agent] model=%r 走 Vertex 协议，Phase 1 尚未实现 "
                "streamGenerateContent；streaming=True 已自动降级为非流式",
                model,
            )
            streaming = False

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.tools = tools or []
        self.sandbox = sandbox
        self.streaming = streaming
        self.thinking = thinking
        self.llm_metrics_log: list[dict] = []
        self._prompt_template = _load_prompt_template()

        if use_anthropic:
            from app.core.agent.llm.anthropic_chat import (
                make_anthropic_client,
            )
            self._llm_client = make_anthropic_client(
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout=self._LLM_REQUEST_TIMEOUT,
                temperature=self._resolve_temperature(model),
                thinking=thinking,
                streaming=streaming,
            )
        elif use_vertex:
            # 局部 import 避免非 gemini 路径强依赖 httpx
            from app.core.agent.llm.vertex_chat import VertexChat

            # base_url 智能替换：调用方传的通常是 OpenAI 路径根
            # （.../api/openai/v1），转成 vertex 根（.../api/vertex/v1beta）
            vertex_base = base_url.rstrip("/")
            if vertex_base.endswith("/api/openai/v1"):
                vertex_base = vertex_base[: -len("/api/openai/v1")] + "/api/vertex/v1beta"
            elif "/api/openai" in vertex_base:
                vertex_base = vertex_base.replace("/api/openai", "/api/vertex/v1beta", 1)
            logger.info(
                "[Agent] gemini 模型走 Vertex 协议 (model=%r, base_url=%r)",
                model, vertex_base,
            )
            self._llm_client = VertexChat(
                model=model,
                base_url=vertex_base,
                api_key=api_key,
                timeout=self._LLM_REQUEST_TIMEOUT,
                temperature=self._resolve_temperature(model),
                # 透传 thinking 开关：vertex 路径不读 _THINKING_POLICY，靠
                # VertexChat 内的 thinkingConfig.thinkingBudget=0 关闭
                thinking=thinking,
            )
        else:
            self._llm_client = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                disable_streaming=not streaming,
                timeout=self._LLM_REQUEST_TIMEOUT,
                temperature=self._resolve_temperature(model),
                **thinking_kwargs,
            )
        if self.tools:
            self._llm_client = self._llm_client.bind_tools(self.tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        # 运行时拼接的工具简表，注入到 system prompt 的 {tools_brief}，
        # 与 bind_tools 暴露给 LLM 的工具集**永远一致**，避免 prompt 列表与
        # bind 列表漂移导致 LLM 调用未注册工具。
        self._tools_brief = self._build_tools_brief()

        # 跨轮 checkpointer + 编译好的图：必须复用同一个 MemorySaver，否则
        # 同一 thread_id 在每次新建的 saver 中查不到历史，多轮记忆会失效。
        self._memory = MemorySaver()
        self._graph = self._build_graph()

    @classmethod
    def _resolve_temperature(cls, model: str) -> float:
        """根据 model 名应用 ``_MODEL_SPECIFIC_TEMPERATURE``，返回最终注入值。

        命中表 → 用表中的覆盖值；未命中 → ``_DEFAULT_TEMPERATURE``（0.7）。
        """
        name_l = (model or "").lower()
        for substr, override in cls._MODEL_SPECIFIC_TEMPERATURE.items():
            if substr.lower() in name_l:
                if override != cls._DEFAULT_TEMPERATURE:
                    logger.info(
                        "[Agent] model=%r temperature 覆盖为 %s（"
                        "_MODEL_SPECIFIC_TEMPERATURE 命中 %r，原默认 %s）",
                        model, override, substr, cls._DEFAULT_TEMPERATURE,
                    )
                return override
        return cls._DEFAULT_TEMPERATURE

    @classmethod
    def _resolve_streaming(cls, model: str, requested: bool) -> bool:
        """根据 model 名应用 ``_STREAMING_POLICY``，返回实际生效的 streaming。

        命中策略表 → 用表里的 forced 值；未命中 → 返回
        ``_DEFAULT_STREAMING_FOR_UNLISTED`` (非流式)。任何与请求值不一致的
        覆盖都会打 WARNING，便于排查"为什么我开了 --streaming 没生效"。
        """
        name_l = (model or "").lower()
        for substr, forced in cls._STREAMING_POLICY.items():
            if substr in name_l:
                if requested != forced:
                    logger.warning(
                        "[Agent] model=%r 强制 streaming=%s（请求 streaming=%s 已覆盖）；"
                        "原因：网关对该模型的协议适配硬性约束（_STREAMING_POLICY 命中 %r）",
                        model, forced, requested, substr,
                    )
                return forced
        # 未命中 → 默认非流式（覆盖任何 streaming=True 请求）
        if requested != cls._DEFAULT_STREAMING_FOR_UNLISTED:
            logger.warning(
                "[Agent] model=%r 未在 _STREAMING_POLICY 中显式列出，"
                "强制使用默认 streaming=%s（请求 streaming=%s 已覆盖）；"
                "如确认该模型适合开流式，请加到 _STREAMING_POLICY",
                model, cls._DEFAULT_STREAMING_FOR_UNLISTED, requested,
            )
        return cls._DEFAULT_STREAMING_FOR_UNLISTED

    @classmethod
    def _resolve_thinking_kwargs(cls, model: str, thinking: bool) -> dict:
        """根据 model 名应用 ``_THINKING_POLICY``，返回要注入 ChatOpenAI 的 kwargs。

        决策树（按优先级）：
          1. model 命中 ``_THINKING_POLICY`` substring：
             a. thinking=True 且 entry.on is None → raise（该模型不支持开 thinking，
                如 gpt-4.1 / gpt-5.x-chat 这类非 reasoning 模型）
             b. thinking=True 且 entry.on is dict → 返 entry.on（注入 extra_body 开）
             c. thinking=False 且 entry.off is None → 返 ``{}``（默认即 off，
                如 claude——零注入即关）
             d. thinking=False 且 entry.off is dict → 返 entry.off（**强制关**：
                服务端默认 on 的模型必须显式发关闭载荷，如 DashScope 系
                qwen3 / deepseek-v4-pro / deepseek-v3.2）
          2. model 未命中任何 substring：
             - thinking=True  → raise（白名单严格）
             - thinking=False → 返 ``{}``（静默通过，未知模型不擅自加 extra_body）

        gemini-* 不会进本函数：``__init__`` 的 use_vertex 分支已分流到
        ``VertexChat``，由其内部 ``thinkingConfig.thinkingBudget=0`` 控制。
        """
        name_l = (model or "").lower()
        matched: Optional[tuple[str, dict[str, Optional[dict]]]] = None
        for substr, entry in cls._THINKING_POLICY.items():
            if substr in name_l:
                matched = (substr, entry)
                break

        if matched is None:
            if thinking:
                supported = list(cls._THINKING_POLICY.keys())
                raise ValueError(
                    f"模型 {model!r} 不在 _THINKING_POLICY 中，无法启用 thinking。"
                    f"已支持 substring: {supported}。"
                    f"如该模型本就不支持 thinking，去掉 thinking=True；"
                    f"如确认支持，请在 _THINKING_POLICY 加 entry。"
                )
            return {}

        substr, entry = matched
        key = "on" if thinking else "off"
        kwargs = entry.get(key)

        if thinking and kwargs is None:
            raise ValueError(
                f"模型 {model!r}（命中 substring {substr!r}）不支持 thinking 模式："
                f"_THINKING_POLICY[{substr!r}]['on'] 为 None。"
                f"非 reasoning 模型（如 gpt-4.1 / gpt-5.x-chat）请去掉 thinking=True。"
            )

        if not thinking and kwargs is None:
            # 默认 off 即可，零注入；DEBUG 级日志便于审计
            logger.debug(
                "[Agent] model=%r 默认非 thinking 模式（命中 %r，entry.off=None）"
                "，跳过 extra_body 注入",
                model, substr,
            )
            return {}

        # 命中并且有载荷
        logger.info(
            "[Agent] model=%r thinking=%s（命中 %r）注入 kwargs=%s",
            model, thinking, substr, kwargs,
        )
        return dict(kwargs)

    def _build_tools_brief(self) -> str:
        """按 self.tools 渲染工具列表（一行一个），作为 system prompt 的工具清单。"""
        if not self.tools:
            return "（当前未绑定任何工具）"
        lines: list[str] = []
        for t in self.tools:
            desc = (getattr(t, "description", "") or "").strip()
            first_line = desc.splitlines()[0] if desc else ""
            lines.append(f"**{t.name}** {first_line}")
        return "\n".join(lines)

    def _render_prompt(self, context: dict) -> str:
        """将 YAML 模板中的 {参数名} 替换为 context 中的同名值。"""
        if not self._prompt_template:
            return "You are a helpful AI assistant."
        template = self._prompt_template

        def _replacer(match: re.Match) -> str:
            key = match.group(1)
            value = context.get(key, "")
            return str(value) if value is not None else ""

        return re.sub(r"\{(\w+)\}", _replacer, template)

    async def _execute_tool_call(self, tool_call: dict) -> ToolMessage:
        """单条 tool_call 的执行 + 错误归一化。

        - 工具名未注册：返回带 error 的 ToolMessage，让 LLM 在下一步纠错（而非
          抛 KeyError 终止整轮）
        - 工具内部抛异常：同上，记录异常类型 + message
        - 强制 sandbox：覆盖 args["sandbox"] = self.sandbox，防止 LLM 在 args 里
          自己塞 sandbox=true 把真实调用偷换成 mock
        """
        tool_name = tool_call.get("name") or ""
        tool_call_id = tool_call.get("id") or str(uuid.uuid4())
        args = dict(tool_call.get("args") or {})
        args["sandbox"] = self.sandbox  # 强制覆盖，无视 LLM 输入

        tool = self.tools_by_name.get(tool_name)
        if tool is None:
            available = ", ".join(self.tools_by_name) or "(none)"
            err_msg = f"未注册的工具 '{tool_name}'。可用工具: {available}"
            logger.warning("[Tool] unknown tool_name=%s", tool_name)
            return ToolMessage(
                content=json.dumps({"error": err_msg}, ensure_ascii=False),
                name=tool_name or "unknown",
                tool_call_id=tool_call_id,
            )

        try:
            result = await tool.ainvoke(args)
        except Exception as e:
            # 注意：这里**不是**模拟器 bug，而是 LLM 给了错参数（典型如
            # pydantic ValidationError 在 LangChain `_parse_input` 阶段抛——
            # 早于工具体，工具内 try/except 救不了，必须在这里兜底）；
            # 异常被吞 + 错误进 ToolMessage 喂回 agent 让其纠错。原始
            # name / args / 错误信息会经 _extract_agent_output 一并落到
            # turn.tool_calls.response。
            # 用 WARNING 而非 ERROR + traceback，避免日志看起来像致命崩溃。
            logger.warning(
                "[Tool] %s 参数/执行失败（已作为 error 喂回 agent）: %s: %s",
                tool_name, type(e).__name__, str(e)[:300],
            )
            # 把请求 args（去 sandbox 内部字段）+ 异常类型一并写入 response，
            # 让结果表 turn.tool_calls.response 一眼看清"送了什么参数 / 怎么炸的"，
            # LLM 看到也能直接定位是哪个字段类型错了。
            req_args = {k: v for k, v in args.items() if k != "sandbox"}
            return ToolMessage(
                content=json.dumps(
                    {
                        "error": f"{type(e).__name__}: {e}",
                        "exception_type": type(e).__name__,
                        "tool_name": tool_name,
                        "request_args": req_args,
                    },
                    ensure_ascii=False,
                    default=str,
                ),
                name=tool_name,
                tool_call_id=tool_call_id,
            )

        try:
            content_str = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            content_str = str(result)

        # 单条 tool 结果太长直接截断，否则 ToolMessage 会被一直带在历史里把上下文打爆。
        # 截断后追加明确提示，让 LLM 知道结果不完整、可以缩小搜索范围或换关键词。
        original_len = len(content_str)
        if original_len > self._TOOL_RESULT_MAX_CHARS:
            keep = self._TOOL_RESULT_MAX_CHARS - 200
            content_str = (
                content_str[:keep]
                + f"\n\n[TOOL_RESULT_TRUNCATED: 原始 {original_len} 字符已截至 {keep}；"
                  f"后续内容省略，如需更完整请缩小检索范围或换更精确的关键词]"
            )
            # logger.info(
            #     "[Tool] %s 结果过长，截断 %d→%d chars",
            #     tool_name, original_len, len(content_str),
            # )

        # logger.info(
        #     "[Tool] %s | args=%s | result=%s",
        #     tool_name,
        #     json.dumps(args, ensure_ascii=False, default=str),
        #     content_str[: self._TOOL_RESULT_LOG_LIMIT],
        # )
        return ToolMessage(
            content=content_str,
            name=tool_name,
            tool_call_id=tool_call_id,
        )

    async def tool_node(self, state: AgentState, config: RunnableConfig | None = None):
        """LangGraph tool 节点：并发执行最后一条 AIMessage 上的所有 tool_calls。"""
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        if not tool_calls:
            return {"messages": []}

        # 注入 trace_id / sandbox 一次。asyncio.gather 创建子 Task 时会复制
        # 当前 ContextVar 上下文，所以并发子任务都能拿到。
        trace_id = None
        if config:
            trace_id = config.get("configurable", {}).get("thread_id")
        set_tool_trace_id(trace_id)
        set_tool_sandbox(self.sandbox)

        # 并发执行：LLM 一轮多个 tool_calls 时不再串行 await
        tool_messages = await asyncio.gather(
            *(self._execute_tool_call(tc) for tc in tool_calls)
        )
        return {"messages": list(tool_messages)}

    @staticmethod
    def _stash_empty_dump_if_needed(message):
        """LLM 网关返回后**最早期**的快照：当 message 看起来是"空响应"
        （content strip 后为空 + 无 tool_calls）时，把 ``repr(message)``
        写到 ``additional_kwargs['empty_response_dump']``。

        必须在任何后处理（如 ``_merge_reasoning_into_content_if_empty``
        把 reasoning_content 拼进 content）之前调用，否则原始"content 为
        空"的事实会被改写丢失，无法用于故障诊断。

        触发条件刻意收窄到「看起来要走下游空响应兜底」的场景：
          1. ``content`` 为空 / 全 whitespace
          2. 没有 ``tool_calls``
        其余场景 no-op，避免 state.messages 里所有 AIMessage 都背着几 KB
        dump 字符串、加重 LangGraph checkpointer 序列化与显存占用。

        幂等：用 ``setdefault`` 避免流式 chunk 累加 / retry 重复挂载。
        """
        content = getattr(message, "content", "")
        # ChatAnthropic 在 thinking on / tool_use 时返 content=list[blocks]，
        # 直接 isinstance(str) 判会 false-negative（把含 text 块的非空响应误判
        # 成"空"），整条 AIMessage 连同 thinking signature 被 repr 进 dump。
        # 走 normalize 同时兼容 str / list 形态，与 ``_is_empty_response`` 一致。
        text = BaseSimulationAgent._normalize_anthropic_content_to_str(content)
        if text.strip():
            return message
        if getattr(message, "tool_calls", None):
            return message
        if not isinstance(getattr(message, "additional_kwargs", None), dict):
            return message
        # 先 repr 拿快照字符串，再写 setdefault；顺序颠倒会让 dump 包含
        # 它自己（自引用）形成无限套娃。
        dump = repr(message)
        message.additional_kwargs.setdefault("empty_response_dump", dump)
        return message

    @staticmethod
    def _merge_reasoning_into_content_if_empty(message):
        """Thinking 模型（典型 qwen3 系 enable_thinking=True）把推理放在
        ``additional_kwargs['reasoning_content']`` 单独字段，LangChain
        ``ChatOpenAI`` 不会自动并入 ``content``。当模型把"调用 X 工具"
        这种决策只写在 reasoning_content 没回流到 content 时，content
        会是空字符串，导致后续处理逻辑拿不到有效内容。

        本方法在三个条件**同时**满足时把 reasoning_content 拼进 content：
          1. ``content`` 为空或只含空白（已有真实回复时不动）
          2. 没有 ``tool_calls``（隐式 ReAct 走 native tool_calls 路径，
             此时 content 为空属于正常协议，不应合并，否则 csv content
             列会被思考文字污染）
          3. ``additional_kwargs['reasoning_content']`` 非空字符串

        任一条件不满足 → 函数 no-op、原样返回 message，保证：
          - 非 thinking 模型不受影响
          - 隐式 ReAct + bind_tools 协议不受影响
          - 已经拿到正常 content 的 thinking 模型不被重复拼接
        """
        content = getattr(message, "content", "") or ""
        # ChatAnthropic 把 thinking + text + tool_use 都放在 content 数组里——
        # 这条 list 形态早就不"空"，且 thinking 内容也无需"补"进 content（它
        # 本身就在 list 里、下一轮回传时 ChatAnthropic 序列化会处理）。
        # 进一步：list 形态也意味着 ChatAnthropic 路径，`additional_kwargs` 一般
        # 不存 reasoning_content，强行合并反而把 list 改成 str 破坏结构。
        # 直接 no-op。
        if isinstance(content, list):
            return message
        if isinstance(content, str) and content.strip():
            return message
        if getattr(message, "tool_calls", None):
            return message
        extra = getattr(message, "additional_kwargs", None) or {}
        reasoning = extra.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning.strip():
            return message

        logger.info(
            "[Agent LLM] content 为空且无 tool_calls，合并 reasoning_content "
            "(len=%d) 进 content 让下游解析（thinking 模型典型路径）",
            len(reasoning),
        )
        message.content = reasoning
        return message

    async def _ainvoke_with_retry(self, messages: list, config: RunnableConfig):
        """LLM 调用的 retry wrapper：仅对**可恢复的**错误重试。

        触发重试的异常：
          - asyncio.TimeoutError       : ChatOpenAI 内部 timeout 触发（per-request）
          - openai.APITimeoutError     : openai SDK 超时
          - openai.APIConnectionError  : 网络抖动
          - openai.InternalServerError : 5xx
          - openai.RateLimitError      : 429（队列已满）

        不重试：BadRequest（如 context_length_exceeded）、Auth、PermissionDenied 等
        语义性错误——再重试也是同样结果。

        退避策略：1s → 2s（指数）。3 次总尝试在 AGENT_TURN_TIMEOUT=180s 内控制得住。
        """
        # 局部 import 避免在没装 openai SDK 的环境下 module 加载失败
        from openai import (
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
            RateLimitError,
        )
        # claude 路径走 Anthropic SDK，异常类型独立——同名但不同 module；
        # 不加这一组 claude 的 5xx / 网络抖动会被当成不可重试错误直接抛出。
        import anthropic

        retryable = (
            asyncio.TimeoutError,
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
            RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
            anthropic.RateLimitError,
        )

        last_exc: Exception | None = None
        for attempt in range(self._LLM_MAX_RETRIES + 1):
            try:
                response = await self._llm_client.ainvoke(messages, config)
                # stash 必须在 merge 之前：merge 会把 reasoning_content 拼到
                # content，会改变"content 为空"这个原始事实
                response = self._stash_empty_dump_if_needed(response)
                return self._merge_reasoning_into_content_if_empty(response)
            except retryable as e:
                last_exc = e
                if attempt >= self._LLM_MAX_RETRIES:
                    logger.error(
                        "[Agent LLM] ainvoke 最终失败（%d/%d）: %s: %s",
                        attempt + 1,
                        self._LLM_MAX_RETRIES + 1,
                        type(e).__name__,
                        str(e)[:200],
                    )
                    break
                backoff = self._LLM_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "[Agent LLM] ainvoke 第 %d/%d 次失败（%s: %s），%.1fs 后重试",
                    attempt + 1,
                    self._LLM_MAX_RETRIES + 1,
                    type(e).__name__,
                    str(e)[:120],
                    backoff,
                )
                await asyncio.sleep(backoff)

        # 重试用尽，把最后一次异常抛出去；外层 dialogue_simulator 的 except 会
        # 把这一轮 agent_turn 标成 status="error"
        assert last_exc is not None
        raise last_exc

    async def _astream_with_retry(
        self, messages: list, config: RunnableConfig
    ) -> tuple:
        """流式 LLM 调用 + 同样的 retry 语义。返回 ``(assembled_message, ttft_ms)``。

        实现要点：
          - 用 ``astream`` 接收 ``AIMessageChunk`` 序列；首 chunk 到达时间作为
            真实 TTFT
          - 多 chunk 通过 ``+`` 累加（LangChain 内置 reducer 处理 content /
            tool_call_chunks → 最终 ``tool_calls`` 数组）
          - retry 异常类型与 ``_ainvoke_with_retry`` 一致；同类网络/超时/5xx
            错误都进入指数退避
        """
        from openai import (
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
            RateLimitError,
        )
        import anthropic  # claude 路径独立异常族；详见 _ainvoke_with_retry 注释

        retryable = (
            asyncio.TimeoutError,
            APITimeoutError,
            APIConnectionError,
            InternalServerError,
            RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
            anthropic.RateLimitError,
        )

        last_exc: Exception | None = None
        for attempt in range(self._LLM_MAX_RETRIES + 1):
            try:
                first_chunk_at: float | None = None
                started_at = time.time()
                assembled = None
                async for chunk in self._llm_client.astream(messages, config):
                    if first_chunk_at is None:
                        first_chunk_at = time.time()
                    assembled = chunk if assembled is None else assembled + chunk
                if assembled is None:
                    # 流式直接返回空（极少见）；视为成功但内容空
                    raise RuntimeError("astream 没有任何 chunk 返回")
                ttft_ms = int(((first_chunk_at or started_at) - started_at) * 1000)
                # stash 必须在 merge 之前；见 _ainvoke_with_retry 同理注释
                assembled = self._stash_empty_dump_if_needed(assembled)
                assembled = self._merge_reasoning_into_content_if_empty(assembled)
                return assembled, ttft_ms
            except retryable as e:
                last_exc = e
                if attempt >= self._LLM_MAX_RETRIES:
                    logger.error(
                        "[Agent LLM] astream 最终失败（%d/%d）: %s: %s",
                        attempt + 1,
                        self._LLM_MAX_RETRIES + 1,
                        type(e).__name__,
                        str(e)[:200],
                    )
                    break
                backoff = self._LLM_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "[Agent LLM] astream 第 %d/%d 次失败（%s: %s），%.1fs 后重试",
                    attempt + 1,
                    self._LLM_MAX_RETRIES + 1,
                    type(e).__name__,
                    str(e)[:120],
                    backoff,
                )
                await asyncio.sleep(backoff)

        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _compact_history(messages: list) -> list:
        """跨轮压缩历史：仅用于发给 LLM 的副本，不修改 state["messages"]。

        历史轮（最后一条 HumanMessage 之前）：
          - 保留：HumanMessage、每轮最后一条**有内容**的 AIMessage
          - 丢弃：所有 ToolMessage、空 content 的 tool_calls 中间步骤

        当前轮（最后一条 HumanMessage 起）：
          - 原样保留，确保 React loop 内 AIMessage(tool_calls) ↔ ToolMessage 配对完整

        作用：长会话时历史 ToolMessage 累积是 token 暴涨的主因；该方法把
        历史压缩成「user 说了什么 / agent 最终答了什么」，丢掉中间 tool 细节。
        """
        if not messages:
            return []

        human_idx = [
            i for i, m in enumerate(messages)
            if getattr(m, "type", "") == "human"
        ]
        # 0 或 1 条 HumanMessage 都视为「全部属于当前轮」，原样保留
        if len(human_idx) <= 1:
            return list(messages)

        compacted = []
        # 处理已结束的历史轮次（不含最后一条 HumanMessage 起的当前轮）
        for ti in range(len(human_idx) - 1):
            start, end = human_idx[ti], human_idx[ti + 1]
            turn_msgs = messages[start:end]
            compacted.append(turn_msgs[0])  # HumanMessage 必保
            # 倒序找最后一条**有文本内容**的 AIMessage 作为该轮的回复
            for m in reversed(turn_msgs[1:]):
                if getattr(m, "type", "") != "ai":
                    continue
                # ChatAnthropic 在 thinking on / tool_use 时返 content=list[blocks]，
                # 直接 .strip() 会 AttributeError（'list' has no .strip）。先把 list
                # normalize 成纯文本再判空。
                raw_content = getattr(m, "content", "")
                content = BaseSimulationAgent._normalize_anthropic_content_to_str(
                    raw_content
                ).strip()
                if not content:
                    continue
                # 把 tool_calls 清空：对应的 ToolMessage 已经被丢弃，留 tool_calls
                # 会让 OpenAI 校验失败「tool_calls 后必须接 tool 消息」。
                # 同理：list content 里的 tool_use 块也是孤儿（tool_result 已被
                # compaction 丢弃），会触发 Anthropic 端的 orphan tool_use 校验
                # 失败；并且历史轮的 thinking/signature 已无 server-side 价值。
                # 所以历史副本里把 content 一并替换成 normalize 后的纯字符串。
                updates: dict = {}
                if getattr(m, "tool_calls", None):
                    updates["tool_calls"] = []
                if isinstance(raw_content, list):
                    updates["content"] = content
                if updates:
                    m = m.model_copy(update=updates)
                compacted.append(m)
                break
        # 当前轮原样保留
        compacted.extend(messages[human_idx[-1]:])
        return compacted

    # 节点内真空响应重试次数（独立于 _LLM_MAX_RETRIES 网络层重试）。
    _MAX_EMPTY_RETRIES = 2

    @staticmethod
    def _normalize_anthropic_content_to_str(content) -> str:
        """ChatAnthropic 在响应含 thinking / tool_use 块时返 ``content: list[dict]``，
        其它情况返 ``str``。本 helper 把任何形态规整成纯文本字符串：

          * ``str``  → 原样返
          * ``list`` → 遍历 block，type='text' 取 ``text`` 拼接，其它 type
                       （thinking / tool_use / tool_result）忽略
          * 其它    → 返空串

        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    t = b.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            return "".join(parts)
        return ""

    @classmethod
    def _is_empty_response(cls, resp) -> bool:
        """判定 LLM response 是否真空——content 空 + 无 valid tool_calls
        + 无 invalid tool_calls。三者同时满足说明 LLM 既没说话也没决定调
        工具，该 response 喂下游会触发兜底空回复。

        ChatAnthropic 的 ``content`` 可能是 list[dict]：先 normalize 成 str
        再判空；list 里只要含 tool_use 块、tool_calls 字段也会被填充，所以
        判 tool_calls 仍能识别"虽然 content 文本空但模型决策调工具"的情况。
        """
        c = cls._normalize_anthropic_content_to_str(getattr(resp, "content", ""))
        if c.strip():
            return False
        if getattr(resp, "tool_calls", None):
            return False
        if getattr(resp, "invalid_tool_calls", None):
            return False
        return True

    async def _invoke_and_record_metrics(
        self, prompt_messages: list, config: RunnableConfig
    ):
        """单次 LLM call + 抽 metrics + push log，返回 response。

        一轮内多次 LLM call 时每次都 push 一条 metrics，
        ``dialogue_simulator._aggregate_metrics_from`` 会跨 call 求和 token /
        logid 逗号拼接。重试场景透明继承同样口径。
        """
        t_start = time.time()
        if self.streaming:
            resp, ttft = await self._astream_with_retry(prompt_messages, config)
        else:
            resp = await self._ainvoke_with_retry(prompt_messages, config)
            ttft = None
        total = int((time.time() - t_start) * 1000)
        if ttft is None:
            ttft = total

        # todo:打印下resp看下是否是thinking模式
        # logger.info(f"agent_thought:***{resp}***")

        pt = 0
        ct = 0
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            pt = usage.get("input_tokens", 0)
            ct = usage.get("output_tokens", 0)
        else:
            meta = getattr(resp, "response_metadata", None) or {}
            tu = meta.get("token_usage", {})
            pt = tu.get("prompt_tokens", 0)
            ct = tu.get("completion_tokens", 0)
        rid = getattr(resp, "id", "") or ""
        # reasoning_tokens：vertex 路径由 VertexChat 把
        # usageMetadata.thoughtsTokenCount 挂到 additional_kwargs；其余路径
        # （OpenAI 兼容协议）拿不到该字段就为 0。不并入 completion_tokens。
        rt = int(
            (getattr(resp, "additional_kwargs", None) or {}).get(
                "reasoning_tokens", 0
            )
            or 0
        )
        self.llm_metrics_log.append(
            {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "reasoning_tokens": rt,
                "ttft_ms": ttft,
                "total_ms": total,
                "streaming": self.streaming,
                "response_id": rid,
            }
        )
        return resp

    async def _retry_if_empty_response(
        self,
        response,
        prompt_messages: list,
        config: RunnableConfig,
        *,
        log_prefix: str = "[Agent LLM]",
        retry_hint_text: str | None = None,
    ):
        """检测真空响应；若真空则节点内重试 _MAX_EMPTY_RETRIES 次。

        每次重试在 ``prompt_messages`` 末尾追加一条 HumanMessage hint 提示
        模型上一步响应为空，要求重新输出（**只用于本次重试调用**，不进
        state.messages、不污染 csv）。每次重试都 push 一条 metrics（与既有
        "一轮多次 LLM call" 行为一致）。

        ``retry_hint_text=None`` 用通用 hint。

        最终 response 即使仍空也直接返回——调用方负责后续兜底（
        进 state.messages 后 should_continue 转 end）。
        """
        if not self._is_empty_response(response):
            return response

        # vertex 路径特殊处理：当 prompt 末尾是 ToolMessage 时，追加 hint
        # HumanMessage 会被 ``_messages_to_vertex._append_part`` 合并到同
        # 一个 user content 的 parts，导致 ``functionResponse + text``
        # 混合。vertex Gemini 对这种混合 part 解析异常（实测继续返空响应
        # / 截断），故跳过 hint 注入，原样重试。
        # ChatOpenAI 路径无此限制（OpenAI 允许 tool message 后跟 user
        # message），保留 hint 行为不变。
        inner_client = getattr(self._llm_client, "bound", self._llm_client)
        is_vertex = type(inner_client).__name__ == "VertexChat"
        last_is_tool = bool(prompt_messages) and getattr(
            prompt_messages[-1], "type", ""
        ) == "tool"
        skip_hint = is_vertex and last_is_tool

        if skip_hint:
            logger.warning(
                "%s vertex 路径末尾是 ToolMessage，跳过 hint 注入"
                "（避免 user content 内混合 functionResponse+text 触发 gemini 异常）",
                log_prefix,
            )
            retry_messages = list(prompt_messages)
        else:
            if retry_hint_text is None:
                retry_hint_text = (
                    "上一步 LLM 响应为空（既无 content 也无 tool_calls）。"
                    "请基于当前对话与工具结果重新输出有效回复（如需调用工具请"
                    "输出 tool_calls，否则直接输出文本回复）；不要返回空响应。"
                )
            retry_hint = HumanMessage(content=retry_hint_text)
            retry_messages = list(prompt_messages) + [retry_hint]
        for attempt in range(1, self._MAX_EMPTY_RETRIES + 1):
            logger.warning(
                "%s 检测到真空响应，节点内重试 %d/%d",
                log_prefix, attempt, self._MAX_EMPTY_RETRIES,
            )
            response = await self._invoke_and_record_metrics(
                retry_messages, config
            )
            if not self._is_empty_response(response):
                logger.info(
                    "%s 真空重试第 %d 次救回", log_prefix, attempt,
                )
                return response
        logger.warning(
            "%s 真空重试 %d 次后仍空，按原逻辑继续",
            log_prefix, self._MAX_EMPTY_RETRIES,
        )
        return response

    # ── invalid_tool_calls 自纠 ──────────────────────────────────────────
    # LangChain 在 OpenAI/Vertex 协议把 LLM 给的非法 JSON arguments
    # 从顶层 ``tool_calls`` 移到 ``invalid_tool_calls``，并附 error。
    # 实测样例：
    #   * arguments 漏右括号 / 重复 key（langchain OUTPUT_PARSING_FAILURE）
    #   * arguments 含 // 行注释 / 尾逗号
    # 此时 content 也常空，``_is_empty_response`` 判 not empty（因为
    # invalid_tool_calls 非空）→ 不进真空重试；但 ``should_continue``
    # 看 tool_calls 为空 → 直接 end，整轮静默截断。本机制把错误回喂给
    # LLM 让它重出合法 JSON。
    _MAX_INVALID_TOOL_CALL_RETRIES = 2

    @staticmethod
    def _has_invalid_tool_calls(resp) -> bool:
        return bool(getattr(resp, "invalid_tool_calls", None))

    def _get_tool_schema_text(self, tool_name: str) -> str:
        """渲染指定 tool 的 args JSON schema 为紧凑字符串，供 hint 注入。

        优先 ``tool.args``（LangChain 已规范化的 inputSchema dict），缺失
        时回退 ``tool.args_schema.model_json_schema()``（pydantic 模型）。
        最长 800 字符截断防 prompt 爆炸。
        """
        tool = self.tools_by_name.get(tool_name)
        if tool is None:
            return f"<工具 {tool_name!r} 未在 tools_by_name 中，无 schema 可参考>"
        try:
            schema = getattr(tool, "args", None)
            if not schema:
                args_schema = getattr(tool, "args_schema", None)
                if args_schema is not None and hasattr(args_schema, "model_json_schema"):
                    schema = args_schema.model_json_schema()
            if not schema:
                return "<schema 不可用>"
            text = json.dumps(schema, ensure_ascii=False)
            if len(text) > 800:
                text = text[:800] + "...[truncated]"
            return text
        except Exception as e:  # noqa: BLE001
            return f"<schema 渲染失败: {type(e).__name__}: {e}>"

    def _build_invalid_tool_calls_hint(self, invalid_tool_calls: list) -> str:
        """把 invalid_tool_calls 列表渲染成给 LLM 的修正指令文本。

        包含：
          * 每条错误的 tool 名 / 原始 args（截 500 字）/ LangChain 给的
            error 描述（截 300 字）
          * 该 tool 的 JSON Schema（截 800 字），让 LLM 直接对照参数名 / 类型
          * 通用 JSON 规则提醒（双引号、无尾逗号、无注释、无重复 key）
        """
        error_lines: list[str] = []
        for itc in invalid_tool_calls or []:
            name = itc.get("name") or "<no name>"
            args_raw = str(itc.get("args") or "")[:500]
            err = str(itc.get("error") or "<unparsed>")[:300]
            schema_text = self._get_tool_schema_text(name)
            error_lines.append(
                f"- 工具 [{name}] arguments 解析失败：\n"
                f"    args:   {args_raw}\n"
                f"    error:  {err}\n"
                f"    schema: {schema_text}"
            )
        errors_text = "\n".join(error_lines)
        return (
            "上一步尝试调用工具但 arguments 不是合法 JSON：\n"
            f"{errors_text}\n\n"
            "请在下一步重新输出工具调用，严格遵守：\n"
            "  * arguments 必须是合法 JSON（双引号字符串、无尾逗号、无 //"
            " /* */ 注释）；\n"
            "  * 不允许重复 key；\n"
            "  * 字段名 / 类型严格对齐上面 schema；\n"
            "  * 数值用 number 不要带引号，除非 schema 要求字符串；\n"
            "  * 如无需调用工具，直接输出文本回复。"
        )

    async def _retry_if_invalid_tool_calls(
        self,
        response,
        prompt_messages: list,
        config: RunnableConfig,
        *,
        log_prefix: str = "[Agent LLM]",
    ):
        """检测 invalid_tool_calls；若有则节点内重试 _MAX_INVALID_TOOL_CALL_RETRIES 次。

        - 重试时在 prompt 末尾追加一条 HumanMessage hint（含错误 + schema + 修正
          指令）；vertex 路径 + 末尾是 ToolMessage 时跳过 hint 注入（避免
          functionResponse + text 混合 part 触发 gemini 解析异常，与
          ``_retry_if_empty_response`` 同款 skip_hint 逻辑）。
        - 每次重试都会 push 一条 metrics（与一轮多次 LLM call 行为一致）。
        - 最终响应即使仍 invalid 也直接返回——下游 ``should_continue`` 走 end，
          metrics_log 已记录所有重试调用便于事后统计。
        """
        if not self._has_invalid_tool_calls(response):
            return response

        inner_client = getattr(self._llm_client, "bound", self._llm_client)
        is_vertex = type(inner_client).__name__ == "VertexChat"
        last_is_tool = bool(prompt_messages) and getattr(
            prompt_messages[-1], "type", ""
        ) == "tool"
        skip_hint = is_vertex and last_is_tool

        if skip_hint:
            logger.warning(
                "%s vertex+ToolMessage 末尾，跳过 invalid_tool_calls hint 注入"
                "（避免 functionResponse+text 混合 part 触发 gemini 异常），"
                "原 prompt 重试",
                log_prefix,
            )
            retry_messages = list(prompt_messages)
        else:
            hint = self._build_invalid_tool_calls_hint(
                getattr(response, "invalid_tool_calls", []) or []
            )
            retry_messages = list(prompt_messages) + [HumanMessage(content=hint)]

        for attempt in range(1, self._MAX_INVALID_TOOL_CALL_RETRIES + 1):
            invalid_count = len(getattr(response, "invalid_tool_calls", []) or [])
            logger.warning(
                "%s 检测到 %d 条 invalid_tool_calls，节点内重试 %d/%d",
                log_prefix, invalid_count,
                attempt, self._MAX_INVALID_TOOL_CALL_RETRIES,
            )
            response = await self._invoke_and_record_metrics(
                retry_messages, config
            )
            if not self._has_invalid_tool_calls(response):
                logger.info(
                    "%s invalid_tool_calls 重试第 %d 次救回",
                    log_prefix, attempt,
                )
                return response
        logger.warning(
            "%s invalid_tool_calls 重试 %d 次后仍非法，透传最终响应",
            log_prefix, self._MAX_INVALID_TOOL_CALL_RETRIES,
        )
        return response

    async def call_llm(self, state: AgentState, config: RunnableConfig):
        """LangGraph agent 节点：异步调 LLM 并记录 token 指标。

        覆盖路径：``_invoke_and_record_metrics`` 拿首次响应 →
        ``_retry_if_empty_response`` 检测真空时节点内重试最多 2 次 →
        ``_retry_if_invalid_tool_calls`` 检测 LangChain 抓到的非法 JSON
        arguments 时节点内重试最多 2 次（每次重试 prompt 末尾追加 hint
        含错误描述 + 该 tool 的 JSON Schema）。两个重试都不消耗 React
        loop iteration。最终 response 即使仍空 / 仍
        invalid 也透传到 state，由 ``should_continue`` 判定结束；
        ``_ainvoke_with_retry`` 内的 ``_stash_empty_dump_if_needed`` 已
        把 raw response repr 挂到 ``additional_kwargs['empty_response_dump']``，
        下游 ``_extract_agent_output`` 透传到 ``DialogueTurn.empty_response_dump``
        落 csv，便于事后排查 vertex / 其它后端的真空响应。
        """
        # 仅保留 YAML 实际引用到的字段，避免「dict 收集了但 prompt 用不上」
        # 的死代码。persona / full_intent 故意不暴露给被测 agent。
        prompt_context = {
            "query": state.get("query", "") or "",
            "context": state.get("context", "") or "",
            "time": state.get("time", "") or "",
            "tools_brief": self._tools_brief,
        }
        system_prompt_text = self._render_prompt(prompt_context)
        system_prompt = SystemMessage(content=system_prompt_text)

        # 标准 React：完整把 AIMessage(tool_calls) + ToolMessage 配对传给 LLM
        full_messages = list(state["messages"])
        # 跨轮压缩：只影响发给 LLM 的副本，不动 state（state 仍用于落盘 / 抽取）
        messages = self._compact_history(full_messages)

        # info 级只打要点；详细 dump 降为 debug，避免长会话刷屏
        # logger.info(
        #     "[Agent LLM] msgs=%d→%d (human=%d, ai=%d, tool=%d)",
        #     len(full_messages),
        #     len(messages),
        #     sum(1 for m in messages if getattr(m, "type", "") == "human"),
        #     sum(1 for m in messages if getattr(m, "type", "") == "ai"),
        #     sum(1 for m in messages if getattr(m, "type", "") == "tool"),
        # )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[Agent LLM] system_prompt:\n%s", system_prompt_text)
            logger.debug(
                "[Agent LLM] messages:\n%s",
                json.dumps(
                    [
                        {
                            "type": getattr(m, "type", "unknown"),
                            "content": (getattr(m, "content", "") or "")[:200],
                            "tool_calls": getattr(m, "tool_calls", None),
                            "tool_call_id": getattr(m, "tool_call_id", None),
                        }
                        for m in messages
                    ],
                    ensure_ascii=False,
                    default=str,
                ),
            )

        prompt_messages = [system_prompt] + messages
        response = await self._invoke_and_record_metrics(prompt_messages, config)

        response = await self._retry_if_empty_response(
            response, prompt_messages, config, log_prefix="[Agent LLM]",
        )
        # 顺序：先空响应再 invalid_tool_calls。empty 重试若救回一个带
        # invalid_tool_calls 的响应，invalid 重试会接力处理；反过来 invalid
        # 重试不识别 empty。
        response = await self._retry_if_invalid_tool_calls(
            response, prompt_messages, config, log_prefix="[Agent LLM]",
        )


        # logger.info(f"agent_thought:***{response}***")

        return {"messages": [response]}

    def should_continue(self, state: AgentState):
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None)
        return "continue" if tool_calls else "end"

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.call_llm)
        workflow.add_node("tools", self.tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", self.should_continue, {"continue": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile(checkpointer=self._memory)

    def run_graph(self):
        """返回编译好的图。整个 Agent 实例共用同一份 graph + MemorySaver，
        以便跨轮通过 thread_id 续接对话历史。"""
        return self._graph
