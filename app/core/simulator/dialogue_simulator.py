import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

from app.core.simulator.agent_simulator import BaseSimulationAgent
from app.core.simulator.base import STOP
from app.core.simulator.user_simulator import UserSimulator
from app.schemas.dialogue_simulator import DialogueCase, DialogueResult, DialogueTurn

if TYPE_CHECKING:
    # 仅用于类型注解，运行时不导入避免与 dialogue_recorder 循环依赖
    from app.services.dialogue_recorder import DialogueResultWriter

logger = logging.getLogger(__name__)


class DialogueSimulator:
    """多轮对话仿真引擎。

    编排 BaseSimulationAgent（被测智能体）与 UserSimulator（用户模拟器）的
    完整多轮交互；每生成一条 DialogueTurn 通过 ``writer.append(turn)`` 由
    DialogueResultWriter 缓冲写入 xlsx，避免长会话因异常中断丢失中间数据。

    turn_index 编号约定（与 role 联合唯一定位，B2 修复）：
      - 初始 user 行          : turn_index=0
      - 第 t 轮 agent 回复    : turn_index=t, role=assistant
      - 第 t 轮 user 追问     : turn_index=t, role=user
      - 触底强制截断的 user 行 : turn_index=max_turns + 1, role=user, is_forced_stop=True
    """

    # 整段 graph.ainvoke 的预算（含一轮 React loop 多次 LLM 调用 + 多个 tool 调用 +
    # call_llm 的 retry 退避）。设为 950s 以容纳「单次 LLM 5 分钟 × 3 次尝试 + 退避」
    # 的最坏路径（_LLM_REQUEST_TIMEOUT=300 × 3 + 1+2s = 903s + 缓冲）。
    AGENT_TURN_TIMEOUT = 950.0
    # UserSimulator 单次 LLM 调用超时。原 60s 在大模型 / thinking 模型偶发慢响应时
    # 会触发 timeout（实测 0528_01 首轮 claude-sonnet-4-6 / qwen3-30b / gemini-pro
    # / gemini-flash-lite 各产 1-3 次超时），把超时单元判为"timeout"导致 case 强行
    # 截断、对话不完整。300s（5 分钟）能覆盖 thinking 模型的长 reasoning 路径，且
    # 大幅大于 OpenAI / 兼容协议默认网络层 timeout（120s），单 LLM 长尾足够吸收。
    USER_TURN_TIMEOUT = 300.0

    # LangGraph 默认 recursion_limit=25，每个节点执行算 1 step，React 一轮
    # = agent + tools = 2 step，约能跑 12-13 次 LLM-工具往返。提到 50 给合理
    # 高频调用（多 POI 串行查询、路径规划多段）留足空间；触底仍说明
    # LLM 进入纠错死循环（典型：tool 入参类型校验反复失败）。
    GRAPH_RECURSION_LIMIT = 50

    def __init__(
        self,
        agent: BaseSimulationAgent,
        user: UserSimulator,
        conversation_id: str,
        writer: "DialogueResultWriter | None" = None,
        max_turns: int = 20,
    ):
        self.agent_simulator = agent
        self.user_simulator = user
        self.max_turns = max_turns
        self.writer = writer

        self.history: list[dict[str, str]] = []
        self.conversation_id = conversation_id

    async def simulate(
        self,
        case: DialogueCase,
    ) -> DialogueResult:
        """对单条用例执行完整的多轮对话仿真。

        Returns:
            DialogueResult 包含初始用户 query、每一轮 agent / user 回复，
            以及自然 / 强制 / 异常 终止状态。
        """
        turns: list[DialogueTurn] = []
        self.history = []
        thread_id = self.conversation_id

        # ── 初始 user 行（turn_index=0；B2 联合定位避免与第 1 轮 user 撞键）──
        user_src = DialogueTurn(
            conversation_id=self.conversation_id,
            turn_index=0,
            role="user",
            content=case.query,
            query=case.query,
            context=case.context,
            time=case.time,
            location=case.location,
            tool=case.tool,
            ground_truth=case.ground_truth,
        )
        turns.append(user_src)
        await self._persist_turn(user_src)
        self.history.append({"role": "user", "content": case.query})

        logger.info(
            f"[{self.conversation_id}] 评测开始 | query={case.query[:80]}"
        )

        natural_stop = False

        for turn_idx in range(1, self.max_turns + 1):
            # ── 一轮 agent 仿真 ──
            agent_turn = await self._run_agent_turn(
                case=case,
                thread_id=thread_id,
                conversation_id=self.conversation_id,
                turn_idx=turn_idx,
            )
            turns.append(agent_turn)
            await self._persist_turn(agent_turn)
            self.history.append(
                {"role": "assistant", "content": agent_turn.content}
            )

            if agent_turn.status != "success":
                logger.warning(
                    f"[{self.conversation_id}] Agent 第 {turn_idx} 轮 "
                    f"状态异常({agent_turn.status})，终止仿真"
                )
                break

            # ── 一轮用户仿真 ──
            user_turn = await self._run_user_turn(
                self.user_simulator,
                case,
                self.conversation_id,
                turn_idx,
            )
            # user 行的 LLM 调用是 user_simulator 单次：直接取 LLM API 响应里的真值
            user_turn.input_tokens = self.user_simulator.last_prompt_tokens or 0
            user_turn.output_tokens = self.user_simulator.last_completion_tokens or 0
            user_turn.logid = self.user_simulator.last_response_id or ""
            turns.append(user_turn)
            await self._persist_turn(user_turn)
            self.history.append({"role": "user", "content": user_turn.content})

            if user_turn.is_stop:
                # 仅当不是 forced_stop 也不是 error/timeout 时，才算自然终止
                natural_stop = (
                    not user_turn.is_forced_stop
                    and user_turn.status == "success"
                )
                break
            if user_turn.status != "success":
                break
        else:
            # for-else：循环跑满 max_turns 没有 break，强制截断
            logger.warning(
                f"[{self.conversation_id}] 达到最大轮次 {self.max_turns}，强制终止"
            )
            forced_stop_turn = DialogueTurn(
                conversation_id=self.conversation_id,
                # B2: max_turns+1 与最后一轮 user_turn (turn_index=max_turns) 区分，
                # 联合 (turn_index, role) 仍唯一
                turn_index=self.max_turns + 1,
                role="user",
                content=STOP,
                is_stop=True,
                is_forced_stop=True,
                query=case.query,
                context=case.context,
                time=case.time,
                location=case.location,
                tool=case.tool,
                ground_truth=case.ground_truth,
            )
            turns.append(forced_stop_turn)
            await self._persist_turn(forced_stop_turn)

        return DialogueResult(
            conversation_id=self.conversation_id,
            case=case,
            turns=turns,
            total_turns=len(turns),
            is_natural_stop=natural_stop,
        )

    async def _run_agent_turn(
        self,
        *,
        case: DialogueCase,
        thread_id: str,
        conversation_id: str,
        turn_idx: int = 0,
    ) -> DialogueTurn:
        """执行单轮 Agent 调用。

        每轮以 ``self.history`` 中最后一条 user 消息作为新 HumanMessage 注入；
        history 为空（如单测直接调用）时回退到 ``case.query``。LangGraph 通过
        ``thread_id`` + ``MemorySaver`` 自动续接历史，所以每轮只需追加一条新消息。
        """
        if self.history and self.history[-1].get("role") == "user":
            user_message = self.history[-1]["content"]
        else:
            user_message = getattr(case, "query", "") or ""

        # 记录调用前的 LLM 指标位置，用于本轮 React loop 内累计 completion_tokens
        # （一轮可能含多次 LLM call：第 1 次出 tool_calls → tool 执行 → 第 2 次出最终回复）
        metrics_mark = len(getattr(self.agent_simulator, "llm_metrics_log", []) or [])

        try:
            graph = self.agent_simulator.run_graph()
            graph_input = {
                "messages": [HumanMessage(content=user_message)],
                "query": getattr(case, "query", "") or "",
                "context": getattr(case, "context", "") or "",
                "time": getattr(case, "time", "") or "",
                "location": getattr(case, "location", "") or "",
            }
            graph_config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": self.GRAPH_RECURSION_LIMIT,
            }

            start = time.time()
            result_state = await asyncio.wait_for(
                graph.ainvoke(graph_input, config=graph_config),
                timeout=self.AGENT_TURN_TIMEOUT,
            )

            content, tool_calls, empty_dump = self._extract_agent_output(result_state)
            elapsed = int((time.time() - start) * 1000)

            # 取本轮所有 LLM call 的 token 与 response_id 聚合：
            # tokens 跨 call 求和、response_id 逗号拼接
            input_total, output_total, reasoning_total, logid_csv = (
                self._aggregate_metrics_from(metrics_mark)
            )

            return DialogueTurn(
                conversation_id=conversation_id,
                turn_index=turn_idx,
                role="assistant",
                content=content,
                tool_calls=tool_calls,
                execution_time_ms=elapsed,
                query=getattr(case, "query", "") or "",
                context=getattr(case, "context", None),
                time=getattr(case, "time", None),
                location=getattr(case, "location", None),
                tool=getattr(case, "tool", None),
                ground_truth=getattr(case, "ground_truth", None),
                input_tokens=input_total,
                output_tokens=output_total,
                reasoning_tokens=reasoning_total,
                logid=logid_csv,
                llm_metrics=self._dump_last_llm_metrics(),
                empty_response_dump=empty_dump,
            )

        except asyncio.TimeoutError:
            logger.exception(
                f"[{conversation_id}] Agent 第 {turn_idx} 轮调用超时"
            )
            return DialogueTurn(
                conversation_id=conversation_id,
                turn_index=turn_idx,
                role="assistant",
                content="",
                status="timeout",
                execution_time_ms=int(self.AGENT_TURN_TIMEOUT * 1000),
                error_message=f"Agent 调用超时（{self.AGENT_TURN_TIMEOUT}s）",
                query=getattr(case, "query", "") or "",
                context=getattr(case, "context", None),
                time=getattr(case, "time", None),
                location=getattr(case, "location", None),
                tool=getattr(case, "tool", None),
                ground_truth=getattr(case, "ground_truth", None),
                tool_calls=[],
            )

        except GraphRecursionError as e:
            # React loop 触底：通常 LLM 反复纠错失败、或工具调用陷入循环。
            # 不同于一般 Exception——已经累积的中间消息（含 tool_calls 与
            # ToolMessage）都在 checkpointer 的 state 里。best-effort 取出
            # 来作为 agent 回复，至少 turn 不空、对话不被强行截短。
            elapsed = int((time.time() - start) * 1000)
            recovered_content = ""
            recovered_tool_calls: list[dict] = []
            recovered_empty_dump: str | None = None
            try:
                snapshot = await graph.aget_state(graph_config)
                state_values = getattr(snapshot, "values", None) or {}
                recovered_content, recovered_tool_calls, recovered_empty_dump = (
                    self._extract_agent_output(state_values)
                )
            except Exception as snap_err:  # pragma: no cover - defensive
                logger.warning(
                    "[%s] GraphRecursionError 后从 checkpointer 还原失败: %s",
                    conversation_id, snap_err,
                )

            input_total, output_total, reasoning_total, logid_csv = (
                self._aggregate_metrics_from(metrics_mark)
            )
            logger.warning(
                "[%s] Agent 第 %d 轮触发 GraphRecursionError "
                "(limit=%d)；已还原 %d 条 tool_calls + content 长度=%d",
                conversation_id, turn_idx, self.GRAPH_RECURSION_LIMIT,
                len(recovered_tool_calls), len(recovered_content),
            )
            return DialogueTurn(
                conversation_id=conversation_id,
                turn_index=turn_idx,
                role="assistant",
                content=recovered_content,
                status="recursion_limit",
                execution_time_ms=elapsed,
                error_message=(
                    f"GraphRecursionError: recursion_limit={self.GRAPH_RECURSION_LIMIT} "
                    f"触底，LLM 未在限定步数内收敛到无 tool_calls 的最终回复；"
                    f"已 best-effort 还原中间消息：tool_calls={len(recovered_tool_calls)}"
                ),
                query=getattr(case, "query", "") or "",
                context=getattr(case, "context", None),
                time=getattr(case, "time", None),
                location=getattr(case, "location", None),
                tool=getattr(case, "tool", None),
                ground_truth=getattr(case, "ground_truth", None),
                tool_calls=recovered_tool_calls,
                input_tokens=input_total,
                output_tokens=output_total,
                reasoning_tokens=reasoning_total,
                logid=logid_csv,
                llm_metrics=self._dump_last_llm_metrics(),
                empty_response_dump=recovered_empty_dump,
            )

        except Exception as e:
            logger.exception(
                f"[{conversation_id}] Agent 第 {turn_idx} 轮调用失败"
            )
            # vertex / 其它后端的 HTTP 4xx/5xx 异常若携带 ``body_text``
            # 属性（如 VertexChat.VertexHTTPError），把完整服务端
            # 返回体写到 empty_response_dump，error_message 保留 str(e) 的
            # 简短形态（含 url + body 前 200 字符）；事后 csv 单字段就能
            # 看完整 vertex 错误原因，无需翻日志。
            error_body = getattr(e, "body_text", None) or None
            return DialogueTurn(
                conversation_id=conversation_id,
                turn_index=turn_idx,
                role="assistant",
                content="",
                status="error",
                execution_time_ms=0,
                error_message=str(e)[:500],
                query=getattr(case, "query", "") or "",
                context=getattr(case, "context", None),
                time=getattr(case, "time", None),
                location=getattr(case, "location", None),
                tool=getattr(case, "tool", None),
                ground_truth=getattr(case, "ground_truth", None),
                tool_calls=[],
                empty_response_dump=error_body,
            )

    async def _run_user_turn(
        self,
        simulator: UserSimulator,
        case: DialogueCase,
        conversation_id: str,
        turn_idx: int,
    ) -> DialogueTurn:
        """调用 UserSimulator 生成用户追问。

        将 ``self.history`` 作为 conversation_history 传入，保证 UserSimulator
        始终接收到同一 conversation_id 下按序追加的完整对话历史。
        """
        try:
            start = time.time()
            user_reply = await asyncio.wait_for(
                simulator.generate_next_message(self.history),
                timeout=self.USER_TURN_TIMEOUT,
            )
            elapsed = int((time.time() - start) * 1000)
            is_stop = UserSimulator.is_stop(user_reply)

            llm_messages_json = json.dumps(
                simulator.last_llm_messages or [],
                ensure_ascii=False,
                default=str,
            )
            return DialogueTurn(
                conversation_id=conversation_id,
                turn_index=turn_idx,
                role="user",
                content=user_reply,
                is_stop=is_stop,
                execution_time_ms=elapsed,
                query=case.query,
                context=case.context,
                time=case.time,
                location=case.location,
                tool=case.tool,
                ground_truth=case.ground_truth,
                user_simulator_input=llm_messages_json,
            )

        except asyncio.TimeoutError:
            logger.exception(
                f"[{conversation_id}] UserSimulator 第 {turn_idx} 轮调用超时"
            )
            return DialogueTurn(
                conversation_id=conversation_id,
                turn_index=turn_idx,
                role="user",
                content=STOP,
                is_stop=True,
                is_forced_stop=True,
                status="timeout",
                execution_time_ms=int(self.USER_TURN_TIMEOUT * 1000),
                error_message=f"UserSimulator 调用超时（{self.USER_TURN_TIMEOUT}s）",
                query=case.query,
                context=case.context,
                time=case.time,
                location=case.location,
                tool=case.tool,
                ground_truth=case.ground_truth,
            )

        except Exception as e:
            logger.exception(
                f"[{conversation_id}] UserSimulator 第 {turn_idx} 轮调用失败"
            )
            return DialogueTurn(
                conversation_id=conversation_id,
                turn_index=turn_idx,
                role="user",
                content=STOP,
                is_stop=True,
                status="error",
                execution_time_ms=0,
                error_message=str(e),
                query=case.query,
                context=case.context,
                time=case.time,
                location=case.location,
                tool=case.tool,
                ground_truth=case.ground_truth,
            )

    async def _persist_turn(self, turn: DialogueTurn) -> None:
        """通过共享 writer 异步追加一条 turn。

        writer 为 None（如单测）时直接 no-op；写入失败仅记 warning，不影响主流程。
        """
        if self.writer is None:
            return
        try:
            await self.writer.append(turn)
        except Exception as e:
            logger.warning(
                f"[{self.conversation_id}] 写入 turn 失败 "
                f"(turn_index={turn.turn_index}, role={turn.role}): {e}"
            )

    def _dump_last_llm_metrics(self) -> str | None:
        """序列化 BaseSimulationAgent 当轮 LLM 指标，写入 turn.llm_metrics。"""
        log = getattr(self.agent_simulator, "llm_metrics_log", None)
        if not log:
            return None
        try:
            return json.dumps(log[-1], ensure_ascii=False, default=str)
        except Exception:
            return None

    def _aggregate_metrics_from(self, mark: int) -> tuple[int, int, int, str]:
        """聚合本轮 ``llm_metrics_log[mark:]`` 的 LLM 调用指标，用于 agent_turn。

        Returns:
            (input_tokens_sum, output_tokens_sum, reasoning_tokens_sum,
             response_ids_csv)

            一轮 React loop 内可能多次 LLM 调用：所有 token 字段跨 call 求和；
            response_id 逗号拼接全部，便于服务端按 logid 追溯。
            reasoning_tokens 仅 Vertex 路径（gemini 模型）有非 0 值，
            其它路径恒为 0。
        """
        log = getattr(self.agent_simulator, "llm_metrics_log", None) or []
        prompt_total = 0
        output_total = 0
        reasoning_total = 0
        ids: list[str] = []
        for entry in log[mark:]:
            try:
                prompt_total += int(entry.get("prompt_tokens", 0) or 0)
            except (TypeError, ValueError):
                pass
            try:
                output_total += int(entry.get("completion_tokens", 0) or 0)
            except (TypeError, ValueError):
                pass
            try:
                reasoning_total += int(entry.get("reasoning_tokens", 0) or 0)
            except (TypeError, ValueError):
                pass
            rid = entry.get("response_id", "")
            if rid:
                ids.append(str(rid))
        return prompt_total, output_total, reasoning_total, ",".join(ids)

    @staticmethod
    def _extract_agent_output(result_state: dict) -> tuple[str, list[dict], str | None]:
        """从 AgentState 中提取**本轮**的最终回复、tool_calls 与诊断 dump。

        切分规则：
        - 以 messages 中**最后一条 HumanMessage** 为分界，其后的所有消息属于本轮
          （checkpointer 复用同一个 thread，state 会累计历史轮）
        - content：本轮内最后一条 AI 消息的 content（即 React loop 出口的最终回复）
        - tool_calls：本轮内**所有** AI 消息的 tool_calls 聚合，并按 tool_call_id
          关联各自的 ToolMessage 响应
        - empty_response_dump：仅显式 ReAct 真空响应时由 reasoning_node 挂到出口
          AIMessage 的 ``additional_kwargs['empty_response_dump']``；其余场景 None

        ⚠️ 注意：不能只看最后一条 AI 消息——React loop 出口的 AI 消息按定义没有
        tool_calls，否则 should_continue 就不会 'end'。必须聚合本轮所有 AI 消息。
        """
        messages = result_state.get("messages", [])

        # 1) 全局收集 tool_call_id → ToolMessage.content 的映射
        #    tool_call_id 全局唯一，跨轮收集也安全
        tool_responses: dict[str, str] = {}
        for msg in messages:
            if getattr(msg, "type", "") == "tool":
                tid = getattr(msg, "tool_call_id", "")
                tool_responses[tid] = str(getattr(msg, "content", "") or "")

        # 2) 找本轮起点：最后一条 HumanMessage 的索引
        last_human_idx = -1
        for i, msg in enumerate(messages):
            if getattr(msg, "type", "") == "human":
                last_human_idx = i
        current_turn = (
            messages[last_human_idx + 1:] if last_human_idx >= 0 else list(messages)
        )

        # 3) 聚合本轮 tool_calls + 取最后一条 AI 消息的 content 与诊断 dump
        content = ""
        tool_calls: list[dict] = []
        empty_response_dump: str | None = None
        for msg in current_turn:
            if getattr(msg, "type", "") != "ai":
                continue
            raw_tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in raw_tool_calls:
                tc_id = tc.get("id", "")
                tool_calls.append(
                    {
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                        "id": tc_id,
                        "response": tool_responses.get(tc_id, ""),
                    }
                )
            # 持续覆盖 → 循环结束后 content 与 dump 是本轮最后一条 AI 的
            # ChatAnthropic（thinking on / tool_use）会把 content 返成
            # list[{type:"thinking"|"text"|"tool_use", ...}]，直接 str() 会
            # 把 thinking 块的 signature（base64，~3KB）一起塞进 CSV / history。
            # 走 normalize 只保留 text 块拼接结果。
            content = BaseSimulationAgent._normalize_anthropic_content_to_str(
                getattr(msg, "content", "")
            )
            extras = getattr(msg, "additional_kwargs", None) or {}
            dump = extras.get("empty_response_dump")
            empty_response_dump = dump if dump else None

        return content, tool_calls, empty_response_dump
