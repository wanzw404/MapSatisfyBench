import asyncio
import csv
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path

from app.core.simulator import DialogueSimulator
from app.core.simulator.agent_simulator import BaseSimulationAgent
from app.core.tools.manager import tools as all_tools
from app.config import settings
from app.paths import SIMULATOR_RES_DIR as OUTPUT_DIR
from app.schemas.dialogue_simulator import DialogueCase, DialogueResult, DialogueTurn
from app.services.user_simulator_factory import build_user_simulator

logger = logging.getLogger(__name__)


OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "outputs" / "simulator_res"

# 输出列定义（按顺序）
OUTPUT_HEADERS = [
    "conversation_id",
    "turn_index",
    "role",
    "content",
    "tool_calls",
    "is_stop",
    "is_forced_stop",
    "status",
    "execution_time_ms",
    "error_message",
    "query",
    "context",
    "time",
    "location",
    "tool",
    "user_simulator_input",
    "llm_metrics",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "logid",
    "ground_truth",
    "timestamp",
    "empty_response_dump",
]

def _turn_to_row(turn: DialogueTurn) -> list:
    return [
        turn.conversation_id,
        turn.turn_index,
        turn.role,
        turn.content,
        str(turn.tool_calls) if turn.tool_calls else "",
        "true" if turn.is_stop else "false",
        "true" if turn.is_forced_stop else "false",
        turn.status,
        turn.execution_time_ms,
        turn.error_message or "",
        turn.query,
        turn.context or "",
        turn.time or "",
        turn.location or "",
        turn.tool or "",
        turn.user_simulator_input or "",
        turn.llm_metrics or "",
        turn.input_tokens,
        turn.output_tokens,
        turn.reasoning_tokens,
        turn.logid or "",
        turn.ground_truth or "",
        turn.timestamp,
        turn.empty_response_dump or "",
    ]


def _sanitize_filename_suffix(suffix: str) -> str:
    """把 model 名压成文件系统友好的 suffix：保留 [a-zA-Z0-9._-]，其余 → _，截断到 60 字符。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", suffix or "").strip("_")
    return cleaned[:60]


def create_output_file(input_filename: str, suffix: str | None = None) -> Path:
    """创建 ``dialogue_<stem>_<ts>[_<suffix>].csv`` 并写入表头。

    用 CSV 而不是 xlsx 是因为：xlsx 单元格 32,767 字符上限会截断长 tool 响应，
    CSV 没有该限制；同时 ``utf-8-sig`` 让 Excel / WPS 双击打开仍能正确识别中文。

    Args:
        input_filename: 用例 xlsx 路径或文件名（取 stem 作为前缀）。
        suffix: 可选，附加到文件名末尾——通常是 agent 模型名，便于矩阵评测时
            按模型区分产物。会做 filename-safe 清洗。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(input_filename).stem
    suffix_part = ""
    if suffix:
        cleaned = _sanitize_filename_suffix(suffix)
        if cleaned:
            suffix_part = f"_{cleaned}"
    output_name = f"dialogue_{stem}_{timestamp}{suffix_part}.csv"
    output_path = OUTPUT_DIR / output_name

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(OUTPUT_HEADERS)
    logger.info(f"输出文件已创建: {output_path}")
    return output_path


class DialogueResultWriter:
    """缓冲式、并发安全的 dialogue_xxx.csv 追加写入器。

    与原 xlsx 版的区别：
    - **CSV 没有单元格长度上限**，长 tool 响应不再被截断
    - 文件持续打开 + 每次 ``writerow`` 直接写入 Python text buffer；
      ``flush_every`` 控制把 buffer 推到内核（``fh.flush``）的频率
    - 多并发 case 通过 ``asyncio.Lock`` 串行化 writerow + flush，避免 CSV 行撕裂
    - 编码统一 ``utf-8-sig``：BOM + UTF-8，Excel / WPS 双击打开能识别中文

    使用：
        async with DialogueResultWriter(path) as writer:
            await writer.append(turn)
            ...
        # 退出时自动 close → flush
    """

    def __init__(self, path: Path, flush_every: int = 10):
        self.path = Path(path)
        self.flush_every = max(1, flush_every)
        # append 模式打开（表头由 create_output_file 已经写好），保持文件句柄
        # 不靠 atexit；明确通过 close()/__aexit__ 释放
        self._fh = open(self.path, "a", newline="", encoding="utf-8-sig")
        self._writer = csv.writer(self._fh, quoting=csv.QUOTE_MINIMAL)
        self._unflushed = 0
        self._lock = asyncio.Lock()
        self._closed = False

    async def append(self, turn: DialogueTurn) -> None:
        async with self._lock:
            if self._closed:
                logger.warning(
                    "DialogueResultWriter 已关闭，丢弃 turn (cid=%s, idx=%s)",
                    turn.conversation_id, turn.turn_index,
                )
                return
            self._writer.writerow(_turn_to_row(turn))
            self._unflushed += 1
            if self._unflushed >= self.flush_every:
                await self._flush_locked()

    async def flush(self) -> None:
        async with self._lock:
            if not self._closed:
                await self._flush_locked()

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            await self._flush_locked()
            try:
                self._fh.close()
            except Exception as e:
                logger.warning(
                    "DialogueResultWriter 文件句柄关闭异常 (path=%s): %s",
                    self.path, e,
                )
            self._closed = True

    async def _flush_locked(self) -> None:
        """调用方需持有 self._lock；只 flush Python buffer 到内核，不强制 fsync。"""
        if self._unflushed == 0:
            return
        try:
            await asyncio.to_thread(self._fh.flush)
            self._unflushed = 0
        except Exception as e:
            logger.warning(
                "DialogueResultWriter flush 失败 (path=%s): %s", self.path, e
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()


def build_agent_simulator(
    sandbox: bool = False,
    streaming: bool = False,
    model: str | None = None,
    thinking: bool = False,
) -> BaseSimulationAgent:
    """构建 agent。api_key 统一走 AI_STUDIO_TOKEN（B3）。

    Args:
        model: 显式指定 agent 用的大模型；None 时 fallback 到
            ``settings.MODEL_NAME``（.env / Diamond 配置）。
            BaseSimulationAgent.__init__ 的 _STREAMING_POLICY 会按 model 名
            自动选 streaming 策略（claude → 非流式 / qwen3-30b-a3b → 流式）。
        thinking: 是否启用 thinking / reasoning 模式。默认 False。
            传 True 但 model 不在 _THINKING_POLICY 白名单 → BaseSimulationAgent
            会 raise ValueError。
    """
    resolved_model = model if model else settings.MODEL_NAME
    return BaseSimulationAgent(
        base_url=settings.BASE_URL,
        api_key=settings.AI_STUDIO_TOKEN,
        model=resolved_model,
        tools=all_tools,
        sandbox=sandbox,
        streaming=streaming,
        thinking=thinking,
    )


def _make_error_turn(case: DialogueCase, conversation_id: str, exc: Exception) -> DialogueTurn:
    """把 case 启动失败 / simulate 抛异常的情况包成一条 error turn 落盘。"""
    return DialogueTurn(
        conversation_id=conversation_id,
        turn_index=0,
        role="user",
        content=case.query or "",
        status="error",
        error_message=f"{type(exc).__name__}: {exc}",
        query=case.query or "",
        context=case.context,
        time=case.time,
        location=case.location,
        tool=case.tool,
        ground_truth=case.ground_truth,
    )


# 单条 case 执行多轮对话仿真
async def dialogue_simulator_single(
    case: DialogueCase,
    writer: DialogueResultWriter,
    is_sandbox: bool = False,
    language: str = "chinese",
    max_turns: int = 20,
    streaming: bool = False,
    model: str | None = None,
    thinking: bool = False,
) -> DialogueResult:
    """跑单条 case；任何异常都被转成 error DialogueResult，绝不向外抛（B4）。

    Args:
        model: 透传到 ``_build_agent_simulator``，覆盖 settings.MODEL_NAME。
        thinking: 透传 thinking 模式开关（默认 False）。
    """
    conversation_id = case.task_id or str(uuid.uuid4())

    try:
        agent = build_agent_simulator(
            is_sandbox, streaming=streaming, model=model, thinking=thinking,
        )
        # 公共 factory：内部用 OpenAICompatProvider，并支持从 context 解析 user_loc_name 等。
        # language 当前 factory 内部固定为 chinese，留作未来扩展。
        user = build_user_simulator(case)

        dialogue = DialogueSimulator(
            agent=agent,
            user=user,
            conversation_id=conversation_id,
            writer=writer,
            max_turns=max_turns,
        )
        result = await dialogue.simulate(case)
    except Exception as e:
        logger.exception("[%s] case 仿真失败: %s", conversation_id, e)
        err_turn = _make_error_turn(case, conversation_id, e)
        # best-effort 落一行错误记录
        try:
            await writer.append(err_turn)
        except Exception as werr:
            logger.warning("[%s] 错误 turn 写入也失败: %s", conversation_id, werr)
        result = DialogueResult(
            conversation_id=conversation_id,
            case=case,
            turns=[err_turn],
            total_turns=1,
            is_natural_stop=False,
        )
    finally:
        # 每个 case 结束 flush 一次，限定 case 间崩溃丢失上限 = 当前 case 内不到 flush_every 的尾巴
        try:
            await writer.flush()
        except Exception as ferr:
            logger.warning("[%s] writer.flush 失败: %s", conversation_id, ferr)

    return result
