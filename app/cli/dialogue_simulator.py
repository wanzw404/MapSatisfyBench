import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from app.config import settings
from app.schemas.dialogue_simulator import DialogueResult
from app.services.dialogue_recorder import (
    DialogueResultWriter,
    create_output_file,
    dialogue_simulator_single,
)
from app.services.excel_parser import read_dialogue_cases

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_CONCURRENCY = 4


async def run_dialogue_evaluation(
    filename: str,
    max_turns: int = 20,
    sandbox: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
    streaming: bool = False,
    suffix: str | None = None,
    model: str | None = None,
    thinking: bool = False,
) -> list[DialogueResult]:
    """执行多轮对话仿真评测。

    Args:
        filename: /data/inputs 下的 Excel 文件名
        max_turns: 单 case 最大对话轮次
        sandbox: 工具是否走 mock（user_simulator 仍打真实 LLM）
        concurrency: 同时跑多少条 case；DialogueResultWriter 内部 asyncio.Lock
            保证并发写 xlsx 不会撕裂
        streaming: agent LLM 是否走流式调用；启用后能拿到真实首 chunk 时间作为 TTFT
        suffix: 输出 CSV 文件名后缀；None 时 fallback 到 model 名（再 fallback
            settings.MODEL_NAME）
        model: 显式指定被测 agent 用的大模型；None 时 fallback 到
            settings.MODEL_NAME。**只影响 agent，不影响 user_simulator / judge**
        thinking: 是否启用 agent LLM 的 thinking / reasoning 模式。默认 False。
            传 True 但 model 不在 ``BaseSimulationAgent._THINKING_POLICY``
            白名单 → 启动期 raise ValueError 让用户立即可见。
    """
    # B3: 启动期校验 AI_STUDIO_TOKEN（agent 与 user 现在都依赖它）
    if not settings.AI_STUDIO_TOKEN:
        logger.error(
            "AI_STUDIO_TOKEN 未配置：dialogue_simulator 与 user_simulator 都依赖 "
            "AI_STUDIO_TOKEN 环境变量，请在 .env 或环境变量中设置后重试。"
        )
        sys.exit(1)

    try:
        cases = read_dialogue_cases(filename)
    except FileNotFoundError as e:
        logger.error(f"输入文件不存在: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"输入文件解析失败: {e}")
        sys.exit(1)

    if not cases:
        logger.error("Excel 中未找到有效用例")
        sys.exit(1)

    mode_str = "沙箱模式" if sandbox else "真实 API 模式"
    stream_str = "流式" if streaming else "非流式"
    logger.info(
        "读取到 %d 条用例，%s + %s，并发=%d，开始多轮对话仿真...",
        len(cases), mode_str, stream_str, concurrency,
    )

    # 决议 agent 模型：显式 --model 优先，否则 fallback settings.MODEL_NAME
    effective_model = model if model else settings.MODEL_NAME
    logger.info("Agent 模型：%s（来源：%s）",
                effective_model,
                "--model 入参" if model else "settings.MODEL_NAME (.env)")

    # 决议 suffix：显式 --suffix 优先；否则用 effective_model（自然带上 agent 模型）
    effective_suffix = suffix if suffix is not None else effective_model
    output_path = create_output_file(filename, suffix=effective_suffix)
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _run_one(idx: int, case, writer: DialogueResultWriter) -> DialogueResult:
        # B4: dialogue_simulator_single 内部已 try/except 兜住所有异常，
        # 这里再挂 logger 方便从 batch 视角看进度
        async with sem:
            preview = case.query[:50] + ("..." if len(case.query) > 50 else "")
            logger.info("[%d/%d] start | query=%s", idx + 1, len(cases), preview)
            result = await dialogue_simulator_single(
                case,
                writer,
                is_sandbox=sandbox,
                language="chinese",
                max_turns=max_turns,
                streaming=streaming,
                model=effective_model,
                thinking=thinking,
            )
            stop_type = "自然终止" if result.is_natural_stop else "强制截断/异常"
            logger.info(
                "[%d/%d] done | conv_id=%s | turns=%d | %s",
                idx + 1, len(cases),
                result.conversation_id, result.total_turns, stop_type,
            )
            return result

    # B1+B5+P2: 共享 writer + 并发 gather + lock 保证写入安全
    async with DialogueResultWriter(output_path) as writer:
        results = await asyncio.gather(
            *(_run_one(i, c, writer) for i, c in enumerate(cases)),
        )

    # 简要回放每个 case 的 turn 列表，便于查看
    for idx, result in enumerate(results, 1):
        logger.info(
            "===== case %d/%d (conv_id=%s) =====",
            idx, len(results), result.conversation_id,
        )
        for turn in result.turns:
            stop_marker = " [STOP]" if turn.is_stop else ""
            content_preview = (turn.content or "")[:60]
            logger.info(
                "  Turn %s %s: %s%s",
                turn.turn_index, turn.role, content_preview, stop_marker,
            )

    logger.info("全部评测完成，结果已保存: %s", output_path)
    return results


def cmd_run(args):
    asyncio.run(
        run_dialogue_evaluation(
            filename=args.filename,
            max_turns=args.max_turns,
            sandbox=args.sandbox,
            concurrency=args.concurrency,
            streaming=args.streaming,
            suffix=args.suffix,
            model=args.model,
            thinking=args.thinking,
        )
    )


def main():
    parser = argparse.ArgumentParser(description="多轮对话仿真评测 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="执行多轮对话仿真评测")
    run_parser.add_argument("filename", help="/data/inputs 下的 Excel 文件名")
    run_parser.add_argument(
        "--max-turns", type=int, default=20, help="最大对话轮次上限（默认 20）"
    )
    run_parser.add_argument(
        "--sandbox",
        action="store_true",
        default=False,
        help="启用沙箱模式（工具调用走 mock 数据；user_simulator 仍打真实 LLM）",
    )
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"batch 并发上限（默认 {DEFAULT_CONCURRENCY}，设为 1 退化为串行）",
    )
    run_parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help=(
            "启用 agent LLM 流式调用（astream 累加 chunks）；"
            "可拿到真实首 chunk 时间作为 TTFT，LangGraph 上层语义不变。默认关闭。"
        ),
    )
    run_parser.add_argument(
        "--suffix",
        default=None,
        help=(
            "输出 CSV 文件名的后缀（在 timestamp 之后）。通常传 agent 模型名，"
            "用于矩阵评测时按模型区分产物。不传则自动用 --model 值，再 fallback "
            "到 settings.MODEL_NAME。"
        ),
    )
    run_parser.add_argument(
        "--model",
        default=None,
        help=(
            "显式指定被测 agent 用的大模型；不传则 fallback 到 settings.MODEL_NAME "
            "（来自 .env / Diamond）。**只影响 agent**，user_simulator 与 judge "
            "仍走各自锁定的 USER_SIMULATOR_MODEL / JUDGE_MODEL 常量。"
        ),
    )
    run_parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help=(
            "启用 agent LLM 的 thinking / reasoning 模式。默认关闭。"
            "**不传本 flag 时，所有 agent 仿真模型都会被强制关闭 thinking**"
            "（DashScope 系如 qwen3 / deepseek-v4-pro / deepseek-v3.2 会显式发 "
            "enable_thinking=False；gemini 走 vertex 协议显式发 thinkingBudget=0；"
            "其它模型默认即关）。支持开 thinking 的 model 白名单（substring）："
            "qwen3.6-plus / qwen3 / qwen-plus / claude / deepseek-v4-pro / "
            "deepseek-v3.2 / gemini。**传 --thinking 但 model 不在白名单或本就"
            "不支持 thinking（gpt-4.1/gpt-5.x-chat）→ 启动期 raise ValueError**。"
            "仅作用于 agent，user_simulator / judge 不受影响。"
        ),
    )
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
