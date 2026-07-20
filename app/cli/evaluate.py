import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import settings
from app.core.simulator.agent_simulator import BaseSimulationAgent
from app.core.tools.manager import tools
from app.services.batch_runner import BatchRunner
from app.services.excel_parser import read_cases, write_results, INPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_agent() -> BaseSimulationAgent:
    return BaseSimulationAgent(
        base_url=settings.BASE_URL,
        api_key=settings.AI_STUDIO_TOKEN,
        model=settings.MODEL_NAME,
        tools=tools,
    )


def cmd_upload(args):
    """上传本地 Excel 到 /data/inputs。"""
    import shutil
    import uuid

    src = Path(args.file)
    if not src.exists():
        logger.error(f"文件不存在: {src}")
        sys.exit(1)

    if not src.suffix.lower() == ".xlsx":
        logger.error("仅支持 .xlsx 格式")
        sys.exit(1)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_name = f"{uuid.uuid4().hex[:8]}_{src.name}"
    dst = INPUT_DIR / saved_name
    shutil.copy2(src, dst)
    logger.info(f"已上传: {dst}")

    if args.run:
        args.filename = saved_name
        args.output_dir = None
        asyncio.run(cmd_run_async(args))


async def cmd_run_async(args):
    """异步执行评测。"""
    cases = read_cases(args.filename)
    if not cases:
        logger.error("Excel 中未找到有效用例")
        sys.exit(1)

    logger.info(f"读取到 {len(cases)} 条用例，开始执行...")

    agent = _build_agent()
    runner = BatchRunner(agent)
    results = await runner.run_batch(
        cases,
        thread_id_prefix=args.thread_id_prefix,
    )

    output_path = write_results(args.filename, results)
    logger.info(f"结果已保存: {output_path}")

    success = sum(1 for r in results if r.status == "success")
    logger.info(f"执行完成: 成功 {success}/{len(results)}")


def cmd_run(args):
    asyncio.run(cmd_run_async(args))


def cmd_list(args):
    """列出 /data/inputs 下所有 .xlsx 文件。"""
    files = sorted(INPUT_DIR.glob("*.xlsx"))
    if not files:
        logger.info("/data/inputs 下暂无 .xlsx 文件")
        return

    logger.info(f"/data/inputs 下共有 {len(files)} 个文件:")
    for f in files:
        logger.info(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(description="批量评测 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # upload
    upload_parser = sub.add_parser("upload", help="上传 Excel 文件")
    upload_parser.add_argument("file", help="本地 Excel 文件路径")
    upload_parser.add_argument("--run", action="store_true", help="上传后立即执行")
    upload_parser.set_defaults(func=cmd_upload)

    # run
    run_parser = sub.add_parser("run", help="执行指定 Excel 评测")
    run_parser.add_argument("filename", help="/data/inputs 下的文件名")
    run_parser.add_argument(
        "--thread-id-prefix", default="eval", help="thread_id 前缀"
    )
    run_parser.set_defaults(func=cmd_run)

    # list
    list_parser = sub.add_parser("list", help="列出可执行的输入文件")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
