#!/usr/bin/env python3
"""批量对 simulator_res/ 下所有仿真 csv 跑评分任务（每个 csv 默认跑 2 次）。

设计：
  * 不改评分模块——直接 subprocess 调既有 ``app.scripts.batch_evaluate_from_simulator``，
    透传它的参数（``--suffix`` 已支持把模型名拼到产物文件名）。
  * 串行执行：一个仿真 csv 跑完所有 round 再下一个。LLM QPS 限流由
    ``RateLimitedRetryLLMProvider`` 全局共享，串行也能用满限额。
  * 文件名约定：simulator csv 命名形如
    ``dialogue_<input_stem>_<YYYYMMDD>_<HHMMSS>_<model>.csv``，
    本脚本按 ``stem.split('_')[-1]`` 取最后一段作为模型名（与用户约定一致）。
  * 失败时 log 跳到下一项；全部跑完后打印汇总（成功 / 失败列表 / 跳过项）。

示例::

    .venv/bin/python -m app.scripts.batch_eval_all_simulator_runs \\
        --cases-input data/inputs/.xlsx \\
        --runs 2 \\
        --max-concurrency 16 \\
        --llm-qps 100

输出文件命名（每个 csv × 每个 round 各一份）::

    evaluation_result_<ts>_<model>_run<N>.csv
    aggregated_<ts>_<model>_run<N>.json
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SIM_DIR = _PROJECT_ROOT / "data" / "outputs" / "simulator_res" / "实验2-2"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "outputs" / "evaluation_res"


def _model_name_from_csv(csv_path: Path) -> str:
    """按文件名末段约定提取模型名。

    约定：``dialogue_<input_stem>_<YYYYMMDD>_<HHMMSS>_<model>.csv``
    取 ``stem.split('_')[-1]``。模型名内允许含 ``.`` ``-``（如
    ``deepseek-v3.2``、``gemini-3.1-pro-preview``），都被保留。
    """
    return csv_path.stem.rsplit("_", 1)[-1]


def _run_one(
    csv_path: Path,
    cases_input: Path,
    output_dir: Path,
    suffix: str,
    *,
    max_concurrency: int,
    llm_qps: float,
    language: str,
    no_verify: bool,
    enable_meta_judge: bool,
    model_override: str,
) -> int:
    """单次评分任务，返回 subprocess 退出码。"""
    cmd = [
        sys.executable,
        "-m",
        "app.scripts.batch_evaluate_from_simulator",
        "--input", str(csv_path),
        "--cases-input", str(cases_input),
        "--output", str(output_dir),
        "--max-concurrency", str(max_concurrency),
        "--llm-qps", str(llm_qps),
        "--language", language,
        "--suffix", suffix,
    ]
    if no_verify:
        cmd.append("--no-verify")
    if enable_meta_judge:
        cmd.append("--enable-meta-judge")
    if model_override:
        cmd.extend(["--model", model_override])

    logger.info("→ subprocess: %s", " ".join(cmd))
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    except KeyboardInterrupt:
        logger.warning("收到 KeyboardInterrupt，中断当前 subprocess")
        raise
    elapsed = int(time.monotonic() - t0)
    logger.info("← 退出码=%d 耗时=%ds", proc.returncode, elapsed)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulator-dir",
        default=str(_DEFAULT_SIM_DIR),
        help=f"扫描的仿真结果目录（默认 {_DEFAULT_SIM_DIR}）",
    )
    parser.add_argument(
        "--cases-input",
        required=True,
        help=(
            "评分用例 xlsx 路径（含 task_id + full_intent + ground_truth）。"
            "所有仿真 csv 共用同一个 cases-input；如不同批次 cases 不同，"
            "请分多次运行本脚本。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help=f"评分结果输出目录（默认 {_DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="每个仿真 csv 跑几次（默认 2，suffix 区分 _run1 / _run2 ...）",
    )
    parser.add_argument(
        "--pattern",
        default="dialogue_*.csv",
        help='glob 过滤模式，默认 "dialogue_*.csv"',
    )
    # 透传给 batch_evaluate_from_simulator
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=16,
        help="单次评分任务的 case 并发上限（默认 16）",
    )
    parser.add_argument(
        "--llm-qps",
        type=float,
        default=100.0,
        help="评分器全局 LLM QPS 上限（默认 100；<=0 关闭限流）",
    )
    parser.add_argument(
        "--language",
        default="chinese",
        choices=["chinese", "english"],
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="关闭 web 事实校验（更快）",
    )
    parser.add_argument(
        "--enable-meta-judge",
        action="store_true",
        help="开启 Meta-Judge 复核（默认关闭）",
    )
    parser.add_argument(
        "--model",
        default="",
        help="评分用 LLM 模型覆盖（一般不传，走 JUDGE_MODEL 锁定值）",
    )
    args = parser.parse_args()

    sim_dir = Path(args.simulator_dir)
    cases_input = Path(args.cases_input)
    output_dir = Path(args.output_dir)

    if not sim_dir.is_dir():
        logger.error("simulator-dir 不存在: %s", sim_dir)
        return 1
    if not cases_input.exists():
        logger.error("cases-input 不存在: %s", cases_input)
        return 2

    csvs = sorted(sim_dir.glob(args.pattern))
    if not csvs:
        logger.error("simulator-dir=%s 无匹配 %r 的 csv", sim_dir, args.pattern)
        return 3

    logger.info(
        "扫描到 %d 个仿真 csv，每个跑 %d 次，共 %d 次评分任务",
        len(csvs), args.runs, len(csvs) * args.runs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    succeeded: list[str] = []
    failed: list[tuple[str, str, int]] = []  # (csv_name, suffix, returncode)
    total_runs = len(csvs) * args.runs
    idx = 0

    overall_t0 = time.monotonic()
    for csv_path in csvs:
        model = _model_name_from_csv(csv_path)
        for r in range(1, args.runs + 1):
            idx += 1
            suffix = f"{model}_run{r}"
            logger.info(
                "===== [%d/%d] csv=%s round=%d/%d suffix=%s =====",
                idx, total_runs, csv_path.name, r, args.runs, suffix,
            )
            try:
                rc = _run_one(
                    csv_path,
                    cases_input,
                    output_dir,
                    suffix,
                    max_concurrency=args.max_concurrency,
                    llm_qps=args.llm_qps,
                    language=args.language,
                    no_verify=args.no_verify,
                    enable_meta_judge=args.enable_meta_judge,
                    model_override=args.model,
                )
            except KeyboardInterrupt:
                logger.warning("用户中断，已完成 %d/%d 项", idx - 1, total_runs)
                _print_summary(succeeded, failed, time.monotonic() - overall_t0)
                return 130
            except Exception as exc:  # noqa: BLE001
                logger.exception("subprocess 抛异常 csv=%s suffix=%s: %s",
                                 csv_path.name, suffix, exc)
                failed.append((csv_path.name, suffix, -1))
                continue
            if rc == 0:
                succeeded.append(f"{csv_path.name} :: {suffix}")
            else:
                failed.append((csv_path.name, suffix, rc))
                logger.warning(
                    "评分失败 csv=%s suffix=%s 退出码=%d；继续下一项",
                    csv_path.name, suffix, rc,
                )

    _print_summary(succeeded, failed, time.monotonic() - overall_t0)
    return 0 if not failed else 4


def _print_summary(
    succeeded: list[str],
    failed: list[tuple[str, str, int]],
    elapsed: float,
) -> None:
    logger.info("=" * 72)
    logger.info(
        "汇总 | 成功=%d | 失败=%d | 总耗时=%ds",
        len(succeeded), len(failed), int(elapsed),
    )
    if failed:
        logger.info("失败列表：")
        for csv_name, suffix, rc in failed:
            logger.info("  - csv=%s suffix=%s exit_code=%d", csv_name, suffix, rc)


if __name__ == "__main__":
    raise SystemExit(main())
