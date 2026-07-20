#!/usr/bin/env python3
"""跨 agent 模型的矩阵评测编排脚本。

为每个 ``--models`` 列出的模型，串行执行：
  1) 多轮对话仿真（``app.cli.dialogue_simulator run``）
  2) 评分（``app.scripts.batch_evaluate_from_simulator``）

子进程方式 fork——因为 ``app.config.settings`` 在模块 import 时一次性读取
``MODEL_NAME``，进程中途改 ``os.environ`` 不会重新生效。每个 model 一个独立
子进程，env 里覆盖 ``MODEL_NAME``，保证 ``BaseSimulationAgent`` 用对模型。

UserSimulator 与 Judge 不受影响——它们走 ``app.config.USER_SIMULATOR_MODEL``
/ ``JUDGE_MODEL`` 模块级常量（当前都锁定 ``gpt-5.3-chat-0303-global``）。

输出目录保持现状：仿真 → ``data/outputs/simulator_res``，评分 →
``data/outputs/evaluation_res``。本脚本另在 ``--output-dir`` 写汇总
``matrix_summary.json`` + ``matrix_summary.md``，记录各 model 产出文件路径与
耗时。

Usage:
    python scripts/run_eval_matrix.py \\
        --input parsed_demo_0518_01.xlsx \\
        --models claude-opus-4-6 gpt-5.3-chat-0303-global Qwen3-30B-A3B \\
        --concurrency 2

可选：``--sandbox`` / ``--streaming`` / ``--max-turns`` / ``--skip-eval``。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "inputs"
SIM_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "simulator_res"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "evaluation_res"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_eval_matrix")


# ─────────────────────────────────────────────────────────────────────
# 子进程执行
# ─────────────────────────────────────────────────────────────────────


def _run_subprocess(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    label: str,
) -> tuple[int, float]:
    """跑子进程，把 stdout+stderr 写入 log_path；返回 (returncode, elapsed_s)。

    实时把子进程输出 tee 到本进程 stdout（前缀加 label）+ 落盘 log，方便边跑边看。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    logger.info("[%s] $ %s", label, " ".join(cmd))

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# CMD: {' '.join(cmd)}\n")
        logf.write(f"# ENV.MODEL_NAME = {env.get('MODEL_NAME', '<unset>')}\n")
        logf.write(f"# started_at = {datetime.now().isoformat()}\n\n")
        logf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(f"  [{label}] {line}")
            sys.stdout.flush()
            logf.write(line)
        rc = proc.wait()
        elapsed = time.time() - started_at
        logf.write(f"\n# returncode = {rc}\n# elapsed_s = {elapsed:.1f}\n")
    return rc, elapsed


def _newest_file_after(
    directory: Path, pattern: str, since_ts: float
) -> Path | None:
    """返回 ``directory`` 下匹配 pattern 且 mtime > since_ts 的最新文件。"""
    if not directory.exists():
        return None
    candidates = [
        p for p in directory.glob(pattern)
        if p.is_file() and p.stat().st_mtime > since_ts
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ─────────────────────────────────────────────────────────────────────
# 单 model 一轮：仿真 + 评分
# ─────────────────────────────────────────────────────────────────────


def run_one_model(
    *,
    model: str,
    input_filename: str,
    output_dir: Path,
    max_turns: int,
    concurrency: int,
    sandbox: bool,
    streaming: bool,
    skip_eval: bool,
) -> dict:
    """跑单个 model 的 (仿真, 评分)；返回结构化结果记录。

    永远 best-effort 跑完——任一步骤失败仅记录到 record，不抛异常。
    """
    log_dir = output_dir / "logs"
    safe_model = model.replace("/", "_")
    record: dict = {
        "model": model,
        "status": "pending",
        "sim_csv": None,
        "sim_log": None,
        "sim_elapsed_s": None,
        "sim_returncode": None,
        "eval_xlsx": None,
        "eval_log": None,
        "eval_elapsed_s": None,
        "eval_returncode": None,
        "error": None,
    }

    # 子进程环境：继承当前 env + 覆盖 MODEL_NAME
    env = os.environ.copy()
    env["MODEL_NAME"] = model

    # ── 1. 仿真 ────────────────────────────────────────────────────
    sim_log = log_dir / f"{safe_model}.simulate.log"
    record["sim_log"] = str(sim_log)
    sim_started_ts = time.time()
    sim_cmd = [
        sys.executable, "-m", "app.cli.dialogue_simulator", "run", input_filename,
        "--max-turns", str(max_turns),
        "--concurrency", str(concurrency),
        "--suffix", model,
    ]
    if sandbox:
        sim_cmd.append("--sandbox")
    if streaming:
        sim_cmd.append("--streaming")

    try:
        rc, elapsed = _run_subprocess(sim_cmd, env, sim_log, f"{model} sim")
        record["sim_returncode"] = rc
        record["sim_elapsed_s"] = round(elapsed, 1)
    except Exception as e:
        record["status"] = "sim_crashed"
        record["error"] = f"sim subprocess failed: {type(e).__name__}: {e}"
        logger.error("[%s] 仿真子进程崩溃: %s", model, e)
        return record

    # 找仿真输出 csv（mtime 在 sim_started_ts 之后）
    sim_csv = _newest_file_after(SIM_OUTPUT_DIR, "dialogue_*.csv", sim_started_ts)
    if sim_csv is None:
        record["status"] = "sim_no_output"
        record["error"] = (
            f"未在 {SIM_OUTPUT_DIR} 找到 mtime > {sim_started_ts} 的 dialogue_*.csv"
        )
        logger.error("[%s] %s", model, record["error"])
        return record
    record["sim_csv"] = str(sim_csv)
    logger.info("[%s] 仿真产出: %s", model, sim_csv.name)

    if rc != 0:
        # 子进程非 0：标记但仍尝试评分（CSV 已经写出来了，部分数据值得评）
        record["status"] = "sim_partial"
        logger.warning("[%s] 仿真 returncode=%d，但已生成 CSV，继续评分", model, rc)

    if skip_eval:
        if record["status"] == "pending":
            record["status"] = "sim_only"
        return record

    # ── 2. 评分 ────────────────────────────────────────────────────
    eval_log = log_dir / f"{safe_model}.evaluate.log"
    record["eval_log"] = str(eval_log)
    eval_started_ts = time.time()
    eval_cmd = [
        sys.executable, "-m", "app.scripts.batch_evaluate_from_simulator",
        "--input", str(sim_csv),
        "--cases-input", str(INPUT_DIR / input_filename),
        "--output", str(EVAL_OUTPUT_DIR),
        "--max-concurrency", str(concurrency),
        "--suffix", model,
    ]
    try:
        rc2, elapsed2 = _run_subprocess(eval_cmd, env, eval_log, f"{model} eval")
        record["eval_returncode"] = rc2
        record["eval_elapsed_s"] = round(elapsed2, 1)
    except Exception as e:
        record["status"] = "eval_crashed"
        record["error"] = f"eval subprocess failed: {type(e).__name__}: {e}"
        logger.error("[%s] 评分子进程崩溃: %s", model, e)
        return record

    # 评分新版产出 .csv（EvaluationResultWriter 并发即写）；老 xlsx 也兜底匹配，
    # 兼容尚未升级的部署或人工放进来的历史产物
    eval_xlsx = _newest_file_after(
        EVAL_OUTPUT_DIR, "evaluation_result_*.csv", eval_started_ts
    ) or _newest_file_after(
        EVAL_OUTPUT_DIR, "evaluation_result_*.xlsx", eval_started_ts
    )
    if eval_xlsx is None:
        record["status"] = "eval_no_output"
        record["error"] = (
            f"未在 {EVAL_OUTPUT_DIR} 找到 mtime > {eval_started_ts} 的 evaluation_result_*.csv|.xlsx"
        )
        logger.error("[%s] %s", model, record["error"])
        return record
    record["eval_xlsx"] = str(eval_xlsx)
    logger.info("[%s] 评分产出: %s", model, eval_xlsx.name)

    if rc2 != 0:
        record["status"] = "eval_partial"
    else:
        # 仿真 + 评分都成功
        if record["status"] == "pending":
            record["status"] = "ok"
    return record


# ─────────────────────────────────────────────────────────────────────
# 汇总写出
# ─────────────────────────────────────────────────────────────────────


def write_summary(records: list[dict], output_dir: Path, run_ts: str) -> None:
    """写 matrix_summary.json + matrix_summary.md。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "matrix_summary.json"
    summary_md = output_dir / "matrix_summary.md"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {"run_ts": run_ts, "records": records}, f,
            ensure_ascii=False, indent=2,
        )

    lines = [
        f"# Eval Matrix Run {run_ts}",
        "",
        "| Model | Status | Sim time | Eval time | Sim CSV | Eval XLSX | Note |",
        "|-------|--------|---------:|----------:|---------|-----------|------|",
    ]
    for r in records:
        sim_t = f"{r['sim_elapsed_s']:.1f}s" if r["sim_elapsed_s"] is not None else "-"
        eval_t = f"{r['eval_elapsed_s']:.1f}s" if r["eval_elapsed_s"] is not None else "-"
        sim_csv = Path(r["sim_csv"]).name if r["sim_csv"] else "-"
        eval_xlsx = Path(r["eval_xlsx"]).name if r["eval_xlsx"] else "-"
        note = r.get("error") or ""
        # Markdown 表格里管道符要转义
        note = note.replace("|", "\\|").replace("\n", " ")[:200]
        lines.append(
            f"| `{r['model']}` | {r['status']} | {sim_t} | {eval_t} | "
            f"{sim_csv} | {eval_xlsx} | {note} |"
        )

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("矩阵汇总已写入:")
    logger.info("  - %s", summary_json)
    logger.info("  - %s", summary_md)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="跨 agent 模型矩阵评测编排（仿真 + 评分）"
    )
    parser.add_argument(
        "--input", required=True,
        help="data/inputs 下的 xlsx 文件名（如 parsed_demo_0518_01.xlsx）",
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="一个或多个 agent 模型名，每个独立子进程跑（覆盖 MODEL_NAME 环境变量）",
    )
    parser.add_argument(
        "--output-dir",
        default="data/outputs/run_matrix",
        help="矩阵汇总目录（仅写日志与 summary，仿真/评分产出仍在原目录）",
    )
    parser.add_argument(
        "--max-turns", type=int, default=20,
        help="单 case 最大轮次（透传仿真 CLI；默认 20）",
    )
    parser.add_argument(
        "--concurrency", type=int, default=2,
        help="单 model 内 case 并发（透传仿真 + 评分；默认 2）",
    )
    parser.add_argument(
        "--sandbox", action="store_true",
        help="仿真走 mock 数据（user_simulator 仍打真实 LLM）",
    )
    parser.add_argument(
        "--streaming", action="store_true",
        help="agent LLM 流式调用（注意：claude-* 有 tool_call 拼接 bug，慎开）",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="只跑仿真不评分（用于先生成数据后批量评分）",
    )
    args = parser.parse_args()

    # 入参校验
    input_path = INPUT_DIR / args.input
    if not input_path.exists():
        logger.error("输入文件不存在: %s", input_path)
        sys.exit(1)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_ts
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Eval Matrix Run @ %s", run_ts)
    logger.info("Input         : %s", input_path)
    logger.info("Models        : %s", args.models)
    logger.info("Concurrency   : %d", args.concurrency)
    logger.info("Max turns     : %d", args.max_turns)
    logger.info("Sandbox       : %s", args.sandbox)
    logger.info("Streaming     : %s", args.streaming)
    logger.info("Skip eval     : %s", args.skip_eval)
    logger.info("Output dir    : %s", output_dir)
    logger.info("=" * 70)

    records: list[dict] = []
    for i, model in enumerate(args.models, 1):
        logger.info("")
        logger.info("##### [%d/%d] model = %s #####", i, len(args.models), model)
        rec = run_one_model(
            model=model,
            input_filename=args.input,
            output_dir=output_dir,
            max_turns=args.max_turns,
            concurrency=args.concurrency,
            sandbox=args.sandbox,
            streaming=args.streaming,
            skip_eval=args.skip_eval,
        )
        records.append(rec)
        logger.info("[%s] 完成: status=%s", model, rec["status"])
        # 即时刷写 summary，防止中途崩溃丢失中间结果
        write_summary(records, output_dir, run_ts)

    # 退出码：任一非 ok/sim_only 状态 → 1
    ok_states = {"ok", "sim_only"}
    has_failure = any(r["status"] not in ok_states for r in records)
    logger.info("")
    logger.info("=" * 70)
    logger.info("Done. %d / %d models OK.",
                sum(1 for r in records if r["status"] in ok_states), len(records))
    sys.exit(1 if has_failure else 0)


if __name__ == "__main__":
    main()
