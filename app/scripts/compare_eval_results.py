"""CLI 入口：对比多个 evaluation_result（csv / xlsx）的批次统计。

支持两种模式：

1) **按 model 分组（推荐，仅均值双层报告）**——传 ``--group``（可重复）：

       --group claude-opus-4-6 path/to/eval_a.csv path/to/eval_b.csv
       --group gpt-5.3         path/to/eval_c.csv

   每个 model 一个组，组内多份 csv 是多次评测。报告：
     ① 单模型多轮均值（顶部，跨该 model 所有 cases 的算术平均）
     ② 单模型每轮明细（每 csv 一行，per-run 平均）
   只输出 markdown + json，不再生成 xlsx 内嵌图表 + matplotlib png。

2) **不分组（兼容老用法，每文件独立运行）**——传 ``--inputs``：
   走 ``compare()`` 老路径，输出包含 median / std / percentile 等完整 stats，
   xlsx + png 可视化保留。

Examples:
    # 分组对比
    .venv/bin/python -m app.scripts.compare_eval_results \\
        --group claude  data/outputs/evaluation_res/eval_a.csv \\
                        data/outputs/evaluation_res/eval_b.csv \\
        --group qwen3   data/outputs/evaluation_res/eval_c.csv \\
        --output-dir data/outputs/report

    # 老用法（每文件独立、完整 stats）
    .venv/bin/python -m app.scripts.compare_eval_results \\
        --inputs eval_v1.csv eval_v2.csv \\
        --output-dir data/outputs/report \\
        --visualize both
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.eval_compare_service import compare, compare_grouped  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "对比多个 evaluation_result（csv/xlsx）。"
            "传 --group 走「按 model 分组、仅均值」双层报告；"
            "传 --inputs 走老用法（完整 stats + 图表）。"
        )
    )
    # 分组模式
    parser.add_argument(
        "--group", action="append", nargs="+",
        metavar="MODEL CSV",
        help=(
            "按 model 分组：--group <model> <csv1> [<csv2> ...]，可重复。"
            "同一 model 下多个 csv = 多次评测。"
        ),
    )
    # 兼容老用法
    parser.add_argument(
        "--inputs", nargs="+",
        help=(
            "老用法（不分组）：N 个 evaluation_result csv/xlsx 路径，"
            "至少 2 个。与 --group 互斥。"
        ),
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="老用法专用：对应每个 --inputs 文件的展示名称；缺省用文件 stem",
    )
    parser.add_argument(
        "--output-dir", default="data/outputs/report",
        help=(
            "产物落盘根目录（默认 data/outputs/report）；"
            "实际产物会落在 <output-dir>/compare/<时间戳>/ 下"
        ),
    )
    parser.add_argument(
        "--visualize", default="both", choices=["xlsx", "png", "both", "none"],
        help="老用法专用可视化：xlsx (内嵌图表) / png / both / none",
    )
    args = parser.parse_args()

    if args.group and args.inputs:
        logger.error("--group 与 --inputs 互斥；选一个用法")
        sys.exit(2)

    if not args.group and not args.inputs:
        logger.error("必须传 --group 或 --inputs 之一")
        sys.exit(2)

    # ── 分组模式 ──────────────────────────────────────────────────────
    if args.group:
        groups: dict[str, list[Path]] = {}
        for entry in args.group:
            if len(entry) < 2:
                logger.error(
                    "--group 需要至少 1 个 model 名 + 1 个 csv 路径，但收到: %s",
                    entry,
                )
                sys.exit(2)
            model = entry[0]
            paths = [Path(p) for p in entry[1:]]
            missing = [p for p in paths if not p.exists()]
            if missing:
                for p in missing:
                    logger.error("[%s] 输入文件不存在: %s", model, p)
                sys.exit(1)
            groups.setdefault(model, []).extend(paths)

        products = compare_grouped(
            groups=groups,
            output_dir=Path(args.output_dir),
        )
        logger.info("  落盘根目录: %s", products.out_dir)
        logger.info("  MD    → %s", products.md_path)
        logger.info("  JSON  → %s", products.json_path)
        return

    # ── 老用法 ────────────────────────────────────────────────────────
    paths = [Path(p) for p in args.inputs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            logger.error(f"输入文件不存在: {p}")
        sys.exit(1)
    if len(paths) < 2:
        logger.error("至少需要 2 个文件；单文件请用 aggregate_eval_results")
        sys.exit(1)

    if args.labels and len(args.labels) != len(paths):
        logger.error(
            f"--labels 数量 ({len(args.labels)}) 与 --inputs ({len(paths)}) 不一致"
        )
        sys.exit(1)

    products = compare(
        xlsx_paths=paths,
        output_dir=Path(args.output_dir),
        labels=args.labels,
        visualize=args.visualize,  # type: ignore[arg-type]
    )

    logger.info(f"  落盘根目录: {products.out_dir}")
    logger.info(f"  MD    → {products.md_path}")
    logger.info(f"  JSON  → {products.json_path}")
    if products.xlsx_path:
        logger.info(f"  XLSX  → {products.xlsx_path}")
    if products.charts_dir:
        logger.info(f"  PNG   → {products.charts_dir}/")


if __name__ == "__main__":
    main()
