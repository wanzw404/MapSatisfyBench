"""CLI 入口：把评测结果 xlsx 跑一次批次统计 + 可视化。

业务逻辑全部在 ``app.services.eval_summary_service.summarize``；本文件只做
argparse 与调用，方便后续 HTTP 路由直接 import 同一个 ``summarize`` 复用。

Usage:
    .venv/bin/python -m app.scripts.aggregate_eval_results \
        --input data/outputs/evaluation_res/evaluation_result_xxx.xlsx \
        --output-dir data/outputs/evaluation_res \
        --top-n 5 \
        --visualize both
"""

import argparse
import logging
import sys
from pathlib import Path

# 项目根加入 sys.path 便于 python -m
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.eval_summary_service import VisualizeMode, summarize  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="读取评测结果 xlsx，做批次级 6 指标统计 + 可视化"
    )
    parser.add_argument(
        "--input", required=True,
        help="评测结果 xlsx 路径（来自 batch_evaluate_from_simulator）",
    )
    parser.add_argument(
        "--output-dir", default="data/outputs/report",
        help=(
            "产物落盘根目录（默认 data/outputs/report）；"
            "实际产物会落在 <output-dir>/single/<时间戳>/ 下"
        ),
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="每个指标列出的 top-N 高/低 case_id，默认 5",
    )
    parser.add_argument(
        "--visualize", default="both", choices=["xlsx", "png", "both", "none"],
        help="可视化产物：xlsx (内嵌图表) / png (matplotlib + html) / both / none",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)

    products = summarize(
        xlsx_path=input_path,
        output_dir=output_dir,
        top_n=args.top_n,
        visualize=args.visualize,  # type: ignore[arg-type]  # Literal 校验 by argparse
    )

    logger.info(f"  JSON  → {products.json_path}")
    logger.info(f"  TXT   → {products.txt_path}")
    if products.xlsx_path:
        logger.info(f"  XLSX  → {products.xlsx_path}")
    if products.charts_dir:
        logger.info(f"  PNG   → {products.charts_dir}/")
    if products.html_path:
        logger.info(f"  HTML  → {products.html_path}")


if __name__ == "__main__":
    main()
