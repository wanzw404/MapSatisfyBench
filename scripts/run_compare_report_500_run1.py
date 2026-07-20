"""Generate one cross-model comparison report for csvs in
`合并500条_run1/` using app.services.eval_compare_service.compare_grouped.

Per-csv model name = filename's last `_`-separated segment (sans .csv),
stripping a trailing `_runN` suffix if present. Each model becomes one
group with exactly one csv (the merged 500-row run1).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path("<项目地址>")
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.eval_compare_service import compare_grouped  # noqa: E402

SRC_DIR = PROJECT_ROOT / "data" / "outputs" / "evaluation_res" / "report_waiting"
OUT_DIR = PROJECT_ROOT / "data" / "outputs" / "report"


def model_of(fn: str) -> str:
    stem = fn[:-4] if fn.endswith(".csv") else fn
    parts = stem.split("_")
    return parts[-2] if parts[-1].startswith("run") else parts[-1]


def main() -> None:
    if not SRC_DIR.is_dir():
        sys.exit(f"src dir not found: {SRC_DIR}")
    files = sorted(SRC_DIR.glob("*.csv"))
    if not files:
        sys.exit(f"no csv in {SRC_DIR}")

    groups: dict[str, list[Path]] = {}
    for fp in files:
        m = model_of(fp.name)
        groups.setdefault(m, []).append(fp)

    print(f"models found: {len(groups)}")
    for m, paths in groups.items():
        print(f"  [{m}] {len(paths)} file(s): {[p.name for p in paths]}")
    print()

    products = compare_grouped(groups, OUT_DIR)

    print(f"\n=== products ===")
    print(f"  out_dir: {products.out_dir}")
    print(f"  md:      {products.md_path}")
    print(f"  json:    {products.json_path}")


if __name__ == "__main__":
    main()
