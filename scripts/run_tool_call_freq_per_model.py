"""Per-model tool-call name frequency across `500条最终仿真结果/` (recursive).

- Model = last `_`-segment of csv stem (e.g. `..._deepseek-v4-pro-th.csv`
  → `deepseek-v4-pro-th`). Same model across multiple csvs is summed.
- Filter: only names in `app.core.tools.manager.tools` are counted.
- Granularity: per-invocation (row [A,A,B] → A=2, B=1).

Outputs (in `data/outputs/report/tool_call_freq_per_model/<ts>/`):
  - per_model_counts_matrix.csv  (rows=tool, cols=model, cells=count)
  - per_model_counts_long.csv    (model, tool, count) long table
  - per_model_bars.png           (grid of horizontal bar charts)
  - per_model_summary.txt        (totals + top-K per model + dropped names)
"""
from __future__ import annotations

import ast
import csv
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

csv.field_size_limit(sys.maxsize)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("<项目地址>")
sys.path.insert(0, str(PROJECT_ROOT))

SRC_DIR = PROJECT_ROOT / "data/outputs/simulator_res"
OUT_DIR = PROJECT_ROOT / "data/outputs/report/tool_call_freq_per_model"


def setup_font():
    candidates = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "Noto Sans CJK SC",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.sans-serif"] = [c]
            plt.rcParams["axes.unicode_minus"] = False
            logger.info(f"using CJK font: {c}")
            return
    logger.warning("no CJK font found; non-ASCII labels may show as boxes")


def model_of(fn: str) -> str:
    stem = fn[:-4] if fn.endswith(".csv") else fn
    return stem.split("_")[-1]


def load_official_tool_names() -> set[str]:
    from app.core.tools.manager import tools as _tools
    names = {getattr(t, "name", None) for t in _tools}
    names.discard(None)
    return names  # type: ignore[return-value]


def count_one(fp: Path, official: set[str]) -> tuple[Counter, Counter, int]:
    """Return (counter_official, counter_dropped, n_parse_errors)."""
    counter: Counter = Counter()
    dropped: Counter = Counter()
    n_errors = 0
    with fp.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            tc = row.get("tool_calls") or ""
            if not tc.strip():
                continue
            try:
                obj = ast.literal_eval(tc)
            except Exception:
                n_errors += 1
                continue
            if not isinstance(obj, list):
                continue
            for entry in obj:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not (isinstance(name, str) and name.strip()):
                    dropped["__UNKNOWN__"] += 1
                    continue
                name = name.strip()
                if name in official:
                    counter[name] += 1
                else:
                    dropped[name] += 1
    return counter, dropped, n_errors


def render_grid(per_model: dict[str, Counter], png: Path) -> None:
    """One subplot per model in a grid; each subplot sorted by that model's count."""
    items = sorted(per_model.items(), key=lambda kv: kv[0])
    n = len(items)
    if n == 0:
        return
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 5 * nrows),
        squeeze=False,
    )
    for idx, (model, counter) in enumerate(items):
        ax = axes[idx // ncols][idx % ncols]
        if not counter:
            ax.set_title(f"{model}\n(no tool_calls)", fontsize=10)
            ax.axis("off")
            continue
        top = counter.most_common()
        names = [n for n, _ in top]
        counts = [c for _, c in top]
        bars = ax.barh(names, counts, color="#3b82f6")
        ax.invert_yaxis()
        ax.set_title(f"{model}\n({sum(counts):,} invocations, {len(counter)} tools)", fontsize=11)
        ax.bar_label(bars, labels=[f"{c:,}" for c in counts], padding=2, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.margins(x=0.18)

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle("Tool-call frequency per model (manager.py official tools only)",
                 fontsize=15, y=0.999)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def main() -> None:
    if not SRC_DIR.is_dir():
        sys.exit(f"src dir not found: {SRC_DIR}")
    setup_font()

    official = load_official_tool_names()
    print(f"official tools: {len(official)}\n")

    files = sorted(SRC_DIR.rglob("*.csv"))
    print(f"scanning {len(files)} csv files (recursive)\n")

    per_model: dict[str, Counter] = defaultdict(Counter)
    per_model_dropped: dict[str, Counter] = defaultdict(Counter)
    file_index: dict[str, list[Path]] = defaultdict(list)
    total_errors = 0

    for fp in files:
        m = model_of(fp.name)
        file_index[m].append(fp)
        c, dr, ne = count_one(fp, official)
        per_model[m].update(c)
        per_model_dropped[m].update(dr)
        total_errors += ne

    models = sorted(per_model.keys())
    print(f"{len(models)} distinct models\n")

    # global tool order by total count across all models (for matrix row order)
    grand: Counter = Counter()
    for m in models:
        grand.update(per_model[m])
    tool_order = [n for n, _ in grand.most_common()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUT_DIR / ts
    base.mkdir(parents=True, exist_ok=True)

    # 1) matrix csv: rows = tools (global desc), cols = models
    mat_csv = base / "per_model_counts_matrix.csv"
    with mat_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool"] + models + ["GRAND_TOTAL"])
        for tool in tool_order:
            row = [tool] + [per_model[m].get(tool, 0) for m in models] + [grand[tool]]
            w.writerow(row)
        # last row: per-model totals
        w.writerow(["__TOTAL__"] + [sum(per_model[m].values()) for m in models] + [sum(grand.values())])

    # 2) long csv
    long_csv = base / "per_model_counts_long.csv"
    with long_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "tool", "count"])
        for m in models:
            for t, c in per_model[m].most_common():
                w.writerow([m, t, c])

    # 3) grid PNG
    grid_png = base / "per_model_bars.png"
    render_grid(per_model, grid_png)

    # 4) summary
    sumtxt = base / "per_model_summary.txt"
    with sumtxt.open("w", encoding="utf-8") as f:
        f.write(f"Per-model tool-call frequency — {ts}\n")
        f.write(f"source: {SRC_DIR}\n")
        f.write(f"files scanned: {len(files)}  models: {len(models)}  "
                f"official tools: {len(official)}\n")
        f.write(f"grand invocations (official only): {sum(grand.values()):,}\n")
        f.write(f"parse errors: {total_errors}\n\n")
        f.write("=== per-model totals (sorted desc) ===\n")
        totals = sorted(((m, sum(per_model[m].values())) for m in models),
                        key=lambda kv: -kv[1])
        for m, t in totals:
            n_drop = sum(per_model_dropped[m].values())
            n_distinct = len(per_model[m])
            f.write(f"  {t:>7,}  {m}  (tools={n_distinct}, off_list_dropped={n_drop}, "
                    f"files={len(file_index[m])})\n")
        f.write("\n=== per-model top tools ===\n")
        for m in models:
            f.write(f"\n[{m}]  files={len(file_index[m])}  total={sum(per_model[m].values()):,}\n")
            for tool, cnt in per_model[m].most_common():
                f.write(f"  {cnt:>6,}  {tool}\n")
            if per_model_dropped[m]:
                f.write(f"  -- off-list dropped --\n")
                for tool, cnt in per_model_dropped[m].most_common():
                    f.write(f"  {cnt:>6,}  {tool}\n")

    # console summary
    print(f"{'model':<36}{'files':<7}{'tools':<8}{'invocations':<14}{'dropped':<10}")
    print("-" * 82)
    for m in models:
        nd = sum(per_model_dropped[m].values())
        print(f"{m:<36}{len(file_index[m]):<7}{len(per_model[m]):<8}"
              f"{sum(per_model[m].values()):<14,}{nd:<10}")
    print(f"\n=== products in {base} ===")
    print(f"  matrix csv -> {mat_csv.name}")
    print(f"  long csv   -> {long_csv.name}")
    print(f"  grid PNG   -> {grid_png.name}")
    print(f"  summary    -> {sumtxt.name}")


if __name__ == "__main__":
    main()
