"""Count tool_calls.name occurrences across all csvs under
`500条最终仿真结果/` (recursive), output horizontal bar chart.

- Granularity: per-invocation (a row with [A,A,B] counts as A=2, B=1).
- Per-file subplot grid + one global bar chart.
- tool_calls cell is Python-repr; parsed via ast.literal_eval.
- csv.field_size_limit(sys.maxsize) for long cells.
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
OUT_DIR = PROJECT_ROOT / "data/outputs/report/tool_call_freq"


def load_official_tool_names() -> set[str]:
    """Pull the canonical tool list from app.core.tools.manager (live import).

    Uses tool.name (not the variable name) so the manager file is the single
    source of truth.
    """
    from app.core.tools.manager import tools as _tools
    names = {getattr(t, "name", None) for t in _tools}
    names.discard(None)
    return names  # type: ignore[return-value]


def setup_font():
    """Try to find a CJK-capable font so Chinese tool names (if any) render."""
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


def model_label(fp: Path) -> str:
    """Friendly label for a csv: <parent_subdir>/<model>."""
    stem = fp.stem
    parts = stem.split("_")
    model = parts[-1] if parts else stem
    parent = fp.parent.name
    if parent == SRC_DIR.name:
        return model
    return f"{parent}/{model}"


def count_one(fp: Path, official: set[str]) -> tuple[Counter, Counter, int, int]:
    """Count tool-call name occurrences in one csv.

    Returns (counter_official, counter_dropped_non_official, n_rows_with_tool_calls, n_parse_errors).
    Only names present in ``official`` go into the main counter; off-list names
    are tracked separately for visibility (and never plotted/exported as official).
    """
    counter: Counter = Counter()
    dropped: Counter = Counter()
    n_rows = 0
    n_errors = 0
    with fp.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            tc = row.get("tool_calls") or ""
            if not tc.strip():
                continue
            n_rows += 1
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
    return counter, dropped, n_rows, n_errors


def render_global_bar(counter: Counter, png: Path) -> None:
    items = counter.most_common()
    if not items:
        logger.warning("no data for global chart")
        return
    names = [n for n, _ in items]
    counts = [c for _, c in items]
    h = max(4, 0.32 * len(names))
    fig, ax = plt.subplots(figsize=(12, h))
    bars = ax.barh(names, counts, color="#3b82f6")
    ax.invert_yaxis()
    ax.set_xlabel("调用次数 (per-invocation count)")
    ax.set_title(f"Tool-call name frequency — global (across {len(counter)} distinct tools, {sum(counts):,} invocations)")
    ax.bar_label(bars, labels=[f"{c:,}" for c in counts], padding=3, fontsize=9)
    ax.margins(x=0.10)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def render_per_file_grid(per_file: dict[str, Counter], png: Path) -> None:
    """One subplot per file, arranged in a grid."""
    items = sorted(per_file.items(), key=lambda kv: kv[0])
    n = len(items)
    if n == 0:
        return
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 5 * nrows),
        squeeze=False,
    )
    for idx, (label, counter) in enumerate(items):
        ax = axes[idx // ncols][idx % ncols]
        if not counter:
            ax.set_title(f"{label}\n(no tool_calls)", fontsize=10)
            ax.axis("off")
            continue
        top = counter.most_common()
        names = [n for n, _ in top]
        counts = [c for _, c in top]
        bars = ax.barh(names, counts, color="#10b981")
        ax.invert_yaxis()
        ax.set_title(f"{label}\n({sum(counts):,} invocations, {len(counter)} tools)", fontsize=10)
        ax.bar_label(bars, labels=[f"{c:,}" for c in counts], padding=2, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.margins(x=0.18)

    # blank out unused axes
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    fig.suptitle("Tool-call name frequency — per file", fontsize=14, y=0.999)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)


def main() -> None:
    if not SRC_DIR.is_dir():
        sys.exit(f"src dir not found: {SRC_DIR}")
    setup_font()

    official = load_official_tool_names()
    print(f"official tool list (manager.py): {len(official)} tools")
    for n in sorted(official):
        print(f"  - {n}")
    print()

    files = sorted(SRC_DIR.rglob("*.csv"))
    print(f"scanning {len(files)} csv files (recursive)\n")

    global_counter: Counter = Counter()
    global_dropped: Counter = Counter()
    per_file: dict[str, Counter] = {}
    total_rows = total_errors = 0

    for fp in files:
        label = model_label(fp)
        c, dropped, nr, ne = count_one(fp, official)
        per_file[label] = c
        total_rows += nr
        total_errors += ne
        global_counter.update(c)
        global_dropped.update(dropped)
        n_inv = sum(c.values())
        n_drop = sum(dropped.values())
        print(
            f"  {label:<60} rows_with_tc={nr:<6} invocations={n_inv:<7} "
            f"tools={len(c)}  off_list_dropped={n_drop}  parse_errors={ne}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUT_DIR / ts
    base.mkdir(parents=True, exist_ok=True)

    # 1) global PNG
    global_png = base / "tool_call_freq_global.png"
    render_global_bar(global_counter, global_png)

    # 2) per-file grid PNG
    per_file_png = base / "tool_call_freq_per_file.png"
    render_per_file_grid(per_file, per_file_png)

    # 3) global counts csv (sorted desc)
    global_csv = base / "tool_call_counts_global.csv"
    with global_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "count"])
        for n, c in global_counter.most_common():
            w.writerow([n, c])

    # 4) per-file long csv
    per_file_csv = base / "tool_call_counts_per_file.csv"
    with per_file_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_label", "name", "count"])
        for label, cnt in per_file.items():
            for n, c in cnt.most_common():
                w.writerow([label, n, c])

    # 5) summary txt (also lists official tools never called, and off-list drops)
    zero_usage = sorted(official - set(global_counter.keys()))
    summary_txt = base / "tool_call_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"Tool-call frequency report — {ts}\n")
        f.write(f"source: {SRC_DIR}\n")
        f.write(f"official tools (manager.py): {len(official)}\n")
        f.write(f"files scanned: {len(files)}  rows with tool_calls: {total_rows:,}\n")
        f.write(f"counted invocations (official only): {sum(global_counter.values()):,}\n")
        f.write(f"distinct official tools that appeared: {len(global_counter)} / {len(official)}\n")
        f.write(f"off-list invocations dropped: {sum(global_dropped.values())}\n")
        f.write(f"parse errors (ast.literal_eval): {total_errors}\n\n")
        f.write("=== global counts (official only, sorted desc) ===\n")
        for n, c in global_counter.most_common():
            f.write(f"  {c:>7,}  {n}\n")
        if zero_usage:
            f.write(f"\n=== official tools never called ({len(zero_usage)}) ===\n")
            for n in zero_usage:
                f.write(f"  {n}\n")
        if global_dropped:
            f.write(f"\n=== off-list names dropped from count ({len(global_dropped)}) ===\n")
            for n, c in global_dropped.most_common():
                f.write(f"  {c:>7,}  {n}\n")

    print(f"\n=== products in {base} ===")
    print(f"  PNG  global  -> {global_png.name}")
    print(f"  PNG  per-file -> {per_file_png.name}")
    print(f"  CSV  global  -> {global_csv.name}")
    print(f"  CSV  per-file -> {per_file_csv.name}")
    print(f"  TXT  summary -> {summary_txt.name}")
    print(f"\nglobal: {len(global_counter)} distinct tools, {sum(global_counter.values()):,} invocations")


if __name__ == "__main__":
    main()
