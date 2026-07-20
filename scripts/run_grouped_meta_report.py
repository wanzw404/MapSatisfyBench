"""Cross-model metric report grouped by meta_info fields.

For each csv in `evaluation_res/report_ready/`, parse `results` (metrics)
and `meta_info` columns, then for each of the 9 grouping fields compute
per-(model, token) mean metrics:

    domain          csv string (split by ',') — multi-token per case
    difficulty      scalar
    sub_task        csv string — multi-token per case
    scene_tags      list — multi-token per case
    locality        scalar
    ambiguity_type  list — multi-token per case
    time_slot       scalar
    day_kind        scalar
    city_tier       scalar

Aggregation rules align with metrics_summary defaults:
  * 0/1 metrics (ECR/TS/IFS/IISR/AR/CEI): denominator = all cases in
    the group (failed cases counted as 0).
  * Eff: skip cases with details.Eff.skipped=true (mathematically invalid GT);
    failed cases keep Eff=0.0 and stay in the mean.
  * runtime (e2e_latency_ms, total_tokens): success-only mean to avoid 0
    poisoning.

Outputs to `data/outputs/report/grouped/<ts>/`:
  - grouped_summary.md   — human-readable, one section per field
  - grouped_summary.json — structured data for downstream analysis
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

csv.field_size_limit(sys.maxsize)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SRC_DIR = Path("<项目地址>/data/outputs/evaluation_res/report_ready")
OUT_DIR = Path("<项目地址>/data/outputs/report")

CSV_FIELDS = ("domain", "sub_task")            # comma-split multi-token
LIST_FIELDS = ("scene_tags", "ambiguity_type") # list multi-token
SCALAR_FIELDS = ("difficulty", "locality", "time_slot", "day_kind", "city_tier")
FIELDS = ("domain", "difficulty", "sub_task", "scene_tags", "locality",
          "ambiguity_type", "time_slot", "day_kind", "city_tier")

ZERO_OR_ONE_METRICS = ("ECR", "TS", "IFS", "IISR", "AR", "CEI")


def model_of(fn: str) -> str:
    stem = fn[:-4] if fn.endswith(".csv") else fn
    p = stem.split("_")
    return p[-2] if p[-1].startswith("run") else p[-1]


@dataclass
class CaseRow:
    case_id: str
    status: str
    metrics: dict[str, float]      # ECR/TS/.../CEI/e2e_latency_ms/total_tokens etc.
    eff_skipped: bool
    meta_info: dict[str, Any]


def _parse_results(cell: str) -> tuple[dict[str, float], bool]:
    if not cell or not cell.strip():
        return {}, False
    parsed = json.loads(cell)
    metrics = parsed.get("metrics") or {}
    if "ICR" in metrics and "ECR" not in metrics:
        metrics["ECR"] = metrics.pop("ICR")
    details = parsed.get("details") or {}
    eff_detail = details.get("Eff") or {}
    eff_skipped = bool(eff_detail.get("skipped"))
    return metrics, eff_skipped


def _zero_metrics() -> dict[str, float]:
    return {
        "ECR": 0.0, "TS": 0.0, "IFS": 0.0, "IISR": 0.0, "AR": 0.0,
        "Eff": 0.0, "CEI": 0.0,
        "e2e_latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
    }


def load_csv(fp: Path) -> list[CaseRow]:
    rows: list[CaseRow] = []
    with fp.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = (row.get("case_id") or "").strip()
            status = (row.get("status") or "").strip().lower()

            try:
                meta_info = json.loads(row.get("meta_info") or "{}")
                if not isinstance(meta_info, dict):
                    meta_info = {}
            except Exception:
                logger.warning(f"[{fp.name}/{cid}] meta_info parse fail; treated as empty")
                meta_info = {}

            if status in ("skipped", "error"):
                rows.append(CaseRow(cid, status, _zero_metrics(), eff_skipped=False, meta_info=meta_info))
                continue
            try:
                metrics, eff_skipped = _parse_results(row.get("results") or "")
                if not metrics:
                    metrics = _zero_metrics()
                    rows.append(CaseRow(cid, "unparseable", metrics, eff_skipped=False, meta_info=meta_info))
                    continue
                # Eff: None means skipped; if skipped, mark; otherwise ensure float
                if metrics.get("Eff") is None:
                    eff_skipped = True
                    metrics["Eff"] = 0.0  # placeholder, will be filtered when computing
                rows.append(CaseRow(cid, "success", metrics, eff_skipped=eff_skipped, meta_info=meta_info))
            except Exception as e:
                logger.warning(f"[{fp.name}/{cid}] results parse fail ({e}); zero-row")
                rows.append(CaseRow(cid, "unparseable", _zero_metrics(), eff_skipped=False, meta_info=meta_info))
    return rows


def parse_tokens(field_name: str, mi: dict) -> list[str]:
    v = mi.get(field_name)
    if v is None:
        return []
    if field_name in CSV_FIELDS:
        if not isinstance(v, str):
            return []
        return [t.strip() for t in v.split(",") if t.strip()]
    if field_name in LIST_FIELDS:
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if str(x) and str(x).strip()]
    s = str(v).strip()
    return [s] if s else []


@dataclass
class GroupMeans:
    n_cases: int = 0
    metrics: dict[str, float] = field(default_factory=dict)  # name -> mean
    n_eff: int = 0           # cases that contributed to Eff mean (excludes skipped)
    n_success: int = 0       # cases with status=success (used for runtime)


def compute_group_means(rows: list[CaseRow]) -> GroupMeans:
    n = len(rows)
    if n == 0:
        return GroupMeans()
    out = GroupMeans(n_cases=n)
    for k in ZERO_OR_ONE_METRICS:
        vals = [float(r.metrics.get(k, 0.0) or 0.0) for r in rows]
        out.metrics[k] = sum(vals) / n if vals else 0.0
    eff_vals = [float(r.metrics.get("Eff", 0.0) or 0.0) for r in rows if not r.eff_skipped]
    out.n_eff = len(eff_vals)
    out.metrics["Eff"] = (sum(eff_vals) / out.n_eff) if out.n_eff else 0.0
    success_rows = [r for r in rows if r.status == "success"]
    out.n_success = len(success_rows)
    if success_rows:
        ttft = [float(r.metrics.get("e2e_latency_ms", 0.0) or 0.0) for r in success_rows]
        in_t = [int(r.metrics.get("input_tokens", 0) or 0) for r in success_rows]
        out_t = [int(r.metrics.get("output_tokens", 0) or 0) for r in success_rows]
        tot_t = [int(r.metrics.get("total_tokens", 0) or 0) for r in success_rows]
        ns = len(success_rows)
        out.metrics["e2e_latency_ms"] = sum(ttft) / ns
        out.metrics["input_tokens"] = sum(in_t) / ns
        out.metrics["output_tokens"] = sum(out_t) / ns
        out.metrics["total_tokens"] = sum(tot_t) / ns
    else:
        out.metrics["e2e_latency_ms"] = 0.0
        out.metrics["input_tokens"] = 0.0
        out.metrics["output_tokens"] = 0.0
        out.metrics["total_tokens"] = 0.0
    return out


def fmt(v: float, kind: str) -> str:
    if kind == "01":
        return f"{v:.4f}"
    if kind == "ms":
        return f"{v:,.1f}"
    if kind == "tok":
        return f"{v:,.1f}"
    return str(v)


def render_md(grouped: dict, models: list[str], counts_per_model: dict[str, int]) -> str:
    """grouped: {field: {token: {model: GroupMeans}}}"""
    lines: list[str] = []
    lines.append(f"# Grouped Meta-Info Report")
    lines.append("")
    lines.append(f"- generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- source: `{SRC_DIR}`")
    lines.append(f"- models: {len(models)} (each: 500 cases unless noted)")
    lines.append("")
    lines.append("**Aggregation rules**")
    lines.append("- 0/1 metrics (ECR/TS/IFS/IISR/AR/CEI): mean over all cases in group; failed cases count as 0.")
    lines.append("- Eff: skip cases with `details.Eff.skipped=true`; failed cases keep Eff=0 in mean. `n_eff` shows participating count.")
    lines.append("- Runtime (E2E latency / Tokens): success-only mean; `n_success` shows count.")
    lines.append("- Multi-value fields (`domain`/`sub_task`/`scene_tags`/`ambiguity_type`): a case is counted in **every** token-group it belongs to.")
    lines.append("")
    lines.append("**Per-model overall n_cases**: " + ", ".join(f"`{m}`={counts_per_model[m]}" for m in models))
    lines.append("")

    for fld in FIELDS:
        lines.append(f"## 按 `{fld}` 分组")
        lines.append("")
        token_data = grouped.get(fld, {})
        if not token_data:
            lines.append("_(no tokens)_")
            lines.append("")
            continue

        # token order: by total n_cases across all models desc
        def total_n(tok: str) -> int:
            return sum(token_data[tok].get(m, GroupMeans()).n_cases for m in models)
        ordered_tokens = sorted(token_data.keys(), key=lambda t: -total_n(t))

        # Render: one sub-table per token (model x metric)
        for tok in ordered_tokens:
            per_model = token_data[tok]
            tot_n = total_n(tok)
            lines.append(f"### `{tok}` _(total cases across models: {tot_n})_")
            lines.append("")
            lines.append(
                "| Model | n | ECR | TS | IFS | IISR | AR | Eff (n_eff) | CEI | "
                "E2E (ms, n_succ) | Input (mean) | Output (mean) | Total (mean) |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for m in models:
                gm = per_model.get(m, GroupMeans())
                if gm.n_cases == 0:
                    continue
                me = gm.metrics
                lines.append(
                    f"| {m} | {gm.n_cases} | {fmt(me.get('ECR',0),'01')} | {fmt(me.get('TS',0),'01')} | "
                    f"{fmt(me.get('IFS',0),'01')} | {fmt(me.get('IISR',0),'01')} | "
                    f"{fmt(me.get('AR',0),'01')} | "
                    f"{fmt(me.get('Eff',0),'01')} ({gm.n_eff}) | {fmt(me.get('CEI',0),'01')} | "
                    f"{fmt(me.get('e2e_latency_ms',0),'ms')} ({gm.n_success}) | "
                    f"{fmt(me.get('input_tokens',0),'tok')} | "
                    f"{fmt(me.get('output_tokens',0),'tok')} | "
                    f"{fmt(me.get('total_tokens',0),'tok')} |"
                )
            lines.append("")
    return "\n".join(lines)


def render_json(grouped: dict, models: list[str], counts_per_model: dict[str, int]) -> dict:
    out: dict[str, Any] = {
        "generated": datetime.now().isoformat(),
        "source_dir": str(SRC_DIR),
        "models": models,
        "n_cases_per_model": counts_per_model,
        "fields": {},
    }
    for fld in FIELDS:
        token_data = grouped.get(fld, {})
        out["fields"][fld] = {}
        for tok, per_model in sorted(token_data.items()):
            out["fields"][fld][tok] = {}
            for m in models:
                gm = per_model.get(m)
                if gm is None or gm.n_cases == 0:
                    continue
                out["fields"][fld][tok][m] = {
                    "n_cases": gm.n_cases,
                    "n_eff": gm.n_eff,
                    "n_success": gm.n_success,
                    "metrics": {k: round(v, 4) for k, v in gm.metrics.items()},
                }
    return out


def main() -> None:
    if not SRC_DIR.is_dir():
        sys.exit(f"src dir not found: {SRC_DIR}")
    files = sorted(SRC_DIR.glob("*.csv"))
    if not files:
        sys.exit(f"no csv in {SRC_DIR}")

    # model name preserves file order
    by_model: dict[str, Path] = {}
    for fp in files:
        m = model_of(fp.name)
        if m in by_model:
            logger.warning(f"duplicate model {m} (skipping {fp.name})")
            continue
        by_model[m] = fp
    models = list(by_model.keys())
    print(f"models: {len(models)}  -> {models}\n")

    # grouped[field][token][model] = GroupMeans
    grouped: dict[str, dict[str, dict[str, GroupMeans]]] = {f: defaultdict(dict) for f in FIELDS}
    counts_per_model: dict[str, int] = {}

    for m, fp in by_model.items():
        rows = load_csv(fp)
        counts_per_model[m] = len(rows)
        # group rows per field per token
        for fld in FIELDS:
            buckets: dict[str, list[CaseRow]] = defaultdict(list)
            for r in rows:
                for tok in parse_tokens(fld, r.meta_info):
                    buckets[tok].append(r)
            for tok, group_rows in buckets.items():
                grouped[fld][tok][m] = compute_group_means(group_rows)
        print(f"  [{m}] rows={len(rows)} loaded")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUT_DIR / "grouped" / ts
    base.mkdir(parents=True, exist_ok=True)
    md_path = base / "grouped_summary.md"
    json_path = base / "grouped_summary.json"

    md_text = render_md(grouped, models, counts_per_model)
    md_path.write_text(md_text, encoding="utf-8")

    json_obj = render_json(grouped, models, counts_per_model)
    json_path.write_text(
        json.dumps(json_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== products ===")
    print(f"  md   -> {md_path}")
    print(f"  json -> {json_path}")
    # quick stat
    n_groups = sum(len(grouped[f]) for f in FIELDS)
    print(f"  total tokens across 9 fields: {n_groups}")


if __name__ == "__main__":
    main()
