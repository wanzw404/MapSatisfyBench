#!/usr/bin/env python3
"""把同一文件夹里 3 份模型评测 csv 按 case_id 对齐 + 多数投票合并到一份 csv。

输入：一个文件夹（``--folder``），里面**恰好 3 份** ``.csv``。每份 csv 至少
含两列（命名容差：case_id/caseid/id；results/result，BOM 自动处理）：
* ``case_id`` — 案例唯一标识
* ``results`` — JSON 字符串，``{"metrics": {...}, "details": {...}}`` 形态
  （即 eval_result 中 ``results`` 块本身，不含外层 ``case_id`` 包裹）

合并方法（与 ``scripts/merge_models_and_compare.py`` 完全对齐）：

* **IISR** —— 按 ``details.AR.iisr_breakdown.rubrics_detail`` 逐条 Ci 多数投票
  （3-same / 2-same → 众数；3-different → 退第一份）。合并后重算
  ``iisr_component = Σ Wi·Ci / Σ Wi`` 与 ``ar = ecr · iisr``。
* **IFS** —— 对 3 个模型 ``metrics.IFS`` 做 case-level 标量投票
  （3-same / 2-same → 众数；3-different → **中位数**）。
* **ECR / TS / Eff / avg_ttft_ms / *_tokens** —— 不投票，直接抄
  **按文件名排序后的第一份 csv** 的值。
* **AR** —— ``ECR_first · IISR_merged`` 重算。

输出（``--out``，默认 ``<folder.parent>/<folder.name>.csv``）：
* col1 = ``case_id``
* col2 = ``results``，合并后的 ``{"metrics": {...}, "details": {...}}`` JSON
  （默认 ``indent=2`` + 换行；``--compact`` 改紧凑单行）

用法：
    python scripts/merge_csv_results_by_case.py --folder /path/to/folder
    python scripts/merge_csv_results_by_case.py --folder /path/to/folder --out /tmp/merged.csv
    python scripts/merge_csv_results_by_case.py --folder /path/to/folder --compact
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# csv 字段可能很长（评测 results JSON 单行紧凑常 > 100KB），提前放开 csv 限制
csv.field_size_limit(sys.maxsize)


# ─────────────────────────────────────────────────────────────────────
# 内联自 generate_eval_sh.py
# ─────────────────────────────────────────────────────────────────────

def _clean_json_str(raw: str) -> str:
    """清理 Excel 中的非标准空格(\xa0)等字符，使 JSON 可解析。"""
    if not isinstance(raw, str):
        return ""
    return raw.replace("\xa0", " ").strip()


def parse_json_safe(raw: str) -> dict | None:
    """安全解析 JSON 字符串。

    用 ``raw_decode`` 而非 ``loads``：上游 xlsx 里部分单元格末尾会带一个
    多余的逗号（``},``）或重复对象，``loads`` 会报 ``Extra data``，但
    ``raw_decode`` 只返回第一个完整 JSON 对象，把尾部杂质忽略掉。
    """
    cleaned = _clean_json_str(raw)
    if not cleaned:
        return None
    try:
        decoder = json.JSONDecoder()
        obj, _end = decoder.raw_decode(cleaned)
        return obj
    except json.JSONDecodeError as e:
        stripped = cleaned.lstrip(",;\n\r\t ").rstrip(",;\n\r\t ")
        try:
            decoder = json.JSONDecoder()
            obj, _end = decoder.raw_decode(stripped)
            return obj
        except json.JSONDecodeError:
            print(f"  ⚠️  JSON 解析失败: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────
# 内联自 merge_models_and_compare.py
# ─────────────────────────────────────────────────────────────────────

def _ungated_iisr(bd: dict) -> float | None:
    """从 ``iisr_breakdown`` 算出 ungated IISR = Σ Wi·Ci / Σ Wi。"""
    if not isinstance(bd, dict):
        return None
    ws = bd.get("weighted_sum")
    tw = bd.get("total_weight")
    if not isinstance(ws, (int, float)) or not isinstance(tw, (int, float)):
        return None
    if tw <= 0:
        return None
    return max(0.0, min(1.0, ws / tw))


def _unwrap_ar_details(d: dict) -> dict:
    """兼容两种输入：bare AR_details / 完整 eval_result，统一拍平成 bare 形态。"""
    if not isinstance(d, dict):
        return {}
    if "iisr_breakdown" in d or "ecr_component" in d or "icr_component" in d:
        bare = dict(d)
        if "icr_component" in bare and "ecr_component" not in bare:
            bare["ecr_component"] = bare.pop("icr_component")
        bd = bare.get("iisr_breakdown") or {}
        ungated = _ungated_iisr(bd)
        if ungated is not None:
            bare["iisr_component"] = ungated
            ecr = bare.get("ecr_component")
            if isinstance(ecr, (int, float)):
                bare["ar"] = float(ecr) * ungated
            inner_metrics = bare.get("metrics")
            if isinstance(inner_metrics, dict):
                inner_metrics = dict(inner_metrics)
                inner_metrics["IISR"] = ungated
                if isinstance(ecr, (int, float)):
                    inner_metrics["AR"] = float(ecr) * ungated
                bare["metrics"] = inner_metrics
        return bare
    results = d.get("results")
    if not isinstance(results, dict) or not results:
        if "metrics" in d or "details" in d:
            results = d
        else:
            results = {}
    details = results.get("details") or {}
    metrics = results.get("metrics") or {}
    ar = dict(details.get("AR") or {})
    if "icr_component" in ar and "ecr_component" not in ar:
        ar["ecr_component"] = ar.pop("icr_component")
    ecr_key = "ECR" if "ECR" in metrics else "ICR"
    if ecr_key in metrics and "ecr_component" not in ar:
        ar["ecr_component"] = metrics[ecr_key]
    if "iisr_component" not in ar and "IISR" in metrics:
        ar["iisr_component"] = metrics["IISR"]
    ifs_block = details.get("IFS") or {}
    if "ifs_rows" not in ar:
        ar["ifs_rows"] = list(ifs_block.get("rows_detail") or [])
    if "ifs_component" not in ar:
        if "IFS" in metrics:
            ar["ifs_component"] = metrics["IFS"]
        elif "_merged_ifs" in ifs_block:
            ar["ifs_component"] = ifs_block["_merged_ifs"]
    if metrics and "metrics" not in ar:
        ar["metrics"] = dict(metrics)
    ungated = _ungated_iisr(ar.get("iisr_breakdown") or {})
    if ungated is not None:
        ar["iisr_component"] = ungated
        ecr = ar.get("ecr_component")
        if isinstance(ecr, (int, float)):
            ar["ar"] = float(ecr) * ungated
        inner_metrics = ar.get("metrics")
        if isinstance(inner_metrics, dict):
            inner_metrics["IISR"] = ungated
            if isinstance(ecr, (int, float)):
                inner_metrics["AR"] = float(ecr) * ungated
    return ar


def _wrap_into_eval_result(
    full_d1: dict,
    merged_bare: dict,
    full_d2: dict | None = None,
    full_d3: dict | None = None,
) -> dict:
    """把 bare 形态的合并结果回填到 d1 的原 schema 里。"""
    if not isinstance(full_d1, dict):
        return merged_bare
    has_results = isinstance(full_d1.get("results"), dict) and full_d1["results"]
    has_inner = "metrics" in full_d1 or "details" in full_d1
    if not has_results and not has_inner:
        return merged_bare

    candidates = [full_d1, full_d2, full_d3]

    def _get_candidate_details(d: dict | None) -> dict:
        if not isinstance(d, dict):
            return {}
        r = d.get("results")
        if isinstance(r, dict) and r:
            return r.get("details") or {}
        return d.get("details") or {}

    def _get_candidate_metrics(d: dict | None) -> dict:
        if not isinstance(d, dict):
            return {}
        r = d.get("results")
        if isinstance(r, dict) and r:
            return r.get("metrics") or {}
        return d.get("metrics") or {}

    out = deepcopy(full_d1)
    results = out.setdefault("results", {}) if has_results else out
    metrics_out = results.setdefault("metrics", {})
    merged_metrics = merged_bare.get("metrics") or {}
    for k in ("ECR", "IISR", "IFS", "AR", "TS"):
        if k in merged_metrics:
            metrics_out[k] = merged_metrics[k]

    details = results.setdefault("details", {})
    ar_out = details.setdefault("AR", {})
    if "iisr_breakdown" in merged_bare:
        ar_out["iisr_breakdown"] = merged_bare["iisr_breakdown"]
    for k in ("ecr_component", "iisr_component", "ar"):
        if k in merged_bare:
            ar_out[k] = merged_bare[k]

    # IFS winner details
    ifs_winner = merged_bare.get("_ifs_winner_value")
    if ifs_winner is not None:
        winner_ifs_detail = None
        for cand in candidates:
            cand_metrics = _get_candidate_metrics(cand)
            cand_ifs = cand_metrics.get("IFS")
            if isinstance(cand_ifs, (int, float)) and round(float(cand_ifs), 4) == ifs_winner:
                winner_ifs_detail = deepcopy(_get_candidate_details(cand).get("IFS"))
                break
        if winner_ifs_detail:
            winner_ifs_detail["_merged_ifs"] = ifs_winner
            details["IFS"] = winner_ifs_detail
        else:
            ifs_out = details.setdefault("IFS", {})
            ifs_out["_merged_ifs"] = ifs_winner

    # ECR winner details
    ecr_winner = merged_bare.get("_ecr_winner_value")
    if ecr_winner is not None:
        for cand in candidates:
            cand_metrics = _get_candidate_metrics(cand)
            cand_ecr = cand_metrics.get("ECR") or cand_metrics.get("ICR")
            if isinstance(cand_ecr, (int, float)) and round(float(cand_ecr), 4) == ecr_winner:
                winner_ecr_detail = _get_candidate_details(cand).get("ECR") or _get_candidate_details(cand).get("ICR")
                if winner_ecr_detail:
                    details["ECR"] = deepcopy(winner_ecr_detail)
                break

    # TS winner details
    ts_winner = merged_bare.get("_ts_winner_value")
    if ts_winner is not None:
        for cand in candidates:
            cand_metrics = _get_candidate_metrics(cand)
            cand_ts = cand_metrics.get("TS")
            if isinstance(cand_ts, (int, float)) and round(float(cand_ts), 4) == ts_winner:
                winner_ts_detail = _get_candidate_details(cand).get("TS")
                if winner_ts_detail:
                    details["TS"] = deepcopy(winner_ts_detail)
                break

    for key in ("_ifs_winner_value", "_ecr_winner_value",
                "_ts_winner_value"):
        merged_bare.pop(key, None)
    return out


def _get_rubrics(detail: dict) -> list[dict]:
    detail = _unwrap_ar_details(detail)
    bd = detail.get("iisr_breakdown") or {}
    return list(bd.get("rubrics_detail") or bd.get("rows_detail") or [])


def _vote_ci(ci_values: list[float]) -> tuple[float, str]:
    cnt = Counter(round(v, 6) for v in ci_values)
    most, freq = cnt.most_common(1)[0]
    if freq == 3:
        return ci_values[0], "3个都相同"
    if freq == 2:
        return most, "有2个相同"
    return ci_values[0], "3个都不同"


def _vote_metric_with_none(values: list[float | None]) -> tuple[float | None, str]:
    if len(values) != 3:
        raise ValueError(f"_vote_metric_with_none expects 3 values, got {len(values)}")
    none_count = sum(1 for v in values if v is None)
    non_none_values = [v for v in values if v is not None]
    if none_count >= 2:
        return None, "≥2次为None"
    if none_count == 1:
        if len(non_none_values) == 2:
            v1, v2 = round(non_none_values[0], 6), round(non_none_values[1], 6)
            if v1 == v2:
                return non_none_values[0], "1次None且另两次相同"
            else:
                return None, "1次None且另两次不同"
        return None, "1次None且非None值不足2个"
    rounded = [round(v, 6) for v in values]
    cnt = Counter(rounded)
    most, freq = cnt.most_common(1)[0]
    if freq == 3:
        return values[0], "3个都相同"
    if freq == 2:
        return next(v for v, r in zip(values, rounded) if r == most), "有2个相同"
    return sorted(values)[1], "3个都不同, 取中位数"


def _vote_scalar(
    d1: dict, d2: dict, d3: dict, *, getter, label: str
) -> tuple[float | None, str]:
    vals: list[float | None] = []
    for d in (d1, d2, d3):
        v = getter(d) if d is not None else None
        if isinstance(v, (int, float)):
            vals.append(float(v))
        else:
            vals.append(None)
    if len([v for v in vals if v is not None]) == 3:
        v, note = _vote_metric_with_none(vals)
        return v, note
    if len([v for v in vals if v is not None]) >= 1:
        return vals[0], f"仅{len([v for v in vals if v is not None])}列有{label}"
    return None, f"无{label}"


def merge_three(d1: dict, d2: dict, d3: dict) -> tuple[dict, list[str]]:
    """合并 3 份 AR_details，返回 (merged_json, per_rubric_note_list)。"""
    merged = deepcopy(d1)
    bd = merged.get("iisr_breakdown") or {}
    rubrics_a = _get_rubrics(d1)
    rubrics_b = _get_rubrics(d2)
    rubrics_c = _get_rubrics(d3)
    notes: list[str] = []

    if max(len(rubrics_a), len(rubrics_b), len(rubrics_c)) == 0:
        iisr_vals: list[float | None] = [
            float(d.get("iisr_component"))
            if d is not None and isinstance(d.get("iisr_component"), (int, float))
            else None
            for d in (d1, d2, d3)
        ]
        if len([v for v in iisr_vals if v is not None]) == 3:
            iisr, iisr_note = _vote_metric_with_none(iisr_vals)
        elif len([v for v in iisr_vals if v is not None]) >= 1:
            non_none = [v for v in iisr_vals if v is not None]
            iisr, iisr_note = non_none[0], f"仅{len(non_none)}列有IISR(scalar)"
        else:
            iisr, iisr_note = None, "无 IISR(scalar)，且无 rubrics_detail"
        bd["rubrics_detail"] = []
        bd["weighted_sum"] = None
        bd["total_weight"] = None
        bd["n_rubrics_in_gt"] = 0
        bd["n_rubrics_scored"] = 0
        bd["_fallback_to_scalar"] = True
        notes.append(f"#IISR scalar fallback {iisr_note}")
    else:
        merged_rubrics = []
        weighted_sum = 0.0
        total_weight = 0.0
        n = max(len(rubrics_a), len(rubrics_b), len(rubrics_c))
        for i in range(n):
            ra = rubrics_a[i] if i < len(rubrics_a) else None
            rb = rubrics_b[i] if i < len(rubrics_b) else None
            rc = rubrics_c[i] if i < len(rubrics_c) else None
            base = ra if ra is not None else (rb if rb is not None else rc)
            if base is None:
                continue
            new_row = deepcopy(base)
            cis = [
                r.get("Ci") for r in (ra, rb, rc)
                if r is not None and isinstance(r.get("Ci"), (int, float))
            ]
            if len(cis) == 3:
                chosen, note = _vote_ci([float(x) for x in cis])
            elif len(cis) >= 1:
                chosen, note = float(cis[0]), f"仅{len(cis)}列有Ci"
            else:
                chosen, note = 0.0, "无Ci"
            new_row["Ci"] = chosen
            wi = float(new_row.get("Wi") or 0.0)
            new_row["weighted_contribution"] = round(wi * chosen, 4)
            merged_rubrics.append(new_row)
            notes.append(f"#{i+1} {note}")
            weighted_sum += wi * chosen
            total_weight += wi
        bd["rubrics_detail"] = merged_rubrics
        bd["weighted_sum"] = round(weighted_sum, 4)
        bd["total_weight"] = round(total_weight, 4)
        bd["n_rubrics_in_gt"] = len(merged_rubrics)
        bd["n_rubrics_scored"] = len(merged_rubrics)
        iisr = (weighted_sum / total_weight) if total_weight > 0 else 0.0

    # ECR
    ecr_value, ecr_note = _vote_scalar(
        d1, d2, d3, getter=lambda d: d.get("ecr_component"), label="ECR",
    )
    if ecr_value is None:
        ecr_value, ecr_note = 0.0, "无ECR"
    ecr = float(ecr_value)
    merged["ecr_component"] = round(ecr, 4)
    notes.append(f"#ECR metric {ecr_note}")

    if iisr is not None:
        merged["iisr_component"] = round(iisr, 4)
        merged["ar"] = round(ecr * iisr, 4)
    else:
        merged["iisr_component"] = None
        merged["ar"] = None
    merged["iisr_breakdown"] = bd

    # IFS
    ifs_value, ifs_note = _vote_scalar(
        d1, d2, d3, getter=lambda d: d.get("ifs_component"), label="IFS",
    )
    if ifs_value is None:
        ifs_value, ifs_note = 0.0, "无IFS"
    merged["ifs_component"] = round(float(ifs_value), 4)
    ifs_winner_rows = list(d1.get("ifs_rows") or [])
    for d_candidate in (d1, d2, d3):
        candidate_ifs = d_candidate.get("ifs_component") if d_candidate else None
        if isinstance(candidate_ifs, (int, float)) and round(float(candidate_ifs), 4) == round(float(ifs_value), 4):
            ifs_winner_rows = list(d_candidate.get("ifs_rows") or [])
            break
    merged["ifs_rows"] = ifs_winner_rows
    merged["_ifs_winner_value"] = round(float(ifs_value), 4)
    notes.append(f"#IFS metric {ifs_note}")

    # TS
    ts_value, ts_note = _vote_scalar(
        d1, d2, d3, getter=lambda d: (d.get("metrics") or {}).get("TS"), label="TS",
    )
    notes.append(f"#TS metric {ts_note}")
    merged["_ts_winner_value"] = round(float(ts_value), 4) if ts_value is not None else None

    merged["_ecr_winner_value"] = round(ecr, 4)

    merged_metrics = dict(d1.get("metrics") or {})
    merged_metrics["ECR"] = round(ecr, 4)
    merged_metrics.pop("ICR", None)
    merged_metrics["IISR"] = round(iisr, 4) if iisr is not None else None
    merged_metrics["IFS"] = round(float(ifs_value), 4) if ifs_value is not None else None
    merged_metrics["AR"] = round(ecr * iisr, 4) if iisr is not None else None
    if ts_value is not None:
        merged_metrics["TS"] = round(float(ts_value), 4)
    else:
        merged_metrics["TS"] = None
    merged["metrics"] = merged_metrics

    return merged, notes


def _norm(s: str | None) -> str:
    return (s or "").strip().lstrip("﻿").lower()


def load_csv(path: Path) -> dict[str, dict[str, str]]:
    """case_id -> {'results': ..., 'status': ..., 'error': ..., 'reason': ...}。
    status / error / reason / parse_errors 列存在时一并捕获，缺失则为空串。
    用法：``maps[i][cid]["results"]`` 取 results 字符串。
    """
    out: dict[str, dict[str, str]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"CSV header 为空: {path}")
        field_map = {_norm(fn): fn for fn in reader.fieldnames}
        case_key = (
            field_map.get("case_id")
            or field_map.get("caseid")
            or field_map.get("id")
        )
        results_key = field_map.get("results") or field_map.get("result")
        status_key = field_map.get("status")
        error_key = field_map.get("error")
        reason_key = field_map.get("reason")
        parse_err_key = field_map.get("parse_errors")
        if not case_key or not results_key:
            raise SystemExit(
                f"CSV header 缺 case_id / results 列 ({path.name}): "
                f"{reader.fieldnames}"
            )
        for row in reader:
            cid = (row.get(case_key) or "").strip()
            if not cid:
                continue
            out[cid] = {
                "results": row.get(results_key) or "",
                "status": (row.get(status_key) or "") if status_key else "",
                "error": (row.get(error_key) or "") if error_key else "",
                "reason": (row.get(reason_key) or "") if reason_key else "",
                "parse_errors": (
                    (row.get(parse_err_key) or "") if parse_err_key else ""
                ),
            }
    return out


def _format_json_str(d: dict, *, compact: bool) -> str:
    return (
        json.dumps(d, ensure_ascii=False)
        if compact
        else json.dumps(d, ensure_ascii=False, indent=2)
    )


def process_folder(
    folder: Path,
    out_path: Path | None = None,
    *,
    compact: bool = False,
    check: bool = False,
) -> dict[str, int]:
    """对单个文件夹跑合并；返回 stats dict（merged / parse_fail / wrap_fail / skipped）。

    out_path=None 时默认 <folder.parent>/<folder.name>.csv（``check=True``
    时改 <folder.name>_check.csv）。

    check=True 时输出 5 列：model1_results / model2_results / model3_results
    / case_id / results；前 3 列是 3 份输入 csv 的原始 results（按文件名
    排序后的顺序，与 merge 用的"第一份"对齐）。

    遇到 csv 数量 ≠ 3 直接返回 skipped 并打印 warn，不抛异常。
    """
    csv_files = sorted(folder.glob("*.csv"))
    if len(csv_files) != 3:
        print(
            f"[skip] {folder.name}: csv 数量={len(csv_files)} ≠ 3，"
            f"内容={[p.name for p in csv_files]}"
        )
        return {
            "merged": 0, "parse_fail": 0, "wrap_fail": 0,
            "skipped_folder": 1, "missing_align": 0,
            # 把跳过的明细也回传，让批量汇总能在末尾再列一次
            "_skipped_name": folder.name,
            "_skipped_csvs": [p.name for p in csv_files],
        }

    if out_path is None:
        suffix = "_check" if check else ""
        out_path = folder.parent / f"{folder.name}{suffix}.csv"
    if out_path.parent.resolve() == folder.resolve():
        print(
            f"[warn] {folder.name}: --out 落在输入文件夹内，下次跑批会被认成第 4 份输入"
        )

    print(f"\n[info] folder: {folder}")
    for i, p in enumerate(csv_files):
        print(f"[info]   csv{i+1}: {p.name}")

    maps = [load_csv(p) for p in csv_files]
    for p, m in zip(csv_files, maps):
        print(f"[info] {p.name} 行数: {len(m)}")

    ids_first = list(maps[0].keys())
    set1, set2, set3 = set(maps[0]), set(maps[1]), set(maps[2])
    common = [cid for cid in ids_first if cid in set2 and cid in set3]
    missing_from_first = (set2 | set3) - set1
    missing_in_others = [cid for cid in ids_first if cid not in (set2 & set3)]

    print(
        f"[info] 3 份都对齐: {len(common)}    "
        f"在 csv1 缺其它列: {len(missing_in_others)}    "
        f"csv2/3 独有: {len(missing_from_first)}"
    )

    rows_out: list[list[str]] = []
    failure_records: list[dict] = []  # NEW: 逐 case 失败明细
    stats = {"merged": 0, "parse_fail": 0, "wrap_fail": 0,
             "skipped_folder": 0, "missing_align": len(missing_in_others)}

    # 「未对齐丢弃」也写一份失败记录，便于汇总
    for cid in missing_in_others:
        present = [csv_files[i].name for i, mm in enumerate(maps) if cid in mm]
        absent = [csv_files[i].name for i, mm in enumerate(maps) if cid not in mm]
        failure_records.append({
            "case_id": cid,
            "kind": "align_missing",
            "failed_csv_indexes": ",".join(
                str(i + 1) for i, mm in enumerate(maps) if cid not in mm
            ),
            "failed_csv_names": "; ".join(absent),
            "reason": f"该 case_id 仅在 {len(present)}/3 份 csv 中出现",
            "upstream_status": "",
            "upstream_error": "",
        })

    for cid in common:
        rec_per_csv = [m[cid] for m in maps]
        raws = [r["results"] for r in rec_per_csv]
        parsed = [parse_json_safe(r) for r in raws]

        # 收集每份 csv 的失败详情（区分"results 列空"vs"JSON 解析不出 dict"）
        per_csv_fail = []
        for i, (raw, d) in enumerate(zip(raws, parsed)):
            if isinstance(d, dict):
                continue
            if not raw or not str(raw).strip():
                fail_reason = "results 列为空字符串"
            else:
                fail_reason = (
                    f"JSON 解析失败 (got {type(d).__name__}, "
                    f"raw_head={str(raw)[:60]!r})"
                )
            per_csv_fail.append((i, csv_files[i].name, rec_per_csv[i], fail_reason))

        if per_csv_fail:
            stats["parse_fail"] += 1
            # 把上游 status/error 拼出来，方便定位真正根因
            ups_status = ", ".join(
                rec.get("status", "") for _, _, rec, _ in per_csv_fail
                if rec.get("status")
            )
            ups_error_raw = " || ".join(
                rec.get("error", "").replace("\n", " | ")
                for _, _, rec, _ in per_csv_fail
                if rec.get("error")
            )
            ups_error = ups_error_raw[:600]
            failure_records.append({
                "case_id": cid,
                "kind": "parse_fail",
                "failed_csv_indexes": ",".join(str(i + 1) for i, _, _, _ in per_csv_fail),
                "failed_csv_names": "; ".join(name for _, name, _, _ in per_csv_fail),
                "reason": "; ".join(reason for _, _, _, reason in per_csv_fail),
                "upstream_status": ups_status,
                "upstream_error": ups_error,
            })
            continue

        wrapped = []
        for d in parsed:
            if "results" in d and isinstance(d.get("results"), dict):
                wrapped.append(d)
            else:
                wrapped.append({"case_id": cid, "results": d})

        try:
            bares = [_unwrap_ar_details(d) for d in wrapped]
            merged_bare, _notes = merge_three(*bares)
            merged_full = _wrap_into_eval_result(
                wrapped[0], merged_bare,
                full_d2=wrapped[1] if len(wrapped) > 1 else None,
                full_d3=wrapped[2] if len(wrapped) > 2 else None,
            )
        except Exception as exc:
            stats["wrap_fail"] += 1
            print(f"[warn] {cid[:8]}... merge 失败: {exc}")
            failure_records.append({
                "case_id": cid,
                "kind": "merge_exception",
                "failed_csv_indexes": "",
                "failed_csv_names": "",
                "reason": f"{type(exc).__name__}: {exc}"[:600],
                "upstream_status": "",
                "upstream_error": "",
            })
            continue

        inner = merged_full.get("results") if isinstance(merged_full, dict) else None
        if not isinstance(inner, dict):
            inner = merged_full
        results_str = _format_json_str(inner, compact=compact)

        if check:
            src_strs: list[str] = []
            for d in parsed:
                src_inner = (
                    d.get("results") if "results" in d and isinstance(d.get("results"), dict)
                    else d
                )
                src_strs.append(_format_json_str(src_inner, compact=compact))
            rows_out.append([*src_strs, cid, results_str])
        else:
            rows_out.append([cid, results_str])
        stats["merged"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if check:
            writer.writerow(
                ["model1_results", "model2_results", "model3_results",
                 "case_id", "results"]
            )
        else:
            writer.writerow(["case_id", "results"])
        for row in rows_out:
            writer.writerow(row)

    # 写失败明细 csv（每行 1 个 case，含 kind / 失败 csv / 上游错误）
    fail_csv_path = None
    if failure_records:
        fail_csv_path = out_path.with_name(out_path.stem + "_failures.csv")
        with open(fail_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([
                "case_id", "kind",
                "failed_csv_indexes", "failed_csv_names",
                "reason", "upstream_status", "upstream_error",
            ])
            for r in failure_records:
                writer.writerow([
                    r["case_id"], r["kind"],
                    r["failed_csv_indexes"], r["failed_csv_names"],
                    r["reason"], r["upstream_status"], r["upstream_error"],
                ])

    print(f"[ok] 写出 → {out_path}")
    print(f"     合并成功:        {stats['merged']}")
    print(f"     JSON 解析失败:   {stats['parse_fail']}")
    print(f"     merge 异常:      {stats['wrap_fail']}")
    print(f"     未对齐丢弃:      {stats['missing_align']}")
    if fail_csv_path:
        print(f"     失败明细 csv →   {fail_csv_path}")
    stats["_failure_records"] = failure_records  # 给批量层做跨子目录一致性分析用
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--folder", help="包含 3 份 csv 的单个文件夹（单跑模式）")
    grp.add_argument("--parent-folder",
                     help="父目录，遍历其下所有子目录（每个子目录跑一遍合并）")
    ap.add_argument(
        "--out", default=None,
        help="单跑模式下指定输出 csv 路径（完整文件路径）；"
             "批量模式下被忽略，请改用 --out-dir",
    )
    ap.add_argument(
        "--out-dir", default=None,
        help="批量模式下指定输出**目录**；每个子目录对应 "
             "<out-dir>/<子目录名>.csv（开 --check 则 _check.csv）。"
             "默认 = parent-folder 自身。单跑模式下被忽略。",
    )
    ap.add_argument(
        "--compact", action="store_true",
        help="results 列输出紧凑 JSON（默认 indent=2 + 换行）",
    )
    ap.add_argument(
        "--check", action="store_true",
        help="对比模式：在 case_id/results 前再插 3 列，"
             "依次是 3 份输入 csv 的原始 results；"
             "输出文件名加 _check 后缀（<folder>_check.csv）",
    )
    args = ap.parse_args()

    if args.folder:
        folder = Path(args.folder).resolve()
        if not folder.is_dir():
            raise SystemExit(f"不是文件夹: {folder}")
        if args.out_dir:
            print("[warn] 单跑模式下 --out-dir 被忽略；输出文件路径请用 --out")
        out_path = Path(args.out).resolve() if args.out else None
        stats = process_folder(folder, out_path,
                               compact=args.compact, check=args.check)
        if stats["skipped_folder"]:
            raise SystemExit(1)
        return

    # 批量模式：parent-folder 下所有子目录跑一遍
    parent = Path(args.parent_folder).resolve()
    if not parent.is_dir():
        raise SystemExit(f"不是文件夹: {parent}")
    if args.out:
        print("[warn] 批量模式下 --out 被忽略；输出目录请用 --out-dir")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.resolve() == parent.resolve():
        print(f"[info] 输出目录: {out_dir}  (= parent，可用 --out-dir 改写)")
    else:
        print(f"[info] 输出目录: {out_dir}")

    subdirs = sorted([p for p in parent.iterdir() if p.is_dir()])
    if not subdirs:
        raise SystemExit(f"父目录下没有子文件夹: {parent}")
    print(f"[info] parent: {parent}")
    print(f"[info] 子目录数: {len(subdirs)}")

    total = {"merged": 0, "parse_fail": 0, "wrap_fail": 0,
             "skipped_folder": 0, "missing_align": 0}
    n_done = 0
    skipped_detail: list[tuple[str, list[str]]] = []
    error_detail: list[tuple[str, str]] = []
    # 跨子目录的失败收集：subfolder_name -> list[failure_record]
    per_folder_failures: dict[str, list[dict]] = {}
    suffix = "_check" if args.check else ""
    for sub in subdirs:
        out_path = out_dir / f"{sub.name}{suffix}.csv"
        try:
            st = process_folder(sub, out_path,
                                compact=args.compact, check=args.check)
        except Exception as exc:
            print(f"[err] {sub.name}: {exc}")
            error_detail.append((sub.name, str(exc)))
            continue
        for k in total:
            total[k] += st.get(k, 0)
        if not st["skipped_folder"]:
            n_done += 1
            recs = st.get("_failure_records", [])
            if recs:
                per_folder_failures[sub.name] = recs
        else:
            skipped_detail.append(
                (st.get("_skipped_name", sub.name), st.get("_skipped_csvs", []))
            )

    print(f"\n========= 批量汇总 =========")
    print(f"父目录:        {parent}")
    print(f"子目录数:      {len(subdirs)}")
    print(f"成功跑完:      {n_done}")
    print(f"跳过(csv ≠3):  {total['skipped_folder']}")
    print(f"合并 case 总数:{total['merged']}")
    print(f"JSON 解析失败: {total['parse_fail']}")
    print(f"merge 异常:    {total['wrap_fail']}")
    print(f"未对齐丢弃:    {total['missing_align']}")
    if skipped_detail:
        print(f"\n--- 被跳过的子目录明细 ({len(skipped_detail)}) ---")
        for name, files in skipped_detail:
            print(f"  ✗ {name}/  csv 数={len(files)}")
            if files:
                for fn in files:
                    print(f"      - {fn}")
    if error_detail:
        print(f"\n--- 抛异常的子目录 ({len(error_detail)}) ---")
        for name, msg in error_detail:
            print(f"  ✗ {name}/  {msg}")

    # ── 跨子目录失败一致性分析 + 落 _failures_summary.csv ────────────
    if per_folder_failures:
        # case_id -> {folder: failure_record}
        cid_to_folders: dict[str, dict[str, dict]] = {}
        for fname, recs in per_folder_failures.items():
            for r in recs:
                cid_to_folders.setdefault(r["case_id"], {})[fname] = r

        n_total_unique = len(cid_to_folders)
        n_processed_subdirs = len(per_folder_failures)
        n_consistent = sum(
            1 for fmap in cid_to_folders.values()
            if len(fmap) == n_processed_subdirs
        )
        n_partial = n_total_unique - n_consistent

        # kind 分布（取每个 case 出现过的 kind 集合的并集）
        from collections import Counter
        kind_counter: Counter = Counter()
        for fmap in cid_to_folders.values():
            kinds = {r["kind"] for r in fmap.values()}
            for k in kinds:
                kind_counter[k] += 1

        print(f"\n--- 失败 case 跨子目录一致性分析 ---")
        print(f"  unique 失败 case 数:                 {n_total_unique}")
        print(
            f"  在所有 {n_processed_subdirs} 个子目录都失败的 case:  "
            f"{n_consistent}  "
            f"({'100% 一致 → 上游数据问题' if n_consistent == n_total_unique and n_total_unique else '部分跨模型一致'})"
        )
        print(f"  仅部分子目录失败的 case:              {n_partial}")
        if kind_counter:
            print(f"  失败类型分布:")
            for kind, n in kind_counter.most_common():
                print(f"    [{kind:<16}] {n}")

        summary_csv = out_dir / "_failures_summary.csv"
        folder_names_sorted = sorted(per_folder_failures.keys())
        with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([
                "case_id",
                "n_folders_failed",
                "consistent_across_all",  # 是否在所有处理过的子目录都失败
                "failed_in_folders",
                "kinds",
                "sample_reason",
                "sample_upstream_status",
                "sample_upstream_error",
            ])
            for cid in sorted(cid_to_folders.keys()):
                fmap = cid_to_folders[cid]
                n_fail = len(fmap)
                # 取一条作样本（优先取有 upstream_error 的那条）
                sample = next(
                    (r for r in fmap.values() if r.get("upstream_error")),
                    next(iter(fmap.values())),
                )
                writer.writerow([
                    cid,
                    n_fail,
                    "yes" if n_fail == n_processed_subdirs else "no",
                    ",".join(sorted(fmap.keys())),
                    ",".join(sorted({r["kind"] for r in fmap.values()})),
                    sample.get("reason", "")[:300],
                    sample.get("upstream_status", ""),
                    sample.get("upstream_error", "")[:300],
                ])
        print(f"\n[failures-summary] {summary_csv}")


if __name__ == "__main__":
    main()
