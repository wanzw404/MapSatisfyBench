"""工具执行沙箱逻辑（CSV 版本）。

两套 API：

1. 旧入口 ``run_sandbox(tool_name, params) -> str``：strict 模式专用。
   - 命中 mock：返回 result 字符串原样
   - 未命中：按首行 result 的 schema 派生「空值映射」JSON 字符串
   - 用于 ``sandbox=True`` 时强制走 mock、不允许真调

2. 新入口 ``sandbox_cache_lookup(tool_name, params) -> dict | None``：
   cache-aside 模式（``@sandbox_cache`` 装饰器用）。
   - 命中：返回 parsed dict（不再回 str，与真调返回形态一致）
   - 未命中：返回 ``None``（**不走空值兜底**，让装饰器去调真实 API）
   - 文件不存在视同未命中（不抛），允许新 tool 首次调用通过

匹配规则保持不变：``mock_args ⊆ real_params``（mock 录入的每个 key 都必须在
真实入参里同名存在且 value 相等；real_params 多出的字段 / sandbox / 内部字段
忽略）。

数据格式：CSV，utf-8-sig，4 列 ``trace_id, tool_name, arguments, result``。
"""

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from app.paths import EXACT_SEARCH_DIR

logger = logging.getLogger(__name__)

# tool 返回的 result 可能很长（高德 around POI 整页返回轻松 > 128KB），
# 解锁 csv 单字段大小限制，由 OS 文件大小约束
csv.field_size_limit(sys.maxsize)


# 子集匹配时跳过的系统字段：这些字段是工具内部参数，不参与语义对比
SYSTEM_KEYS_TO_SKIP: set[str] = {"sandbox"}

_COORDINATE_KEYS: frozenset[str] = frozenset({
    "x", "y",
    "start_x", "start_y", "end_x", "end_y",
    "start_lon", "start_lat", "end_lon", "end_lat",
})


# ---------------------------------------------------------------------------
# 共用辅助
# ---------------------------------------------------------------------------


def _csv_path(tool_name: str) -> Path:
    return EXACT_SEARCH_DIR / f"{tool_name}.csv"


def _is_empty_value(v: Any) -> bool:
    return v is None or v == ""


def _coords_equal(a: Any, b: Any, precision: int = 5) -> bool:
    try:
        fa = float(str(a))
        fb = float(str(b))
        factor = 10 ** precision
        return int(fa * factor) == int(fb * factor)
    except (ValueError, TypeError):
        return False


def _values_equal(a: Any, b: Any) -> bool:
    """保守 value 相等：
    - None 仅与 None 等；其它情况类型严格匹配
    - 字符串 strip() 头尾后比较
    - dict 递归（key 集合必须严格相等）
    - list 递归（长度 + 同位置）
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if type(a) is not type(b):
        return False
    if isinstance(a, str):
        return a.strip() == b.strip()
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b))
    return a == b


def _is_subset_match(mock_args: dict, real_params: dict) -> bool:
    """mock_args 的每个 key（除系统字段和空值）都在 real_params 里且 value 相等。
    经纬度字段截断到小数点后 5 位比较。"""
    for k, v in mock_args.items():
        if k in SYSTEM_KEYS_TO_SKIP:
            continue
        if _is_empty_value(v):
            continue
        if k not in real_params:
            return False
        if k in _COORDINATE_KEYS:
            if not _coords_equal(v, real_params[k]):
                return False
        else:
            if not _values_equal(v, real_params[k]):
                return False
    return True


# ---------------------------------------------------------------------------
# 核心查找：返回命中行的 result 字符串（未命中 None）
# ---------------------------------------------------------------------------


def _structured_exact_match(tool_name: str, real_params: dict) -> Optional[str]:
    """按 csv 行做 mock_args ⊆ real_params 子集匹配。

    Returns:
        命中：result 列字符串
        未命中：None
        文件不存在：None（调用方按未命中处理；strict 模式上层会另作错误处理）
    """
    csv_file = _csv_path(tool_name)
    if not csv_file.exists():
        return None

    with open(csv_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "arguments" not in reader.fieldnames \
                or "result" not in reader.fieldnames:
            raise ValueError(
                f"沙箱文件 {csv_file} 缺少必要列 arguments / result"
            )
        for row_no, row in enumerate(reader, start=2):
            row_args = row.get("arguments")
            row_result = row.get("result")
            if not row_args or row_result is None:
                continue
            try:
                mock_args = json.loads(str(row_args))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"[{tool_name}] csv 第 {row_no} 行 arguments 非 JSON，跳过: {e}"
                )
                continue
            if not isinstance(mock_args, dict):
                logger.warning(
                    f"[{tool_name}] csv 第 {row_no} 行 arguments 非 dict，跳过"
                )
                continue
            if _is_subset_match(mock_args, real_params):
                logger.info(
                    f"[{tool_name}] csv 第 {row_no} 行命中 "
                    f"(mock_keys={list(mock_args.keys())})"
                )
                return str(row_result)
    return None


# ---------------------------------------------------------------------------
# 向量相似度匹配（精确匹配未命中后的 fallback）
# ---------------------------------------------------------------------------


_VECTOR_SIMILARITY_THRESHOLD: float = 0.85


def _vector_similarity_match(
    tool_name: str, real_params: dict, *, threshold: float = _VECTOR_SIMILARITY_THRESHOLD
) -> Optional[str]:
    """精确匹配未命中后的向量相似度 fallback。

    仅在同一 trace_id 下的记录中做 embedding cosine 匹配，
    取最高分且 >= threshold 的行返回其 result。
    """
    from app.core.tools.base import get_tool_trace_id
    from app.core.agent.embedding import EmbeddingModelProvider, _cosine_similarity

    trace_id = get_tool_trace_id()
    if not trace_id:
        return None

    csv_file = _csv_path(tool_name)
    if not csv_file.exists():
        return None

    candidates: list[tuple[str, dict]] = []
    with open(csv_file, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "feature" not in reader.fieldnames:
            return None
        for row in reader:
            if row.get("trace_id", "").strip() != trace_id:
                continue
            feat_raw = (row.get("feature") or "").strip()
            result = row.get("result")
            if not feat_raw or result is None:
                continue
            try:
                feat = json.loads(feat_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(feat, dict) and feat:
                candidates.append((str(result), feat))

    if not candidates:
        return None

    feature_keys = list(candidates[0][1].keys())
    texts_to_embed: dict[str, str] = {}
    for k in feature_keys:
        v = real_params.get(k)
        if v is not None and str(v).strip():
            texts_to_embed[k] = str(v).strip()

    if not texts_to_embed:
        return None

    try:
        provider = EmbeddingModelProvider()
        request_vectors: dict[str, list[float]] = {}
        for k, text in texts_to_embed.items():
            request_vectors[k] = provider.text_embedding(text)
    except Exception as e:
        logger.warning("[%s] 向量匹配 embedding 调用失败: %s", tool_name, e)
        return None

    best_score = -1.0
    best_result: Optional[str] = None
    for result_str, feat_dict in candidates:
        scores = []
        for k, req_vec in request_vectors.items():
            mock_vec = feat_dict.get(k)
            if not isinstance(mock_vec, list) or not mock_vec:
                continue
            scores.append(_cosine_similarity(req_vec, mock_vec))
        if not scores:
            continue
        avg_score = sum(scores) / len(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_result = result_str

    if best_score >= threshold and best_result is not None:
        logger.info(
            "[%s] 向量匹配命中 (trace=%s, score=%.4f, threshold=%.2f)",
            tool_name, trace_id, best_score, threshold,
        )
        return best_result

    logger.debug(
        "[%s] 向量匹配未命中 (trace=%s, best=%.4f, threshold=%.2f)",
        tool_name, trace_id, best_score, threshold,
    )
    return None


# ---------------------------------------------------------------------------
# 空值映射（strict 模式未命中兜底）
# ---------------------------------------------------------------------------


def _empty_value_for(v: Any) -> Any:
    """递归把任意 leaf 替换为类型默认空值。"""
    if isinstance(v, dict):
        return {k: _empty_value_for(val) for k, val in v.items()}
    if isinstance(v, list):
        return []
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return 0
    if isinstance(v, str):
        return ""
    return None


def _build_empty_result(tool_name: str) -> str:
    """按首行 result 的 schema 镜像 + 清空 leaf。文件不存在 / 无可解析行 → ``"无mock数据"``。"""
    csv_file = _csv_path(tool_name)
    fallback = "无mock数据"
    if not csv_file.exists():
        return fallback
    try:
        with open(csv_file, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "result" not in reader.fieldnames:
                return fallback
            for row in reader:
                cell = row.get("result")
                if cell is None or not str(cell).strip():
                    continue
                try:
                    schema = json.loads(str(cell))
                except json.JSONDecodeError:
                    continue
                return json.dumps(_empty_value_for(schema), ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[{tool_name}] 构造空值映射失败，退回固定字符串: {e}")
    return fallback


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def run_sandbox(tool_name: str, params: dict) -> str:
    """**Strict 模式专用**：严格走 mock 数据，不允许真调。

    优先级：
      1. mock_data/<tool>.csv 子集匹配命中 → 原样返 result 字符串
      2. 未命中 → 调 ``tool_simulator.simulate_tool_response`` 让 LLM
         参考工具描述 + 沙箱首条样例仿造一条响应（**不写回沙箱**）
      3. tool_simulator 失败 → 退回按首行 result schema 派生的空值映射 JSON
      4. 沙箱文件不存在：仍尝试 tool_simulator（凭工具描述裸生成）；
         裸生成也失败时 fallback 到固定字符串 ``"无mock数据"``
    """
    csv_file = _csv_path(tool_name)
    real_params = {
        k: v for k, v in params.items() if k not in SYSTEM_KEYS_TO_SKIP
    }

    # 1) 子集匹配（文件存在时）
    if csv_file.exists():
        try:
            hit = _structured_exact_match(tool_name, real_params)
            if hit is not None:
                return hit
        except Exception as e:
            logger.error(f"[{tool_name}] 精确匹配阶段异常: {e}")

    # 2) 向量相似度匹配
    try:
        hit = _vector_similarity_match(tool_name, real_params)
        if hit is not None:
            return hit
    except Exception as e:
        logger.warning("[%s] 向量匹配阶段异常: %s", tool_name, e)

    # 3) tool_simulator 仿造（**不写回沙箱**）
    try:
        # 局部 import 避开模块加载期循环依赖
        from app.core.simulator.tool_simulator import simulate_tool_response
        fabricated = simulate_tool_response(tool_name, real_params)
        if fabricated is not None:
            logger.warning(
                "[%s] 沙箱 strict 模式未命中 → 已由 tool_simulator 仿造响应（不写回）",
                tool_name,
            )
            return json.dumps(fabricated, ensure_ascii=False)
    except Exception as e:
        logger.warning("[%s] tool_simulator 仿造异常: %s", tool_name, e)

    # 3) 空值映射兜底
    if csv_file.exists():
        logger.warning(
            f"[{tool_name}] tool_simulator 失败，退回空值映射"
        )
        return _build_empty_result(tool_name)
    logger.warning(
        f"[{tool_name}] 沙箱文件不存在且 tool_simulator 失败，退回 '无mock数据'"
    )
    return "无mock数据"


def sandbox_cache_lookup(tool_name: str, params: dict) -> Optional[dict]:
    """**Cache-aside 模式专用**：命中返 parsed dict，未命中 / 文件不存在返 None。

    与 ``run_sandbox`` 的区别：
      - 命中后**反序列化为 dict** 返回（让装饰器与真调返回形态一致）
      - 未命中**不走空值兜底**，让装饰器去调真实 API
      - 沙箱文件不存在**不抛**（视同未命中，允许新 tool 首次调用走真调 + 创建文件）

    Returns:
        命中：result JSON 解析后的 dict（解析失败时返回原始字符串供调用方降级处理）
        未命中：None
    """
    real_params = {
        k: v for k, v in params.items() if k not in SYSTEM_KEYS_TO_SKIP
    }
    try:
        hit = _structured_exact_match(tool_name, real_params)
    except Exception as e:
        logger.error(f"[{tool_name}] sandbox_cache_lookup 异常: {e}")
        return None

    if hit is None:
        try:
            hit = _vector_similarity_match(tool_name, real_params)
        except Exception as e:
            logger.warning("[%s] sandbox_cache_lookup 向量匹配异常: %s", tool_name, e)

    if hit is None:
        return None
    try:
        parsed = json.loads(hit)
        return parsed if isinstance(parsed, (dict, list)) else hit
    except (json.JSONDecodeError, TypeError):
        # result 不是 JSON（极少见，老数据可能），原样返
        return hit  # type: ignore[return-value]
