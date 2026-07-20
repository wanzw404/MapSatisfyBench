"""Tool 响应仿造模拟器（strict 沙箱模式专用）。

触发场景：``BaseSimulationAgent(sandbox=True)`` → 工具调用 →
``@sandbox_cache`` 装饰器进入 strict 路径 → ``run_sandbox()`` →
mock_data/<tool>.csv 子集匹配未命中 → 本模块调 LLM 仿造一条 result 返回。

设计要点：
  - **绝不写回沙箱**——只有真实业务调用的响应才写 mock_data（由
    ``@sandbox_cache`` cache-aside 路径负责）；沙箱命中 / LLM 仿造的
    响应都仅返回给调用方
  - LLM 同步调（``provider.chat()`` 而非 achat）—— tool 体跑在
    ``run_in_executor`` 线程里，sync 最简单
  - Prompt 模板从 ``app/core/agent/prompt/tool_simulator_prompt.yaml`` 加载
  - 历史样例 JSON-aware 压缩：长字符串截短、列表截短，**保证 JSON 结构完整**
  - 失败（无 sample 也无 desc / LLM 非 JSON / 解析异常）→ 返 None，让上层
    ``run_sandbox`` 退回空值映射兜底
"""
from __future__ import annotations

import csv
import json
import logging
import re
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import yaml

from app.config import TOOL_SIMULATOR_MODEL, settings
from app.core.agent.llm.base import BaseLLMProvider
from app.core.agent.llm.openai_chat import OpenAICompatProvider
from app.paths import EXACT_SEARCH_DIR

# 沙箱 csv 长 result 可能 > 128KB
csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)


_PROMPT_FILE = (
    Path(__file__).resolve().parent.parent
    / "agent"
    / "prompt"
    / "tool_simulator_prompt.yaml"
)


# ---------------------------------------------------------------------------
# JSON-aware 压缩：保证 sample.result 序列化后体积可控，同时不破坏 JSON 结构
# ---------------------------------------------------------------------------


def _shrink(obj: Any, max_str_len: int, max_list_items: int) -> Any:
    """递归压缩：长 str → 截短 + 占位说明；list → 取前 N 项 + 占位说明；
    dict 递归。**不改字段名、不改类型、不丢嵌套层级**。"""
    if isinstance(obj, str):
        if len(obj) > max_str_len:
            return obj[:max_str_len] + f"...({len(obj) - max_str_len} chars omitted)"
        return obj
    if isinstance(obj, list):
        if len(obj) <= max_list_items:
            return [_shrink(x, max_str_len, max_list_items) for x in obj]
        head = [_shrink(x, max_str_len, max_list_items) for x in obj[:max_list_items]]
        head.append(f"<truncated {len(obj) - max_list_items} items>")
        return head
    if isinstance(obj, dict):
        return {k: _shrink(v, max_str_len, max_list_items) for k, v in obj.items()}
    return obj


def shrink_for_prompt(obj: Any, max_bytes: int = 8 * 1024) -> Any:
    """收敛式压缩：从宽到紧调几档参数，直到 ``json.dumps(obj)`` 字节数 ≤ max_bytes。

    宽 → 紧 档位（max_str_len, max_list_items）：
        (200, 5) → (100, 3) → (50, 2) → (30, 1)
    最紧档仍超：返回最紧档的结果（让 prompt 稍微超一点也比裸送强）。
    """
    levels = [(200, 5), (100, 3), (50, 2), (30, 1)]
    for ms, ml in levels:
        candidate = _shrink(obj, ms, ml)
        size = len(json.dumps(candidate, ensure_ascii=False).encode("utf-8"))
        if size <= max_bytes:
            return candidate
    return _shrink(obj, *levels[-1])


# ---------------------------------------------------------------------------
# 取工具描述（来自 LangChain @tool 装饰器 .description 属性）
# ---------------------------------------------------------------------------


_TOOLS_BY_NAME: dict[str, Any] | None = None
_TOOLS_BY_NAME_LOCK = threading.Lock()


def _get_tools_by_name() -> dict[str, Any]:
    """懒加载并缓存 manager.tools → name dict（避免每次仿造都重 import）。"""
    global _TOOLS_BY_NAME
    if _TOOLS_BY_NAME is not None:
        return _TOOLS_BY_NAME
    with _TOOLS_BY_NAME_LOCK:
        if _TOOLS_BY_NAME is not None:
            return _TOOLS_BY_NAME
        try:
            from app.core.tools.manager import tools as _all_tools
            _TOOLS_BY_NAME = {t.name: t for t in _all_tools}
        except Exception as e:
            logger.warning("[ToolSimulator] 加载 tools.manager 失败: %s", e)
            _TOOLS_BY_NAME = {}
    return _TOOLS_BY_NAME


def get_tool_description(tool_name: str) -> str:
    t = _get_tools_by_name().get(tool_name)
    return (getattr(t, "description", "") or "").strip()


# ---------------------------------------------------------------------------
# Sample 加载：取 mock_data/<tool>.csv 首条可解析的 (args, result)
# ---------------------------------------------------------------------------


def load_first_sample(tool_name: str) -> Optional[tuple[dict, Any]]:
    """从 mock_data/<tool>.csv 找首条 arguments / result 都可解析的记录。

    文件不存在 / 全部行都解析失败 → 返 None。
    """
    csv_path = EXACT_SEARCH_DIR / f"{tool_name}.csv"
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames \
                    or "arguments" not in reader.fieldnames \
                    or "result" not in reader.fieldnames:
                return None
            for row in reader:
                raw_args = row.get("arguments")
                raw_result = row.get("result")
                if not raw_args or raw_result is None or not str(raw_result).strip():
                    continue
                try:
                    args = json.loads(str(raw_args))
                    result = json.loads(str(raw_result))
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(args, dict):
                    return args, result
    except Exception as e:
        logger.warning("[ToolSimulator] 加载样例失败 (%s): %s", tool_name, e)
    return None


# ---------------------------------------------------------------------------
# Prompt 模板加载 + 渲染
# ---------------------------------------------------------------------------


_PROMPT_PLACEHOLDERS = (
    "tool_name",
    "tool_description",
    "sample_arguments_json",
    "sample_result_json",
    "request_args_json",
)
_PLACEHOLDER_RE = re.compile(r"\{(" + "|".join(_PROMPT_PLACEHOLDERS) + r")\}")


def _load_prompt_template(language: str = "chinese") -> str:
    if not _PROMPT_FILE.exists():
        raise FileNotFoundError(f"Tool simulator prompt 文件不存在: {_PROMPT_FILE}")
    with open(_PROMPT_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    template = data.get(language)
    if not isinstance(template, str):
        raise ValueError(
            f"Tool simulator prompt 文件缺少 language='{language}' 块"
        )
    return template


def _render_prompt(template: str, context: dict[str, str]) -> str:
    """只替换 _PROMPT_PLACEHOLDERS 里列出的 key，不动 JSON 里的 {}。"""
    def _sub(m: re.Match) -> str:
        return str(context.get(m.group(1), ""))
    return _PLACEHOLDER_RE.sub(_sub, template)


# ---------------------------------------------------------------------------
# 响应清洗：剥 markdown fence、定位首尾 {...}
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_to_json(raw: str) -> str:
    text = (raw or "").strip()
    text = _FENCE_RE.sub("", text).strip()
    if not text.startswith("{") and not text.startswith("["):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    return text


# ---------------------------------------------------------------------------
# ToolSimulator
# ---------------------------------------------------------------------------


class ToolSimulator:
    """LLM 仿造 tool 响应。线程安全（chat() 调用本身 thread-safe）。"""

    # sample.result 压缩阈值；过大的样例会让 prompt 暴涨
    _MAX_SAMPLE_BYTES = 8 * 1024

    def __init__(self, llm_provider: BaseLLMProvider, language: str = "chinese"):
        self.llm = llm_provider
        self.language = language
        self._template = _load_prompt_template(language)

    def simulate(
        self,
        tool_name: str,
        tool_description: str,
        request_args: dict,
    ) -> Optional[dict]:
        """仿造一条工具响应。

        Returns:
            仿造的 dict 响应；任何失败（LLM 异常 / 解析失败 / 非 JSON）返 None
        """
        sample = load_first_sample(tool_name)
        if sample is None:
            sample_args_json = "(无历史样例可参考)"
            sample_result_json = "(无历史样例可参考)"
        else:
            args, result = sample
            compact_result = shrink_for_prompt(
                result, max_bytes=self._MAX_SAMPLE_BYTES
            )
            sample_args_json = json.dumps(args, ensure_ascii=False, indent=2)
            sample_result_json = json.dumps(
                compact_result, ensure_ascii=False, indent=2
            )

        clean_req = {k: v for k, v in request_args.items() if k != "sandbox"}

        rendered = _render_prompt(
            self._template,
            {
                "tool_name": tool_name,
                "tool_description": tool_description or "(no description)",
                "sample_arguments_json": sample_args_json,
                "sample_result_json": sample_result_json,
                "request_args_json": json.dumps(
                    clean_req, ensure_ascii=False, indent=2
                ),
            },
        )

        messages = [
            {"role": "system", "content": rendered},
            {"role": "user", "content": "请按要求输出 JSON。"},
        ]

        try:
            resp = self.llm.chat(messages)
        except Exception as e:
            logger.warning("[ToolSimulator] %s LLM 调用失败: %s", tool_name, e)
            return None

        raw = (resp.content or "").strip()
        if not raw:
            logger.warning("[ToolSimulator] %s LLM 返空", tool_name)
            return None

        text = _strip_to_json(raw)
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                "[ToolSimulator] %s LLM 响应非合法 JSON: %s; head=%r",
                tool_name, e, text[:200],
            )
            return None

        if not isinstance(parsed, (dict, list)):
            logger.warning(
                "[ToolSimulator] %s LLM 响应顶层非 dict/list: type=%s",
                tool_name, type(parsed).__name__,
            )
            return None

        logger.info(
            "[ToolSimulator] %s 已仿造响应（sample=%s, request_keys=%s）",
            tool_name,
            "yes" if sample is not None else "no",
            list(clean_req.keys()),
        )
        return parsed  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------

_SINGLETON: Optional[ToolSimulator] = None
_SINGLETON_LOCK = threading.Lock()


def get_tool_simulator() -> ToolSimulator:
    """懒加载 + 线程安全 的 ToolSimulator 单例。"""
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is not None:
            return _SINGLETON
        provider = OpenAICompatProvider(
            base_url=settings.BASE_URL,
            api_key=settings.AI_STUDIO_TOKEN or settings.API_KEY,
            model=TOOL_SIMULATOR_MODEL,
        )
        _SINGLETON = ToolSimulator(llm_provider=provider)
    return _SINGLETON


def simulate_tool_response(
    tool_name: str,
    request_args: dict,
) -> Optional[dict]:
    """run_sandbox / 测试入口：拿单例 ToolSimulator 仿造一条响应。

    Returns:
        仿造的 dict；失败返 None（让调用方退回空值映射等兜底）
    """
    tool_desc = get_tool_description(tool_name)
    try:
        sim = get_tool_simulator()
    except Exception as e:
        logger.warning("[ToolSimulator] 构造单例失败: %s", e)
        return None
    return sim.simulate(tool_name, tool_desc, request_args)
