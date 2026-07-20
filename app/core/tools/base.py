import asyncio
import contextvars
import functools
import inspect
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from app.paths import EXACT_SEARCH_DIR, NEW_DATA_DIR

# 沙箱回写解锁 csv 单字段大小限制（tool result 可能 > 128KB）
import csv as _csv_module
import sys as _sys
_csv_module.field_size_limit(_sys.maxsize)

logger = logging.getLogger(__name__)


# =============================================================================
# 通用异常兜底装饰器
# =============================================================================

def safe_tool(fn: Callable) -> Callable:
    """工具函数级异常兜底装饰器。

    把工具体内任何未捕获的异常（HTTP 底层错、KeyError、TypeError、第三方库
    崩等）统一转成结构化响应 dict，**永不向外抛**：

        {
            "status": "0",
            "info": "<ExcType>: <message>",
            "exception_type": "<ExcType>",
            "request_args": {<入参快照，已去 sandbox>},
        }

    与 ``@tool`` 配合时**必须放在 @tool 内层**（即更靠近原函数）::

        @tool(parse_docstring=True)
        @safe_tool
        def search_poi(...): ...

    注意：pydantic ValidationError 由 LangChain ``_parse_input`` 在工具体
    **之外** 抛出，本装饰器接不到——那条路径由
    ``agent_simulator._execute_tool_call`` 的 except 块兜底。两者互补。

    sync / async 两种函数都支持。
    """

    def _snapshot_args(args: tuple, kwargs: dict) -> dict[str, Any]:
        """位置参数 + kwargs → dict（去 sandbox）。"""
        try:
            sig = inspect.signature(fn)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            req = dict(bound.arguments)
        except TypeError:
            # 极端 case：签名不匹配，至少把 kwargs 落下
            req = {"_positional": list(args), **kwargs}
        req.pop("sandbox", None)
        return req

    def _build_error_response(exc: Exception, req: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "0",
            "info": f"{type(exc).__name__}: {exc}",
            "exception_type": type(exc).__name__,
            "request_args": req,
        }

    if asyncio.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                req = _snapshot_args(args, kwargs)
                logger.warning(
                    "[safe_tool] %s 内部异常 → 结构化响应: %s: %s",
                    fn.__name__, type(e).__name__, str(e)[:200],
                )
                return _build_error_response(e, req)
        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            req = _snapshot_args(args, kwargs)
            logger.warning(
                "[safe_tool] %s 内部异常 → 结构化响应: %s: %s",
                fn.__name__, type(e).__name__, str(e)[:200],
            )
            return _build_error_response(e, req)
    return sync_wrapper


class TokenBucket:
    """令牌桶限流器."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_time = time.time()
        self.lock = threading.Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_time = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            wait_time = (1 - self.tokens) / self.rate

        if timeout is not None and wait_time > timeout:
            return False

        time.sleep(wait_time)
        return self.acquire(timeout=0)


# HTTP 请求限流：最大 QPS = 10
_http_rate_limiter = TokenBucket(rate=10, capacity=10)


def http_get(url: str, params: dict = None, headers: dict = None, timeout: float = 5, encoding: str = "utf-8") -> requests.Response:
    """带限流的 HTTP GET 请求封装.

    所有代码中的 HTTP GET 请求均应通过此方法发送，以确保统一限流与日志。
    """
    if not _http_rate_limiter.acquire(timeout=5):
        raise RuntimeError("HTTP 请求限流，获取令牌超时")

    logger.info("http_get request: url=%s, params=%s, timeout=%s", url, params, timeout)
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.encoding = encoding
    logger.info(
        "http_get response: url=%s, status_code=%s, content_length=%s",
        url, response.status_code, len(response.text)
    )
    return response


def http_post(url: str, json: dict = None, headers: dict = None, timeout: float = 5, encoding: str = "utf-8") -> requests.Response:
    """带限流的 HTTP POST 请求封装.

    所有代码中的 HTTP POST 请求均应通过此方法发送，以确保统一限流与日志。
    """
    if not _http_rate_limiter.acquire(timeout=5):
        raise RuntimeError("HTTP 请求限流，获取令牌超时")

    # logger.info("http_post request: url=%s, json=%s, timeout=%s", url, json, timeout)
    response = requests.post(url, json=json, headers=headers, timeout=timeout)
    response.encoding = encoding
    # logger.info(
    #     "http_post response: url=%s, status_code=%s, content_length=%s",
    #     url, response.status_code, len(response.text)
    # )
    return response


def _safe_get_text(element, tag_name: str) -> Optional[str]:
    """安全获取 XML 元素的文本内容."""
    if element is None:
        return None
    child = element.find(tag_name)
    return child.text if child is not None else None


def _safe_get_float(element, tag_name: str) -> Optional[float]:
    """安全获取 XML 元素的浮点数内容."""
    text = _safe_get_text(element, tag_name)
    try:
        return float(text) if text else None
    except (ValueError, TypeError):
        return None


def _do_vipserver_get_request(domain: str, path: str, param: dict, head: dict, timeout: int, encoding: str) -> str:
    """模拟 doVipserverGetRequest：通过 vipserver 获取 host 后发起 HTTP GET 请求."""
    try:
        from vipserver.vip_client import get_one_validate_host
    except ImportError as exc:
        logger.error("vipserver 模块不可用: %s", exc)
        raise RuntimeError(f"vipserver 模块不可用: {exc}") from exc

    host = get_one_validate_host(domain)
    url = f"http://{host.ip}:{host.port}{path}"
    response = http_get(url, params=param, headers=head, timeout=timeout / 1000, encoding=encoding)
    return response.text


def _do_vipserver_post_request(domain: str, path: str, body: dict, head: dict, timeout: int, encoding: str) -> str:
    """模拟 doVipserverPostRequest：通过 vipserver 获取 host 后发起 HTTP POST 请求."""
    try:
        from vipserver.vip_client import get_one_validate_host
    except ImportError as exc:
        logger.error("vipserver 模块不可用: %s", exc)
        raise RuntimeError(f"vipserver 模块不可用: {exc}") from exc

    host = get_one_validate_host(domain)
    url = f"http://{host.ip}:{host.port}{path}"
    response = http_post(url, json=body, headers=head, timeout=timeout / 1000)
    response.encoding = encoding
    return response.text


def _is_prod() -> bool:
    """判断当前是否生产环境（优先 ENV，其次 RUNTIME_ENV）."""
    env = os.environ.get("ENV", os.environ.get("RUNTIME_ENV", "")).lower()
    return env == "prod"


# =============================================================================
# 工具调用记录逻辑（真实调用结果回写到 /data/sandbox/mock_data/{tool}.csv）
# =============================================================================

# trace_id 上下文变量（通过 agent_simulator 的 tool_node 注入）
_tool_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("tool_trace_id", default=None)


def set_tool_trace_id(trace_id: str | None) -> None:
    """设置当前工具调用的 trace_id。"""
    _tool_trace_id.set(trace_id)


def get_tool_trace_id() -> str | None:
    """获取当前工具调用的 trace_id。"""
    return _tool_trace_id.get()


# sandbox 上下文变量（通过 agent_simulator 的 tool_node 注入）
# LangChain @tool 装饰器会用 args_schema 过滤掉不在函数签名里的 kwargs，
# 导致 sandbox=True 无法通过 kwargs 传递到 sandbox_cache 装饰器层。
# 改用 contextvars 绕过 @tool 的 schema 过滤。
_tool_sandbox: contextvars.ContextVar[bool] = contextvars.ContextVar("tool_sandbox", default=False)


def set_tool_sandbox(sandbox: bool) -> None:
    _tool_sandbox.set(sandbox)


def get_tool_sandbox() -> bool:
    return _tool_sandbox.get()


def _normalize_text(text: str) -> str:
    """去除空格、回车、换行——保持 mock_data 里一行一记录、易于 grep。"""
    if text is None:
        return ""
    return str(text).replace(" ", "").replace("\n", "").replace("\r", "")


# CSV 写入线程锁（threading 而非 asyncio——agent 走 run_in_executor，
# 多个 case 的 tool 在线程池里跑，threading.Lock 保护正确）
_csv_write_lock = threading.Lock()


_SANDBOX_CSV_HEADERS = ["trace_id", "tool_name", "arguments", "result"]


def _append_to_sandbox_csv(
    tool_name: str,
    trace_id: str,
    arguments: str,
    result: str,
) -> None:
    """把单条 (args, result) 追加到 ``mock_data/<tool_name>.csv``。

    线程安全；文件不存在时自动创建 + 写表头；utf-8-sig 兼容 Excel 双击。
    用 csv 模块原生 append，无需 load 整本，O(1) 行成本。
    """
    EXACT_SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = EXACT_SEARCH_DIR / f"{tool_name}.csv"

    with _csv_write_lock:
        need_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = _csv_module.writer(f, quoting=_csv_module.QUOTE_MINIMAL)
            if need_header:
                writer.writerow(_SANDBOX_CSV_HEADERS)
            writer.writerow([trace_id, tool_name, arguments, result])


def record(tool_name: str, arguments: dict, result: Any) -> Any:
    """[DEPRECATED] 老调用方使用：记录工具真实请求并原样返回 result。

    新版工具体不直接调；改由 ``@sandbox_cache`` 装饰器统一处理回写。
    保留是为了让任何手写的旧调用方还能 work，无新写入到 ``new_data/`` 了——
    本函数已切到写 ``mock_data/<tool>.csv``，与新装饰器同目标。

    与 ``@sandbox_cache`` 同款 cacheable gate：result 是 None / 含 error /
    含 exception_type / status 命中失败值时**跳过写盘**，避免把失败响应或
    工具返回的 None 字面量（如服务端返 ``{"data":{"data":null}}``）写入沙箱
    污染后续 cache lookup。
    """
    if not _is_cacheable_result(result):
        logger.debug(
            "[%s] record() 跳过写沙箱（result 不可缓存：None/error/失败）",
            tool_name,
        )
        return result

    trace_id = get_tool_trace_id() or ""
    clean_args = {k: v for k, v in arguments.items() if k != "sandbox"}
    normalized_args = _normalize_text(
        json.dumps(clean_args, sort_keys=True, ensure_ascii=False)
    )
    if isinstance(result, (dict, list)):
        normalized_result = _normalize_text(
            json.dumps(result, ensure_ascii=False)
        )
    else:
        normalized_result = _normalize_text(str(result))

    try:
        _append_to_sandbox_csv(
            tool_name=tool_name,
            trace_id=trace_id,
            arguments=normalized_args,
            result=normalized_result,
        )
    except Exception as e:
        logger.error("[%s] 工具记录写入失败: %s", tool_name, e)

    return result


# =============================================================================
# sandbox_cache 装饰器
# =============================================================================
#
# 行为：
#   sandbox=True  → strict 模式：调旧 run_sandbox（严格 mock，不命中走空值映射，
#                   不调真实 API、不写回）
#   sandbox=False → cache-aside 模式（默认）：
#                   1) sandbox_cache_lookup(mock_data/<tool>.csv) → 命中即返
#                   2) 未命中 → 调真实工具函数
#                   3) 返回结构判定为"成功"（dict 无 exception_type、无非空 error
#                      值、status 非工具级失败值）→ 写回
#
# 与 @tool / @safe_tool 的叠加顺序（外→内）：
#   @tool                  # langchain schema 校验
#   @safe_tool             # 工具体异常 → 结构化 error dict
#   @sandbox_cache         # 本装饰器：cache-aside 或 strict
#   def fn(...):           # 原始工具体


# dict.status 失败白名单：命中即视为失败响应、不写回沙箱。
# 仅"工具/框架级失败"标识，不含上游接口的 status 码：
#   - "error"       ：search_poi_around_multipoints 等自定义返的失败码
#   - "fail" / "failed" ：其它通用约定
# 注意："0" 不在此集合：高德等接口约定 status="0" 是上游真实失败响应，沙箱应
# 缓存（保留确定性，避免下次同 args 重复打外网）。
_FAILURE_STATUS_VALUES: frozenset[str] = frozenset(
    {"error", "fail", "failed"}
)


def _is_cacheable_result(result: Any) -> bool:
    """启发式判定是否值得写回沙箱（避免缓存"工具/框架级失败"响应）。

    - dict：满足以下任一即**不缓存**：
        * 含顶层 ``exception_type`` key（safe_tool 转的内部异常包装）
        * 顶层 ``error`` 字段值**非空**（非 None 且非 ""）—— 注意是值检查
          不是 key 存在性：``{"error": None}`` / ``{"error": ""}`` 视为成功
        * ``status`` 取值（小写化后）命中 ``_FAILURE_STATUS_VALUES`` ——
          即 ``"error"/"fail"/"failed"``。``status="0"`` 不在此列表，会被缓存
    - list：直接缓存（没有 error 字段的概念）
    - str：跳过（多是工具内入参校验提示，如 "search_poi 需要提供 query"，
          缓存反而妨碍 LLM 自纠）
    - 其它（None / int / float / bool）：跳过
    """
    if isinstance(result, list):
        return True
    if not isinstance(result, dict):
        return False

    if "exception_type" in result:
        return False
    if "error" in result:
        err_val = result["error"]
        if err_val is not None and err_val != "":
            return False
    status = result.get("status")
    if status is not None:
        if str(status).strip().lower() in _FAILURE_STATUS_VALUES:
            return False
    return True


def _snapshot_args_for_cache(fn: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    """位置参数 + kwargs → dict（去 sandbox 内部字段）。"""
    try:
        sig = inspect.signature(fn)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        req = dict(bound.arguments)
    except TypeError:
        req = {"_positional": list(args), **kwargs}
    req.pop("sandbox", None)
    return req


def sandbox_cache(fn: Callable) -> Callable:
    """Cache-aside 装饰器（详见模块顶部说明）。

    与 @tool 配合时**放在 @safe_tool 内层**::

        @tool(parse_docstring=True)
        @safe_tool
        @sandbox_cache
        def search_poi(...): ...

    sync / async 双支持（业务 tool 当前全是 sync，但为以后预留 async 路径）。
    """
    from app.services.run_sandbox import (  # 局部 import 避开循环依赖
        run_sandbox,
        sandbox_cache_lookup,
    )

    tool_name = fn.__name__

    def _strict(args: tuple, kwargs: dict) -> Any:
        """sandbox=True 路径：走 run_sandbox（命中或空值映射），不调 fn。"""
        req = _snapshot_args_for_cache(fn, args, kwargs)
        # run_sandbox 返回字符串（一般是 JSON），按 dict 解析后返回与 fn 形态对齐
        raw = run_sandbox(tool_name, req)
        if not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw  # 兜底固定字符串如 "无mock数据"

    def _writeback_if_cacheable(args: tuple, kwargs: dict, result: Any) -> None:
        if not _is_cacheable_result(result):
            return
        req = _snapshot_args_for_cache(fn, args, kwargs)
        trace_id = get_tool_trace_id() or ""
        normalized_args = _normalize_text(
            json.dumps(req, sort_keys=True, ensure_ascii=False)
        )
        normalized_result = _normalize_text(
            json.dumps(result, ensure_ascii=False)
            if isinstance(result, (dict, list))
            else str(result)
        )
        try:
            _append_to_sandbox_csv(
                tool_name=tool_name,
                trace_id=trace_id,
                arguments=normalized_args,
                result=normalized_result,
            )
            logger.info("[sandbox_cache] %s 真调成功 → 写回沙箱", tool_name)
        except Exception as e:
            logger.warning("[sandbox_cache] %s 写回失败 (不阻塞返回): %s", tool_name, e)

    if asyncio.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            sandbox = get_tool_sandbox() or kwargs.get("sandbox", False)
            if sandbox is True:
                return _strict(args, kwargs)
            # cache-aside
            req = _snapshot_args_for_cache(fn, args, kwargs)
            cached = sandbox_cache_lookup(tool_name, req)
            if cached is not None:
                logger.info("[sandbox_cache] %s 命中沙箱缓存，跳过真调", tool_name)
                return cached
            result = await fn(*args, **kwargs)
            _writeback_if_cacheable(args, kwargs, result)
            return result
        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(*args, **kwargs):
        sandbox = get_tool_sandbox() or kwargs.get("sandbox", False)
        if sandbox is True:
            return _strict(args, kwargs)
        # cache-aside
        req = _snapshot_args_for_cache(fn, args, kwargs)
        cached = sandbox_cache_lookup(tool_name, req)
        if cached is not None:
            logger.info("[sandbox_cache] %s 命中沙箱缓存，跳过真调", tool_name)
            return cached
        result = fn(*args, **kwargs)
        _writeback_if_cacheable(args, kwargs, result)
        return result
    return sync_wrapper
