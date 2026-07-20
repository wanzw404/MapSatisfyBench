"""LLM 调用层的限流 + 重试 wrapper（透明包装任意 BaseLLMProvider）。

设计目的
--------
评分器（JudgeAgent + 5 个 judge + verifier + meta_judge）每个 case 内通过
``asyncio.gather`` 同时发出 5 个 LLM 调用；外层 batch_evaluate_from_simulator
又用 ``Semaphore(N)`` 并发 N 个 case。两者相乘，瞬时 LLM 调用 = case 并发 ×
fan-out，很容易压垮上游网关。

本模块提供一个**透明 wrapper**，对所有 ``achat`` 调用同时施加：

  1. **限流（QPS 上限）**：全局共享 ``AsyncTokenBucket``，所有 case / judge
     共用一桶，超额自动 await sleep
  2. **并发上限（in-flight Semaphore）**：兜底，防止 LLM 响应慢时排队无限堆积
  3. **重试**：网络层错误（timeout / connection / 5xx / 429）自动重试 1 次，
     固定 1s 退避；语义错误（4xx）不重试

为什么写 wrapper 而不动 ``OpenAICompatProvider.achat``
-------------------------------------------------
- 业务零侵入：JudgeAgent / 各 judge / verifier / meta 完全不感知
- 切换 provider 不影响：未来如果接其它 LLM 提供方，wrapper 一样能套
- 测试友好：mock inner provider 就能验证限流 + 重试行为

与 ``agent_simulator._ainvoke_with_retry`` 关系
-----------------------------------------------
后者只覆盖 LangChain ``ChatOpenAI`` 路径（agent 模拟器侧），本 wrapper 只用于
评分器侧（``BaseLLMProvider.achat`` 路径），两条 retry 路径互不重叠。
retryable 异常集合与之对齐保持口径一致。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async token bucket
# ---------------------------------------------------------------------------


class AsyncTokenBucket:
    """asyncio-friendly 令牌桶：rate 个 token/秒，capacity 容量。

    与 ``app/core/tools/base.py:TokenBucket`` 同算法，但用 ``asyncio.Lock``
    + ``asyncio.sleep`` 替代 ``threading.Lock`` + ``time.sleep``——后者在
    async 事件循环里会阻塞所有协程，QPS 限制反而成单线程瓶颈。

    无 ``timeout`` 参数：调用方 ``acquire()`` 一定等到拿到 token，确保限流
    严格生效（不会因为超时跳过 token）。
    """

    def __init__(self, rate: float, capacity: int) -> None:
        self.rate = float(rate)
        self.capacity = float(max(1, capacity))
        self.tokens = self.capacity
        self.last_time = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """阻塞直到拿到一个 token。"""
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self.last_time
                self.tokens = min(
                    self.capacity, self.tokens + elapsed * self.rate
                )
                self.last_time = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait_time = (1 - self.tokens) / self.rate
            # 锁外 sleep，避免阻塞其它 acquire 抢锁
            await asyncio.sleep(wait_time)


# ---------------------------------------------------------------------------
# Retryable exception set（与 agent_simulator._ainvoke_with_retry 对齐）
# ---------------------------------------------------------------------------


def _build_retryable_tuple() -> tuple[type[BaseException], ...]:
    """组装 retryable 异常元组。openai SDK 局部 import，避免硬依赖。"""
    types: list[type[BaseException]] = [asyncio.TimeoutError]
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
        types.extend(
            [APITimeoutError, APIConnectionError, InternalServerError, RateLimitError]
        )
    except ImportError:  # pragma: no cover
        logger.warning(
            "[RateLimitedRetry] openai SDK 缺失，retry 仅覆盖 asyncio.TimeoutError"
        )
    return tuple(types)


_RETRYABLE: tuple[type[BaseException], ...] = _build_retryable_tuple()


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class RateLimitedRetryLLMProvider(BaseLLMProvider):
    """透明 wrapper：在任意 ``BaseLLMProvider`` 外加限流 + 重试。

    用法::

        inner = OpenAICompatProvider(base_url=..., api_key=..., model=...)
        provider = RateLimitedRetryLLMProvider(inner, qps=100)
        # 之后所有 provider.achat(...) 自动经过限流 + 重试

    构造参数
    --------
    inner :
        被包装的 ``BaseLLMProvider`` 实例
    qps :
        每秒 token 上限（全局共享，所有 ``achat`` 调用共用）。``<=0`` 表示
        关闭限流（fast-path 透明转发）
    in_flight :
        in-flight 兜底 Semaphore 上限。默认 ``qps * 2``——防止 LLM 响应慢
        时排队无限堆积。``<=0`` 表示不限
    max_retries :
        网络错误重试次数。``1`` 表示失败后再试 1 次（共最多 2 次尝试）
    retry_backoff :
        重试前固定退避秒数。只重试 1 次时指数退避无意义，固定 1s 足够
    """

    def __init__(
        self,
        inner: BaseLLMProvider,
        *,
        qps: float = 100.0,
        in_flight: int | None = None,
        max_retries: int = 1,
        retry_backoff: float = 1.0,
    ) -> None:
        # 不调 super().__init__()——抽象基类的 __init__ 强制要 base_url/api_key，
        # wrapper 直接代理 inner 的属性即可
        self._inner = inner
        self.base_url = inner.base_url
        self.api_key = inner.api_key
        self.model = inner.model

        self._qps = float(qps)
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff = max(0.0, float(retry_backoff))

        self._bucket: AsyncTokenBucket | None = None
        if self._qps > 0:
            self._bucket = AsyncTokenBucket(rate=self._qps, capacity=int(self._qps))

        if in_flight is None:
            in_flight = int(self._qps * 2) if self._qps > 0 else 0
        self._inflight_sem: asyncio.Semaphore | None = (
            asyncio.Semaphore(in_flight) if in_flight > 0 else None
        )

        logger.info(
            "[RateLimitedRetry] inner=%s model=%s qps=%s in_flight=%s "
            "max_retries=%s backoff=%ss",
            type(inner).__name__, self.model, self._qps,
            in_flight if in_flight > 0 else "off",
            self._max_retries, self._retry_backoff,
        )

    async def _acquire(self) -> None:
        """限流：先 in-flight 占位（同步上下文管理），再 token bucket。"""
        if self._bucket is not None:
            await self._bucket.acquire()

    async def achat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        """限流 + 重试包装的 achat。

        每次重试也走 token bucket（重试也算 1 QPS，不突破上限）。
        in-flight Semaphore 持有跨越整次尝试（含重试间的 sleep），防止失败
        case 长时间挂住超过 in_flight 数。
        """
        # in-flight 兜底
        async def _do_once() -> LLMResponse:
            await self._acquire()
            return await self._inner.achat(messages, **kwargs)

        if self._inflight_sem is not None:
            async with self._inflight_sem:
                return await self._call_with_retry(_do_once)
        return await self._call_with_retry(_do_once)

    async def _call_with_retry(self, do_once):
        last_exc: BaseException | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await do_once()
            except _RETRYABLE as e:
                last_exc = e
                if attempt >= self._max_retries:
                    logger.warning(
                        "[RateLimitedRetry] 最终失败（%d/%d）model=%s: %s: %s",
                        attempt + 1, self._max_retries + 1, self.model,
                        type(e).__name__, str(e)[:200],
                    )
                    break
                logger.warning(
                    "[RateLimitedRetry] 第 %d/%d 次失败（%s: %s），%.1fs 后重试 model=%s",
                    attempt + 1, self._max_retries + 1,
                    type(e).__name__, str(e)[:120],
                    self._retry_backoff, self.model,
                )
                await asyncio.sleep(self._retry_backoff)
            except Exception:
                # 非 retryable：语义错误（400/401/403/404 等），重试也是同样结果
                raise

        assert last_exc is not None
        raise last_exc

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        """sync 通道：保留旧 API 行为，直接转发到 inner.chat（不限流不重试）。

        评分器全 async，本 wrapper 主要面向 achat；sync chat 极少调用，不引入
        额外复杂度。如确有 sync 需求再单独处理。
        """
        return self._inner.chat(messages, **kwargs)
