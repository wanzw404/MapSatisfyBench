"""FactVerifier — cross-checks a fact against the open web, not Amap.

Pipeline for every fact with ``need_verify=True``::

    claim ──► (LLM, inline prompt) ──► natural-language search query
           ──► WebSearchClient.search(query) ──► snippets
           ──► (LLM, inline prompt) ──► {verified_ok, reason}
           ──► back-fill fact.verified_ok / verify_reason / faithful

Design notes:
    * Two short LLM calls per fact (query-rephrase + judgment). Both use
      terse inline prompts kept in this file — they are part of the
      evaluation contract, not user-editable config.
    * If the search returns nothing usable, we leave ``verified_ok = None``
      (IFS treats unknown as "not proven false", the conservative default).
    * Only flip ``fact.faithful = False`` when the LLM judges ``False`` with
      supporting snippets — unknown / inconclusive never hurts the score.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from app.core.agent.llm.base import BaseLLMProvider
from app.core.evaluation.schema import (
    FactStatement,
    JudgeVerdict,
    RubricElementEvidence,
    VerifyLogEntry,
)

from .web_search_client import SearchResult, WebSearchClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inline LLM prompts (short, per user spec)
# ---------------------------------------------------------------------------

_QUERY_REWRITE_PROMPT = (
    "你是一个搜索助手。把下面这条事实陈述改写成一个简洁的中文网页搜索问题，"
    "仅输出问题本身，不要引号、不要解释、不要前缀。\n"
    "事实陈述：{claim}\n"
    "问题："
)

_JUDGMENT_PROMPT = (
    "你是一个事实核查员。请根据下方网络搜索结果，判断事实陈述是否属实。"
    "严格按 JSON 输出，不要添加任何解释或代码围栏。\n"
    "JSON 字段说明：\n"
    '  - verified_ok: true（搜索结果支持该陈述）/ false（搜索结果明确反驳）/ null（信息不足无法判断）\n'
    "  - reason: 一句话解释，引用搜索结果中的关键依据\n\n"
    "事实陈述：{claim}\n\n"
    "搜索结果：\n{snippets}\n\n"
    '输出格式：{{"verified_ok": true|false|null, "reason": "..."}}'
)

# Strip ```json ... ``` fences.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class FactVerifier:
    """Web-search + LLM-judgment fact verifier."""

    def __init__(
        self,
        search_client: WebSearchClient,
        llm_provider: BaseLLMProvider,
        *,
        max_snippets_for_llm: int = 6,
    ) -> None:
        self.client = search_client
        self.llm = llm_provider
        self.max_snippets_for_llm = max_snippets_for_llm

    # -----------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------

    async def verify(self, verdict: JudgeVerdict) -> list[VerifyLogEntry]:
        """Mutate ``verdict.fact_summary`` in place, return audit log."""
        log: list[VerifyLogEntry] = []
        for fact in verdict.fact_summary.faithful_facts:
            entry = await self._verify_one(fact)
            if entry is not None:
                log.append(entry)
        for fact in verdict.fact_summary.unfaithful_facts:
            # Unfaithful facts still get a chance — the LLM may have been
            # wrong and web search may prove the claim true.
            entry = await self._verify_one(fact)
            if entry is not None:
                log.append(entry)
        return log

    # -----------------------------------------------------------------
    # Per-fact pipeline
    # -----------------------------------------------------------------

    async def _verify_one(self, fact: FactStatement) -> Optional[VerifyLogEntry]:
        if not fact.need_verify:
            return None

        claim = (fact.statement or "").strip()
        if not claim:
            return VerifyLogEntry(
                statement=fact.statement,
                source_tool=fact.source_tool,
                ok=None,
                reason="empty statement",
            )

        # Step 1 — fact → search query
        try:
            query = await self._rewrite_to_query(claim)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Query-rewrite LLM failed for claim=%r: %s", claim, exc)
            query = claim  # fall back to raw claim

        # Step 2 — call web search
        try:
            search = await self.client.search(query)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Web search failed for query=%r: %s", query, exc)
            search = SearchResult(
                ok=False, query=query, source="unavailable", error=str(exc)
            )

        # Step 3 — LLM judgment against snippets (only when we have data)
        verified_ok: Optional[bool]
        reason: str
        if search.source == "unavailable" or not search.snippets:
            verified_ok = None
            reason = search.error or "no web evidence available"
        else:
            try:
                verified_ok, reason = await self._judge(claim, search)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Judgment LLM failed for claim=%r: %s", claim, exc)
                verified_ok = None
                reason = f"judgment error: {exc}"

        # Apply result back onto the fact
        fact.verified_ok = verified_ok
        fact.verify_reason = reason
        if verified_ok is False:
            fact.faithful = False
            fact.reason = "verified_wrong"

        return VerifyLogEntry(
            statement=fact.statement,
            source_tool=fact.source_tool,
            ok=verified_ok,
            reason=reason,
            api_response={"query": query, "search": search.raw},
        )

    # -----------------------------------------------------------------
    # LLM helpers (short inline prompts)
    # -----------------------------------------------------------------

    async def _rewrite_to_query(self, claim: str) -> str:
        prompt = _QUERY_REWRITE_PROMPT.format(claim=claim)
        response = await self.llm.achat(
            [{"role": "user", "content": prompt}]
        )
        query = (response.content or "").strip()
        # Strip quotes / stray whitespace / trailing punctuation.
        query = query.strip("\"'“”‘’`").strip()
        query = query.splitlines()[0].strip() if query else claim
        return query or claim

    async def _judge(
        self, claim: str, search: SearchResult
    ) -> tuple[Optional[bool], str]:
        snippets_text = "\n".join(
            f"- {s}" for s in search.snippets[: self.max_snippets_for_llm] if s
        )
        prompt = _JUDGMENT_PROMPT.format(claim=claim, snippets=snippets_text)
        response = await self.llm.achat(
            [{"role": "user", "content": prompt}]
        )
        return self._parse_judgment(response.content or "")

    # -----------------------------------------------------------------
    # Rubric-row element verification (新版 IFS 用)
    # -----------------------------------------------------------------

    async def verify_rubric_rows(
        self, verdict: JudgeVerdict
    ) -> list[VerifyLogEntry]:
        """对 verdict.rubric_row_judgments 中所有 need_external_verify=True
        的要素跑外部 web 搜索 + LLM 判定。

        行级语义（all-or-nothing + skip）::

          element 验证通过  → grounded=True (升级)
          element 验证失败  → grounded=False, reason='external_verify_failed'
          element 验证不可达 → element.skipped=True (从 IFS 中忽略)

          行内全部 element 都 skipped → row.skipped=True (整行从 IFS 中忽略)
          否则按 all-or-nothing 重写 row.score：
            非 skipped 中任一 grounded=False → score=0
            非 skipped 全部 grounded=True   → score=1
        """
        log: list[VerifyLogEntry] = []
        for row in verdict.rubric_row_judgments:
            for element in row.elements:
                if not element.need_external_verify:
                    continue
                entry = await self._verify_element(element)
                if entry is not None:
                    log.append(entry)
            self._refresh_row_score(row)
        return log

    async def _verify_element(
        self, element: RubricElementEvidence
    ) -> Optional[VerifyLogEntry]:
        """对单个要素跑 query rewrite → web search → LLM 判定。"""
        # 拼一段更明确的待验证文本：要素名 + content_quote
        claim_parts = [element.element]
        if element.content_quote:
            claim_parts.append(element.content_quote)
        claim = "：".join(p for p in claim_parts if p).strip()
        if not claim:
            element.skipped = True
            element.external_verify_reason = "empty claim, skipped"
            return VerifyLogEntry(
                statement=element.element, ok=None,
                reason="empty claim, skipped",
            )

        # Step 1 — claim → search query
        try:
            query = await self._rewrite_to_query(claim)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Rubric query-rewrite failed: %s", exc)
            query = claim

        # Step 2 — web search
        try:
            search = await self.client.search(query)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Rubric web search failed for %r: %s", query, exc)
            search = SearchResult(
                ok=False, query=query, source="unavailable", error=str(exc)
            )

        # Step 3 — judge against snippets
        verified_ok: Optional[bool]
        reason: str
        if search.source == "unavailable" or not search.snippets:
            # 验证不可达 → skipped (IFS 忽略此要素)
            verified_ok = None
            reason = search.error or "no web evidence available"
        else:
            try:
                verified_ok, reason = await self._judge(claim, search)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Rubric judgment LLM failed: %s", exc)
                verified_ok = None
                reason = f"judgment error: {exc}"

        # 写回要素
        element.external_verified_ok = verified_ok
        element.external_verify_reason = reason
        if verified_ok is True:
            # 升级：通过外部验证 → grounded
            element.grounded = True
            element.reason = None
        elif verified_ok is False:
            element.grounded = False
            element.reason = "external_verify_failed"
        else:
            # 不可达 / 信息不足 → 该要素从 IFS 中忽略
            element.skipped = True

        return VerifyLogEntry(
            statement=element.element, ok=verified_ok,
            reason=reason,
            api_response={"query": query, "search": search.raw},
        )

    @staticmethod
    def _refresh_row_score(row) -> None:
        """根据要素的最新 grounded / skipped 状态重写 row.score / row.skipped。

        - 行内所有要素都 skipped → row.skipped=True (IFS 忽略此行)
        - 非 skipped 中任一 grounded=False → row.score=0
        - 非 skipped 全部 grounded=True → row.score=1
        """
        if not row.elements:
            return
        non_skipped = [e for e in row.elements if not e.skipped]
        if not non_skipped:
            row.skipped = True
            row.score = 0
            return
        row.skipped = False
        all_ok = all(e.grounded for e in non_skipped)
        row.score = 1 if all_ok else 0

    @staticmethod
    def _parse_judgment(raw: str) -> tuple[Optional[bool], str]:
        text = _FENCE_RE.sub("", (raw or "").strip()).strip()
        if not text.startswith("{"):
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                text = text[start : end + 1]
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None, f"unparseable LLM judgment: {raw[:200]}"

        ok_raw = payload.get("verified_ok")
        if isinstance(ok_raw, bool):
            verified_ok: Optional[bool] = ok_raw
        elif ok_raw is None:
            verified_ok = None
        elif isinstance(ok_raw, str):
            low = ok_raw.strip().lower()
            if low in ("true", "1", "yes", "是"):
                verified_ok = True
            elif low in ("false", "0", "no", "否"):
                verified_ok = False
            else:
                verified_ok = None
        else:
            verified_ok = None

        reason = str(payload.get("reason") or "").strip() or "no reason given"
        return verified_ok, reason
