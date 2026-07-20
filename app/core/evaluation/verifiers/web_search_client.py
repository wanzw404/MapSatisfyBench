"""Web-search client used by FactVerifier to cross-check factual claims.

The default backend is the Amap-hosted MCP gateway wrapping GoogleWebSearch::

    POST http://amap-llm-biz-app-gateway.amap.com/mcps/2389335511851247/sse
    Authorization: <ak_*>
    Content-Type: application/json
    {
        "jsonrpc": "2.0",
        "id": "<uuid-ish>",
        "method": "tools/call",
        "params": {
            "name": "GoogleWebSearch",
            "arguments": {"query": "<自然语言问题>"}
        }
    }

Endpoints that end in ``/sse`` may stream a Server-Sent-Events payload; we
parse both plain-JSON and SSE responses defensively so that callers never
have to care about the transport.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_ENDPOINT = (
    "http://amap-llm-biz-app-gateway.amap.com/mcps/2389335511851247/sse"
)
DEFAULT_TOKEN = "ak_tpsNJVQB8O"
DEFAULT_TOOL_NAME = "GoogleWebSearch"
DEFAULT_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """Normalized search response."""

    ok: bool
    query: str
    snippets: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    source: str = "google_web_search"  # or "mock" / "unavailable"
    error: str = ""

    @property
    def joined_snippets(self) -> str:
        """Human-readable concatenation for LLM consumption."""
        return "\n".join(f"- {s}" for s in self.snippets if s)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class WebSearchClient(ABC):
    """Abstract search backend. ``search`` must be async."""

    @abstractmethod
    async def search(self, query: str) -> SearchResult:
        """Return normalized ``SearchResult`` for a natural-language query."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Real client — MCP GoogleWebSearch gateway
# ---------------------------------------------------------------------------


class GoogleWebSearchClient(WebSearchClient):
    """POSTs JSON-RPC ``tools/call`` to the MCP gateway and extracts snippets."""

    def __init__(
        self,
        endpoint: str | None = None,
        auth_token: str | None = None,
        tool_name: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_snippets: int = 8,
    ) -> None:
        self.endpoint = endpoint or DEFAULT_ENDPOINT
        self.auth_token = auth_token or DEFAULT_TOKEN
        self.tool_name = tool_name or DEFAULT_TOOL_NAME
        self.timeout = timeout
        self.max_snippets = max_snippets

    async def search(self, query: str) -> SearchResult:
        if not query or not query.strip():
            return SearchResult(ok=False, query=query, source="unavailable", error="empty query")

        payload = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "tools/call",
            "params": {
                "name": self.tool_name,
                "arguments": {"query": query.strip()},
            },
        }
        headers = {
            "Authorization": self.auth_token,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            body_text = resp.text
        except httpx.HTTPError as exc:
            logger.warning("WebSearch HTTP error for %r: %s", query, exc)
            return SearchResult(
                ok=False, query=query, source="unavailable", error=f"HTTP error: {exc}"
            )

        parsed = self._parse_response_body(body_text)
        if parsed is None:
            return SearchResult(
                ok=False,
                query=query,
                source="unavailable",
                error=f"Cannot parse response: {body_text[:200]}",
            )

        snippets = self._extract_snippets(parsed)
        return SearchResult(
            ok=bool(snippets),
            query=query,
            snippets=snippets[: self.max_snippets],
            raw=parsed,
            source="google_web_search" if snippets else "unavailable",
            error="" if snippets else "no snippets in response",
        )

    # -----------------------------------------------------------------
    # Response parsing — handle both plain JSON and SSE streams
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_response_body(body: str) -> Optional[dict[str, Any]]:
        body = (body or "").strip()
        if not body:
            return None

        # Plain JSON first
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        # SSE: lines of "data: {...}" possibly interleaved with "event: ...".
        # We take the last successfully parsed data frame that looks like a
        # JSON-RPC response.
        last_rpc: Optional[dict[str, Any]] = None
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data or data == "[DONE]":
                continue
            try:
                frame = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(frame, dict) and ("result" in frame or "error" in frame):
                last_rpc = frame
        return last_rpc

    @staticmethod
    def _extract_snippets(payload: dict[str, Any]) -> list[str]:
        """Best-effort snippet extraction tolerant of schema variations."""
        result = payload.get("result") or {}
        if not isinstance(result, dict):
            return []

        # MCP tools/call standard wraps tool output under ``content`` list.
        snippets: list[str] = []
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text") or item.get("value") or ""
                if isinstance(text, str) and text.strip():
                    snippets.append(text.strip())

        # Some MCP adapters flatten into result.results / result.data
        for alt_key in ("results", "data", "snippets", "items"):
            items = result.get(alt_key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        text = (
                            item.get("snippet")
                            or item.get("description")
                            or item.get("title")
                            or item.get("text")
                            or ""
                        )
                        if text:
                            snippets.append(str(text).strip())
                    elif isinstance(item, str) and item.strip():
                        snippets.append(item.strip())

        # If the tool dumped a giant text blob under result.text, keep it.
        text_blob = result.get("text") or result.get("output")
        if isinstance(text_blob, str) and text_blob.strip() and not snippets:
            snippets.append(text_blob.strip())

        return snippets
