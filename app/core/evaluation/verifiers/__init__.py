"""Fact verification subsystem.

Pipeline role::

    LLM (JudgeVerdict with need_verify flags)
        → FactVerifier.verify(verdict)                            [async]
            → WebSearchClient.search(query)  ── GoogleWebSearch
            → BaseLLMProvider.achat(...)    ── judgment against snippets
        → back-fill verified_ok on each fact
"""

from .fact_verifier import FactVerifier
from .web_search_client import (
    GoogleWebSearchClient,
    SearchResult,
    WebSearchClient,
)

__all__ = [
    "WebSearchClient",
    "GoogleWebSearchClient",
    "SearchResult",
    "FactVerifier",
]
