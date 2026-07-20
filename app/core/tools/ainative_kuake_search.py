"""AINative Kuake search — ainative_kuake_search 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool
@safe_tool
@sandbox_cache
def ainative_kuake_search(
    query: str,
    query_write: str,
):
    """使用 AINative 夸克搜索获取网络信息。

    Args:
        query: 用户查询内容
        query_write: 扩展查询词（逗号分隔的相关搜索词）

    Returns:
        夸克搜索结果，包含 title、content、url 等字段
    """
    raise NotImplementedError(
        "ainative_kuake_search 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
