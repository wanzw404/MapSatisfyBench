import json
from typing import Any
from .base import _do_vipserver_get_request, _is_prod, http_get, record, safe_tool, sandbox_cache

import requests

from langchain_core.tools import tool

from ...services.run_sandbox import run_sandbox


@tool
@safe_tool
def search_user_action_summary(
    uid: str | None = None
):
    """近三个月用户历史行为查询工具。根据用户adiu查询用户近三个月的全域行为总结。会返回用户近三个月的出行方式与大致比例、本异地和不同时间的兴趣偏好、典型出行模式、行为习惯、访问频率较高的 POI 类型、当前生活与活动区域、生活模式等信息。

    Args:
        uid: 用户id(或conversation_id、trace_id)

    Returns:
        用户近三个月的全域行为总结
    """
    return run_sandbox(
        "search_user_action_summary",
        {
            "adiu": uid,
        },
    )