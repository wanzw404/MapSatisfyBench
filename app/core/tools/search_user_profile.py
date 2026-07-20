import json
from typing import Any
from .base import _do_vipserver_get_request, _is_prod, http_get, record, safe_tool, sandbox_cache

import requests

from langchain_core.tools import tool

from ...services.run_sandbox import run_sandbox


@tool
@safe_tool
def search_user_profile(
    uid: str | None = None
):
    """用户画像工具。根据用户adiu查询用户的画像信息。会返回基于用户一年的行为总结出来的用户基础信息，如年龄、性别、消费能力、职业等，以及兴趣偏好，例如出行偏好、旅游偏好、饮食偏好、住宿偏好、休闲娱乐偏好等信息。

    Args:
        uid: 用户id (或conversation_id、trace_id)

    Returns:
        用户画像描述
    """
    return run_sandbox(
        "search_user_profile",
        {
            "adiu": uid,
        },
    )