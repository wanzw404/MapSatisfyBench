"""Rgeo service — get_rgeo 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool
@safe_tool
@sandbox_cache
def get_rgeo(
    x: str | None = None,
    y: str | None = None,
    range_meters: int = 2000
):
    """根据经纬度获取地理信息，包括 adcode、省市区信息、周边 POI 等.

    参数:
        x: 经度
        y: 纬度
        range_meters: 搜索范围（米），默认 2000 米

    返回:
        包含 adcode、省市区信息、周边 POI 等的字典
    """
    raise NotImplementedError(
        "get_rgeo 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
