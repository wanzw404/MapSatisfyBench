"""Taxi route plan — get_taxi_route_plan 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool
@safe_tool
@sandbox_cache
def get_taxi_route_plan(
    start_lat: str,
    start_lon: str,
    end_lat: str,
    end_lon: str,
    start_name: str,
    end_name: str,
    end_poi_id: str,
    start_poi_id: str,
    sandbox: bool = False,
    request: dict | None = None,
):
    """获取打车路线规划信息，包含预估价格、应答时间、到达时间等。

    Args:
        start_lat: 起点纬度
        start_lon: 起点经度
        end_lat: 终点纬度
        end_lon: 终点经度
        start_name: 起点名称
        end_name: 终点名称
        end_poi_id: 终点 POI ID
        start_poi_id: 起点 POI ID
        sandbox: 是否执行沙箱模式
        request: 扩展字段

    Returns:
        打车路线规划结果，包含 estimateAnswerTime、estimateArriveTime、detailsForAiSummary 等字段
    """
    raise NotImplementedError(
        "get_taxi_route_plan 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
