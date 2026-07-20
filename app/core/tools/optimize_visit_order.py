"""Optimize visit order — optimize_visit_order 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any, Dict

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool
@safe_tool
@sandbox_cache
def optimize_visit_order(
    end_lat: str | None = None,
    end_lon: str | None = None,
    end_name: str | None = None,
    mode: str | None = None,
    start_lat: str | None = None,
    start_lon: str | None = None,
    start_name: str | None = None,
    via_names: str | None = None,
    via_points: str | None = None,
):
    """优化访问顺序工具：根据起点、途径点和终点经纬度，计算并返回最佳的访问顺序。该工具用于解决旅行商问题（TSP）的变种，支持两种模式：
        1. fixed_end: 起点出发，终点结束（固定终点）
        2. flexible_end: 起点出发，任意点结束（可在任意点结束）

    Args:
        start_name (str): 起点poi名称，如 "天安门广场"
        start_lat (str): 起点纬度，如 "39.9042"
        start_lon (str): 起点经度，如 "116.4074"
        via_names (str): 途径点poi名称列表，"途径点1,途径点2,..."
        via_points (str): 途径点列表，"lon1,lat1;lon2,lat2;..."
        end_name (str): 终点poi名称，如 "故宫博物院"
        end_lat (str): 终点纬度，如 "39.9151"
        end_lon (str): 终点经度，如 "116.4034"
        mode (str, optional): 访问模式，可选值：
            - "fixed_end": 起点出发，终点结束（默认）
            - "flexible_end": 起点出发，任意点结束（可在任意点结束）
            默认为 "fixed_end"

    Returns:
        Dict[str, Any]: 包含优化结果的字典，格式如下：
            {
                "optimal_visit_order": [
                    {"poi_name": "天安门广场", "lat": 39.9042, "lon": 116.4074, "type": "start"},
                    {"poi_name": "途径点1", "lat": 39.9151, "lon": 116.4034, "type": "via"},
                    ...
                    {"poi_name": "故宫博物院", "lat": 39.9151, "lon": 116.4034, "type": "end"}
                ]
            }
            若计算失败，则返回包含error字段的字典。
    """
    raise NotImplementedError(
        "optimize_visit_order 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
