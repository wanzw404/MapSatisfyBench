"""Navigation service — get_navigation 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool
@safe_tool
@sandbox_cache
def get_navigation(
    start_lon: int | float | str | None = None,
    start_lat: int | float | str | None = None,
    end_lon: int | float | str | None = None,
    end_lat: int | float | str | None = None,
    start_name: str | None = None,
    end_name: str | None = None,
    start_poiid: str | None = None,
    end_poiid: str | None = None,
    via_points: str | None = None,
    via_name: str | None = None,
    preferred_road_names: str | None = None,
    avoid_road_names: str | None = None,
    start_datetime: str | None = None,
    end_datetime: str | None = None,
    mode: list | str | None = None,
    route_type: str | None = None,
    sandbox: bool = False,
):
    """获取多种出行方式的路线规划结果，支持获取驾车、公交、骑行、步行、摩托车、货车六种出行方式的路线规划结果及对应路线详情信息，支持单个或多个出行方式的并行查询。

    Args:
        - end_lat (int | float | str | None) = None: float): 终点纬度。
        - end_lon (int | float | str | None) = None: float): 终点经度。
        - end_name (str | None) = None: str, optional): 终点名称，用于前端展示，不影响接口返回。默认为空字符串。
        - end_poiid (str | None) = None: str, optional): 终点poiid，用于前端展示，不影响接口返回。默认为空字符串。
        - mode (list | str | None) = None: List[str], optional): 出行方式列表。支持"驾车"、"骑行"、"摩托车"、"步行"、"公交"、"货车"。如果为空或None，默认查询所有方式。
        - route_type (str | None) = None: int, optional): 路线类型偏好。可选值：
                - 34: 高速优先
                - 35: 不走高速
                - 36: 少收费
                - 37: 大路优先
                - 38: 速度最快
                - 39: 躲避拥堵+高速优先
                - 40: 躲避拥堵+不走高速
                - 41: 躲避拥堵+少收费
                - 42: 少收费+不走高速
                - 43: 躲避拥堵+少收费+不走高速
                - 44: 躲避拥堵+大路优先
                - 45: 躲避拥堵+速度最快
                默认为None。
        - start_lat (int | float | str | None) = None: float): 起点纬度。
        - start_lon (int | float | str | None) = None: float): 起点经度。
        - start_name (str | None) = None: str, optional): 起点名称，用于前端展示，不影响接口返回。默认为空字符串。
        - start_poiid (str | None) = None: str, optional): 起点poiid，用于前端展示，不影响接口返回。默认为空字符串。
        - sandbox (bool) = False: 是否使用沙箱模式
        - via_points (str) = '': str, optional): 途经点，格式为"经度,纬度;经度,纬度;..."。默认为空字符串。
        - via_name (str) = '': str, optional): 途经点名称，格式为 "名称1,名称2,名称3;..."，用于前端展示，不影响接口返回。默认为空字符串。
        - preferred_road_names (str) = ''
        - avoid_road_names (str) = ''
        - start_datetime (str) = '': str, optional): 用户预期从起点出发的时间，格式如"2025-07-11 09:00:00"。如果提供，将计算并返回导航耗时。默认为空字符串。
        - end_datetime (str) = '': str, optional): 用户预期到达目的地的时间，格式如"2025-07-11 10:00:00"。如果提供，将计算导航耗时并预估建议的出发时间。默认为空字符串。

    Returns:
        Any: 包含所有查询方式路线规划信息的列表或字典。成功时返回路线规划详情列表，每个元素包含路线摘要、距离、时间、费用等信息。失败时返回包含error字段的字典。
    """
    raise NotImplementedError(
        "get_navigation 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
