"""Restaurant reservation service — restaurant_reservation 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool
@safe_tool
@sandbox_cache
def restaurant_reservation(
    poi_id: str,
    table_type: str | None = None,
    people_num: str | None = None,
    order_time: str | None = None,
):
    """该工具用于查询餐厅订座信息，结合人数、桌型及预订时间查询餐厅的相关信息。

    Args:
        poi_id: 餐厅POI ID，用于识别特定餐厅。注意必须是POI ID，不能是餐厅的名字。
        table_type: 订座桌型。可选值：["大厅", "散台", "包厢"]。默认为None。
        people_num: 用餐人数。默认为None。
        order_time: 预约用餐日期及时间，格式如"2025-09-11 18:00"。默认为None。

    Returns:
        Any: 包含预订相关信息的字典，包括：
         - 预订状态
         - 座位信息
         - 餐厅详情
         - 预订时间确认

         如果查询失败，返回包含error字段的字典。
    """
    raise NotImplementedError(
        "restaurant_reservation 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
