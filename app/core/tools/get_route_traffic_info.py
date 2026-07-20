"""Route traffic info service — get_route_traffic_info 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool
@safe_tool
@sandbox_cache
def get_route_traffic_info(
    poi_id: str | None = None,
):
    """用于获取指定道路的交通状况和事件信息，支持查询道路拥堵情况、交通事故、道路施工、临时管制等实时交通数据。

    注意:
        此工具仅查询特定道路的交通状况，不适用于路线规划或导航查询。
        例如"望京Soho附近的交通情况"这类查询不能使用该工具。

    Args:
        poi_id (str): POI点ID，用于指定查询的区域或道路位置标识符。注意不能是道路的名字（比如"北五环"），必须是具体的POI ID。

    Returns:
        Any: 详细的交通信息字典，包含：
             - 拥堵等级
             - 事件类型
             - 影响范围
             - 预计恢复时间
             - 其他实时交通数据
             如果查询失败，返回包含error字段的字典。
    """
    raise NotImplementedError(
        "get_route_traffic_info 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
