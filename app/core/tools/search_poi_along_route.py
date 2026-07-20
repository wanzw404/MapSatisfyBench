"""Search POI along route — search_poi_along_route 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool(parse_docstring=True)
@safe_tool
@sandbox_cache
def search_poi_along_route(
    start_x: str | None = None,
    start_y: str | None = None,
    end_x: str | None = None,
    end_y: str | None = None,
    keywords: str | None = None,
    route_type: str | None = None,
    range: str | None = None,
):
    """搜索从起点到终点沿途的兴趣点（POI）。用于获取用户指定路线沿途的目标 POI，适用于"从颐和园开车到天安门，顺路找个餐厅"这类需求。坐标须为真实值，严禁编造。

    Args:
        start_x: 起点经度。必填。
        start_y: 起点纬度。必填。
        end_x: 终点经度。必填。
        end_y: 终点纬度。必填。
        keywords: 搜索关键词，例如"餐厅"、"酒店"、"加油站"等。必填。
        route_type: 交通类型；"0" 表示驾车（默认），"2" 表示步行。
        range: 沿途搜索距离，单位米。不填时驾车默认 10000，步行默认 3000。

    Returns:
        沿途 POI 列表，每条含 poiId / poiName / location / category / ttag / rankInfo 等字段；搜索失败时返回空列表或 status=error。
    """
    raise NotImplementedError(
        "search_poi_along_route 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
