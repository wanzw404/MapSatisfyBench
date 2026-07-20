"""Search POI around multiple points — search_poi_around_multipoints 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any, Optional

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool(parse_docstring=True)
@safe_tool
@sandbox_cache
def search_poi_around_multipoints(
    pois: list | str | None = None,
    query: str | None = None,
    range_meters: int | float | str | None = None,
    need_centrality_filter: str | bool | None = None,
    rating_range: str | None = None,
    hotel_price_range: str | None = None,
    general_price_range: str | None = None,
    sandbox: bool = False,
):
    """搜索多个中心点周边的兴趣点（POI），可按中心性分数排序。适用于「找一个对多个景点都方便的酒店/餐厅」这类多锚点周边场景。

    Args:
        query: 搜索关键词，如 "酒店" / "饭店" / "餐厅"。必填。
        pois: 中心点列表，每个元素是 {"x": 经度, "y": 纬度} 的 dict；可传 list
            或 JSON 字符串。必填。
        range_meters: 单点周边搜索半径，单位米。默认 5000。
        need_centrality_filter: 是否启用中心性评分排序筛选；启用时只返回评分
            最高的前 10 条。默认 True。可传 "true"/"false" 或 bool。
        rating_range: 评分筛选范围，格式 "[low,high]" 闭区间，0–10。会透传给
            底层 search_around_poi；高德 v5 around API 不原生支持评分筛选，
            底层会加 ``unsupported`` 字段提醒，需要的话请按返回结果的
            business.rating 自行筛选。
        hotel_price_range: 酒店类价格筛选，格式 "[low,high]" 闭区间，最小0，最大999999。例如 "[100,500]" 表示价格 100–500 元。
        general_price_range(str,optional)：非酒店类价格筛选范围，格式为"[low,high]" 闭区间，最小值为0，最大值为999999。例如价格大雨等于100的餐厅："[100,999999]"。默认空字符串。
        sandbox: 内部参数，不要由 LLM 设置。

    Returns:
        成功：``{"query", "scenic_spots_count", "search_range_meters",
        "total_unique_pois", "pois"}``；pois 列表中每条 POI 还含
        ``centrality_metrics``（avg/max 距离与综合评分）。
        失败：``{"error": "...", "search_summary"?: [...], "total_found"?: 0}``。
    """
    raise NotImplementedError(
        "search_poi_around_multipoints 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
