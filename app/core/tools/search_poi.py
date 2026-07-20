"""Search POI service — search_poi 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool(parse_docstring=True)
@safe_tool
@sandbox_cache
def search_poi(
    query: str,
    cur_adcode: str | None = None,
    types: str | None = None,
    city_limit: bool = False,
    page_size: int = 10,
    page_num: int = 1,
):
    """根据关键词搜索 POI（兴趣点）信息。

    Args:
        query: 搜索关键词，可为 POI 名称（"首开广场"）或结构化地址
            （"北京市朝阳区望京阜荣街10号"）。最长 80 字符，且只支持一个关键字。
        cur_adcode: 搜索区划，可传 citycode / adcode / 中文城市名（"北京市"）。
            未指定则全国范围搜索。
        types: 限定 POI 分类码（typecode），多个用 '|' 分隔。可选。
        city_limit: 是否严格限定在 cur_adcode 区域内召回结果。默认 False。
        page_size: 当前分页条数，1-25，默认 10。
        page_num: 第几页，默认 1。

    Returns:
        包含以下字段的字典：
          - status / info / infocode：高德接口状态
          - count：本次返回 POI 数
          - pois：POI 列表，每项含 name / poiid / address / x / y / type /
            typecode / pname / cityname / adname / adcode / citycode 等
    """
    raise NotImplementedError(
        "search_poi 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
