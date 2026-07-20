"""Search around POI service — search_around_poi 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool(parse_docstring=True)
@safe_tool
@sandbox_cache
def search_around_poi(
    query: str | None = None,
    range: int | float | str | None = None,
    rating_range: str | None = None,
    sort_rule: str | None = None,
    x: int | float | str | None = None,
    y: int | float | str | None = None,
    sandbox: bool = False,
    general_price_range: str = '',
    hotel_price_range: str = '',
    request: dict | None = None,
):
    """在指定坐标点周围搜索兴趣点（POI）。用于检索某个圆形区域内的 POI。必须使用真实的中心点坐标，若未知请先调用search_poi 获取，严禁编造坐标。

    Args:
        query: 搜索关键词，例如"餐厅"、"酒店"、"加油站"等。可选；若不填，
            v5 默认按 types=050000|070000|120000（餐饮/生活/商务住宅）召回。
        range: 周边搜索半径，单位米，范围 1-50000；不填默认 5000。
        rating_range: 评分筛选范围，格式 "[low,high]" 闭区间。注意：v5 周边搜索
            原生不支持此参数，传入仅记录在审计日志中；如需评分过滤请在返回
            结果的 business.rating 字段上自行筛选。
        sort_rule: 排序规则，可选；接受字符串或整数。常用取值：1/"distance"
            按距离从近到远；0/"weight" 按综合权重排序（默认）。其它旧版细粒度
            档位（按评分/价格）v5 不支持，会自动回退到综合排序。
        x: 搜索中心点的经度坐标，6 位小数以内。必填。
        y: 搜索中心点的纬度坐标，6 位小数以内。必填。
        sandbox: 是否执行沙箱模式。
        general_price_range: 非酒店类价格筛选范围，格式 "[low,high]" 闭区间，
            最小 0、最大 999999。例如价格 ≥100 元的餐厅："[100,999999]"。
            仅对 type 不含"酒店"的 POI 生效；POI 缺 business.cost 字段时**保留**。
            空字符串不启用筛选。
        hotel_price_range: 酒店类价格筛选范围，格式 "[low,high]" 闭区间，
            最小 0、最大 999999。例如酒店 100-500 元："[100,500]"。
            仅对 type 含"酒店"的 POI 生效；POI 缺 business.cost 字段时**丢弃**。
            空字符串不启用筛选。
        request: 扩展字段

    Returns:
        包含 status / info / infocode / count / pois 的字典；pois 中每条含
        name / poiid / address / x / y / distance（米）/ type / typecode /
        pname / cityname / adname / adcode / citycode 以及可选的 business 字段。
        若启用 general_price_range / hotel_price_range，结果会附带
        ``filter_summary``（before_filter / after_filter / dropped_by_*）。
    """
    raise NotImplementedError(
        "search_around_poi 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
