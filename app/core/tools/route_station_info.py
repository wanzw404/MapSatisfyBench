"""Route station info — route_station_info 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool(parse_docstring=True)
@safe_tool
@sandbox_cache
def route_station_info(
    city: str = "",
    line_nums: str = "",
    line_type: str = "",
    stations: str = "",
    lines: str = "",
    station_type: str = "",
    station_names: str = "",
):
    """公交地铁查询。支持按线路名/线路ID/站点名/站点ID 四路批量并发查询，并对结果做线路类型与站点类型过滤、线路与站点结果的双向交叉裁剪（站点的 buslines 仅留命中线路集合的；线路的 busstops 仅留命中站点集合的）。

    Args:
        city: 城市名称、adcode 或 citycode（如 "110000" 或 "北京"）。必填，缺失时直接报错。
        line_nums: 公交/地铁线路名称，多条用英文逗号 "," 分隔，例如 "地铁10号线,地铁5号线"。必填，缺失时直接报错。
        line_type: 线路交通类型筛选，多条用英文逗号 "," 分隔。可填粗类如 "地铁" / "公交"，也可填高德细分类别如 "普通公交" / "机场巴士" / "夜班车" / "地铁专线" 等；采用双向子串匹配（"公交" 命中 "普通公交"，"机场巴士" 仅命中 "机场巴士"）。空字符串视为不限。必填，缺失时直接报错。
        stations: 站点 ID 列表，多条用英文逗号 "," 分隔，例如 "BV10012405,BV10013496"。可选。
        lines: 线路 ID 列表，多条用英文逗号 "," 分隔，例如 "110100023283,110100023282"。可选。
        station_type: 站点交通类型筛选，多条用英文逗号 "," 分隔。可填 "地铁站" / "公交站"，亦可填 "地铁" / "公交"（双向子串匹配）。可选。
        station_names: 站点名称，多条用英文逗号 "," 分隔，例如 "车道沟,慈寿寺"。可选。

    Returns:
        包含 status / city / line_type_filter / station_type_filter /
        lines_by_name / lines_by_id / stations_by_name / stations_by_id /
        errors / summary 的字典。
        每条线路裁剪后含 id/type/name/citycode/start_stop/end_stop/start_time/
        end_time/company/loop/distance/basic_price/total_price/bounds/busstops；
        每条站点含 id/name/adcode/citycode/location/buslines。
    """
    raise NotImplementedError(
        "route_station_info 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
