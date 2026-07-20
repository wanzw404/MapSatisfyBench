"""火客飞货架查询工具 — search_train_or_flights_tickets 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool(parse_docstring=True)
@safe_tool
@sandbox_cache
def search_train_or_flights_tickets(
    startPoiId: str | None = None,
    endPoiId: str | None = None,
    travelType: str | None = None,
    startTime: str | int | None = None,
    endTime: str | int | None = None,
):
    """火客飞货架查询：按起终点 POI 与出发时间区间查询火车票或飞机票货架。

    Args:
        startPoiId: 起点 POI ID（高德 POI 唯一标识，如 "B0FFKGKFP5"）。未知时先用 search_poi 查询获得。必填。
        endPoiId: 终点 POI ID。同上规则。必填。
        travelType: 出行类型；仅支持 'train'（火车票）或 'flight'（飞机票）。必填。
        startTime: 出发时间起点的毫秒级 UNIX 时间戳（如 1779175497000），
            仅接受数字或纯数字字符串。必填。
        endTime: 出发时间终点的毫秒级 UNIX 时间戳，格式同 start_time。必填。

    Returns:
        含 success / content / raw 的 dict：
          - success：True 表示后端调用成功
          - content：业务数据（票务列表等），失败时为 None
          - raw：完整原始响应，便于排障
          - error：失败时给出错误描述
    """
    raise NotImplementedError(
        "search_train_or_flights_tickets 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
