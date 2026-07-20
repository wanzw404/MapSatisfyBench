"""Weather service — get_weather 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool
@safe_tool
@sandbox_cache
def get_weather(
    location: str | None = None,
    date: str | None = None,
    whole: bool = True,
):
    """获取指定地点的天气信息。提供4天内（今天 + 未来 3 天）的天气查询功能，用于获取指定地点的实时天气和未来多天的逐日天气预报。支持查询实时天气状况（如晴、雨、雪、阴等）、温度、湿度、风力等信息。

    Args:
        location (str): 查询地点，支持多种格式的地理位置信息。当用户输入中能够明确查询地点时，
            必须提取出来，比如城市名（如"北京"）、区域名（如"朝阳区"）、具体地标（如"天安门"）等。
            当用户未指定地点时，默认为"用户当前位置"。
        date (str): 用户希望查询天气的截止日期，必须保持YYYY-MM-DD格式。
            需要参照当下日期得出具体数值。
            示例1: "查询本周六的天气"，当前时间2025年8月25日 -> date='2025-08-30'
            示例2: "明天天气怎么样"，当前时间2021年3月4日 -> date='2021-03-05'
        whole (bool, optional): 天气数据返回范围控制。设为True时返回从今天开始到截止日期的
            每日天气详情，设为False时仅返回截止日期当天的天气信息。默认为True。

    Returns:
        Any: 包含天气信息的字典，包括：
             - 实时天气状况（live 字段：weather/temperature/humidity/winddirection/windpower）
             - 多日预报（forecasts 字段，每条含 date/dayweather/nightweather/
               daytemp/nighttemp/daywind/nightwind/daypower/nightpower/week）
             - 数据源说明（note）

             如果查询失败，返回包含 error 字段的字典。
    """
    raise NotImplementedError(
        "get_weather 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
