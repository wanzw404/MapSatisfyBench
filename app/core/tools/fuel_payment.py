"""Fuel payment service — fuel_payment 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool
@safe_tool
@sandbox_cache
def fuel_payment(
    poi_id: str | None = None,
    oil_num: str | None = None,
    oil_gun: str | None = None,
    oil_price: str | None = None,
    oilgun: str | None = None,
    oilnum: str | None = None,
    oilprice: str | None = None,
    sandbox: bool = False,
):
    """用于查询加油站的支付相关信息，包括油品、油枪、价格等详细信息。

    Args:
        poi_id (str): 加油站的POI ID，用于识别特定加油站，例如"B0FFHL3VW5"。
        oil_num (str, optional): 加油油号。可选值：["92", "95", "98", "0"]。默认为None。
        oil_gun (str, optional): 加油站油枪号，填写油枪编号对应数字。默认为None。
        oil_price (str, optional): 加油金额，填写数字金额，单位是元。默认为None。
        oilgun: str | None = None,
        oilnum: str | None = None,
        oilprice: str | None = None,
        sandbox: 是否使用沙箱模式


    Returns:
        Any: 包含加油支付相关信息的字典，包括：支付状态、订单信息、油品详情、价格信息，如果查询失败，返回包含error字段的字典。
    """
    raise NotImplementedError(
        "fuel_payment 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
