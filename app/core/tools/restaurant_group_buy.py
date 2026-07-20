"""Restaurant group buy service — restaurant_group_buy 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool
@safe_tool
@sandbox_cache
def restaurant_group_buy(
    poi_id: str,
):
    """该工具用于查询指定餐厅的团购套餐信息，包括套餐详情、价格信息、下单链接等相关信息。

    Args:
        poi_id: 餐厅的POI ID，用于识别特定餐厅，例如"B0FFJGDE5L"。

    Returns:
        Any: 包含团购套餐相关信息的字典，包括：
         - 团购套餐详情
         - 价格信息
         - 下单链接
         - 套餐描述

         如果查询失败，返回包含error字段的字典。
    """
    raise NotImplementedError(
        "restaurant_group_buy 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
