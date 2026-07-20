"""Product search by POI ID — search_products_by_poiid 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool
@safe_tool
@sandbox_cache
def search_products_by_poiid(
    poiid: str | None = None,
):
    """商品查询。根据POI ID查询该POI下的所有商品信息。该函数用于查询指定兴趣点（POI）下的所有商品信息，包括商品名称、价格、详细描述、套餐内容、购买须知等详细信息。

    Args:
        - poiid (str | None) = None: str): 兴趣点（POI）的唯一标识符，例如"B0K39CGDOX"。

    Returns:
        Any: 商品信息列表或错误信息。成功时返回包含商品信息的列表，每个商品包含：
        - name: 商品名称
        - price: 商品价格（格式为"XX.XX元"）
        - details: 商品详细信息，包括备注、套餐内容、购买须知等

        如果未找到商品，返回提示信息；如果发生错误，返回包含error字段的字典。
    """
    raise NotImplementedError(
        "search_products_by_poiid 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
