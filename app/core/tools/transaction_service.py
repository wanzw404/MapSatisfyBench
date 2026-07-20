from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache

@tool
@safe_tool
def transaction_service(
    itemName: str | None = None,
    itemType: str | None = None,
    orderObject: str | None = None,
):
    """该工具可以支持用户商品交易需求，辅助用户购买高德支持的商品，包括但不限于门票及团购券购买、酒店及餐厅美食预定、加油支付等等。记得非精确地点的orderObject应该带类别标签，例如定个北京酒店、预订北京的酒店，orderObject需要是北京酒店。

    Args:
        itemName，从用户问题中提取到的商品名称。
        itemType，交易场景类型，必填。
        orderObject，商家名称，可以是具体POI名称，也可以是一类POI的查询词。酒店预定类目orderObject名称必须带酒店。必填。

    Returns:
        error，是否成功
        content，结果列表
    """

    res = "{\"content\", \"全部交易成功\"}"

    return res

