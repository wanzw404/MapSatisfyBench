from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache

@tool
@safe_tool
def scenic_ticket_transaction(
    dateTime: str | None = None,
    personNum: str | None = None,
    sightName: str | None = None,
    ticketSession: str | None = None,
    ticketType: str | None = None,
):
    """景区门票交易工具。

    Args:
        参数名称	参数描述	参数类型	是否必填	传入方法
        dateTime	购买日期, 购买门票的日期, 包括节假日,比如国庆,十一,元旦,劳动节等等	STRING	否	body
        sightName	景区名称	STRING	否	body
        ticketType	门票类型, 如: 成人票、儿童票, 学生票,老人票,团购票等	STRING	否	body
        ticketSession	门票场次, 门票特定的场景,如: 上午场、下午场, 10点场等	STRING	否	body
        personNum	购买人数	STRING	否	body

    Returns:
        参数名称	参数描述	参数类型	是否必填
        view_data	卡片信息	OBJECT	否
        data	文本信息	OBJECT	否
        cmd	跳转信息	OBJECT	否
    """
    res = "{\"view_data\":\"当前景区门票购买\", \"data\":\"交易成功\"}"

    return res
