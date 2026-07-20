"""Sequential navigation — get_sequential_navigation 工具定义（仅沙箱 / 工具仿真）."""
from typing import Any, Dict

from langchain_core.tools import tool

from .base import safe_tool, sandbox_cache


@tool
@safe_tool
@sandbox_cache
def get_sequential_navigation(
    points: list | str | None = None,
    sandbox: bool = False,
):
    """顺序路线信息查询，获取多个点按顺序的各段路线规划结果及用时。

    Args:
        - points (list | str | None) = None: List[Dict[str, Any]]): 点位列表，每个点位为一个字典，包含以下字段：
            - lat (float): 纬度，必填。
            - lon (float): 经度，必填。
            - name (str, optional): 点位名称，可选，用于前端展示。

    Returns:
        Dict[str, Any]: 包含所有路段的路线规划信息的字典，结构如下：
        {
            "success": bool,  # 总体是否成功
            "total_segments": int,  # 总段数
            "segments": [  # 各段路线信息列表
                {
                    "segment_index": int,  # 段序号（从0开始）
                    "start_point": {  # 起点信息
                        "lon": float,
                        "lat": float,
                        "name": str
                    },
                    "end_point": {  # 终点信息
                        "lon": float,
                        "lat": float,
                        "name": str,
                        "poi_id": str
                    },
                    "route_result": Any,  # 合并后的路线结果：如果navigation_result是列表且
                                           # taxi_result是字符串，则合并两者；否则仅保留一个结果
                    "success": bool,  # 本段是否查询成功
                    "error": str,  # 如果失败，记录错误信息
                    "navigation_error": str,  # 导航查询错误信息（如果有）
                    "taxi_error": str  # 打车查询错误信息（如果有）
                },
                ...
            ],
            "error": str  # 如果有总体错误，记录错误信息
        }
    """
    raise NotImplementedError(
        "get_sequential_navigation 业务逻辑已移除，仅支持 sandbox=True 或沙箱缓存命中"
    )
