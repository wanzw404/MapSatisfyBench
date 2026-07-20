from typing import Literal
from pydantic import BaseModel, Field


class EvaluateCase(BaseModel):
    """单条评测用例（从 Excel 读取）"""
    query: str = Field(..., description="用户查询语句")
    location: str | None = Field(default=None, description="上下文位置信息")
    tool: str | None = Field(default=None, description="期望调用的工具名")
    expected: str | None = Field(default=None, description="期望输出（可选）")


class EvaluateResult(BaseModel):
    """单条评测结果"""
    conversation_id: str = Field(..., description="唯一会话 ID")
    query: str = Field(..., description="用户查询语句")
    location: str | None = Field(default=None, description="上下文位置信息")
    tool: str | None = Field(default=None, description="期望调用的工具名")
    expected: str | None = Field(default=None, description="期望输出")
    status: Literal["success", "error", "timeout"] = Field(..., description="执行状态")
    final_response: str = Field(default="", description="Agent 最终回复")
    tool_calls: list[dict] = Field(default_factory=list, description="实际调用的工具列表")
    execution_time_ms: int = Field(default=0, description="执行耗时（毫秒）")
    error_message: str | None = Field(default=None, description="错误信息")


class UploadResponse(BaseModel):
    """文件上传响应"""
    success: bool = Field(default=True)
    filename: str = Field(..., description="保存后的文件名")
    saved_path: str = Field(..., description="文件保存绝对路径")
    row_count: int = Field(..., description="读取到的用例行数")


class RunRequest(BaseModel):
    """执行评测请求"""
    filename: str = Field(..., description="/data/inputs 下的文件名")
    thread_id_prefix: str | None = Field(default=None, description="thread_id 前缀（可选）")
    max_concurrency: int = Field(
        default=4, ge=1, le=32,
        description="case 间并发上限；case 内永远串行。默认 4，1 退化为完全串行。",
    )


class RunResponse(BaseModel):
    """执行评测响应"""
    success: bool = Field(default=True)
    input_file: str = Field(..., description="输入文件名")
    output_file: str = Field(..., description="输出结果文件路径")
    total: int = Field(..., description="总用例数")
    success_count: int = Field(..., description="成功数")
    error_count: int = Field(..., description="失败数")
    results: list[EvaluateResult] = Field(default_factory=list, description="详细结果列表")
