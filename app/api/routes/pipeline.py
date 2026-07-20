"""Pipeline router — async eval pipeline with file-based task tracking.

POST /api/v1/pipeline/run     — Upload xlsx + params, kick off async pipeline
GET  /api/v1/pipeline/task/{id} — Query status or download artifacts
"""

import asyncio
import io
import logging
import zipfile
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.services.pipeline_task import (
    create_task,
    execute_pipeline,
    generate_task_id,
    read_task,
    task_dir_of,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["pipeline"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PipelineRunResponse(BaseModel):
    eval_task_id: str
    status: str
    message: str


class PipelineTaskResponse(BaseModel):
    eval_task_id: str
    status: str
    stage: str
    created_at: str
    updated_at: str
    params: dict[str, Any]
    artifacts: dict[str, str]
    error: str | None = None


# ---------------------------------------------------------------------------
# POST /api/v1/pipeline/run
# ---------------------------------------------------------------------------


@router.post(
    "/pipeline/run",
    response_model=PipelineRunResponse,
    summary="Upload xlsx and start async eval pipeline",
)
async def pipeline_run(
    file: UploadFile = File(..., description="xlsx 用例文件"),
    model: str = Form(default="", description="单模型名（与 models 互斥）"),
    models: str = Form(default="", description="逗号分隔多模型名（与 model 互斥）"),
    max_turns: int = Form(default=20, ge=1, le=100),
    concurrency: int = Form(default=4, ge=1, le=32),
    eval_concurrency: int = Form(default=2, ge=1, le=16),
    sandbox: bool = Form(default=False),
    streaming: bool = Form(default=False),
    thinking: bool = Form(default=False),
    language: str = Form(default="chinese"),
    enable_verification: bool = Form(default=False),
    enable_meta_judge: bool = Form(default=False),
) -> PipelineRunResponse:
    if model and models:
        raise HTTPException(400, "model 与 models 互斥，不可同时传")
    if not file.filename or not file.filename.endswith(".xlsx"):
        raise HTTPException(400, "仅支持 .xlsx 文件")

    task_id = generate_task_id()
    params = {
        "model": model,
        "models": models,
        "max_turns": max_turns,
        "concurrency": concurrency,
        "eval_concurrency": eval_concurrency,
        "sandbox": sandbox,
        "streaming": streaming,
        "thinking": thinking,
        "language": language,
        "enable_verification": enable_verification,
        "enable_meta_judge": enable_meta_judge,
        "original_filename": file.filename,
    }

    task_dir = create_task(task_id, params)

    input_path = task_dir / "input.xlsx"
    content = await file.read()
    input_path.write_bytes(content)

    asyncio.create_task(execute_pipeline(task_dir, task_id, params))

    return PipelineRunResponse(
        eval_task_id=task_id,
        status="running",
        message="流水线已启动",
    )


# ---------------------------------------------------------------------------
# GET /api/v1/pipeline/task/{eval_task_id}
# ---------------------------------------------------------------------------


@router.get(
    "/pipeline/task/{eval_task_id}",
    summary="Query task status or download artifacts",
)
async def pipeline_task(
    eval_task_id: str,
    download: str | None = Query(
        default=None,
        description="要下载的产物: input / simulate / evaluate / report",
    ),
):
    state = read_task(eval_task_id)
    if state is None:
        raise HTTPException(404, f"任务不存在: {eval_task_id}")

    if download is None:
        return PipelineTaskResponse(**state)

    task_dir = task_dir_of(eval_task_id)
    artifacts = state.get("artifacts", {})

    if download == "input":
        fp = task_dir / "input.xlsx"
        if not fp.exists():
            raise HTTPException(404, "输入文件不存在")
        return FileResponse(
            fp, filename=state.get("params", {}).get("original_filename", "input.xlsx"),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if download == "simulate":
        csv_rel = _find_artifact(artifacts, "simulate_csv", prefix="simulate_csv_")
        if not csv_rel:
            raise HTTPException(404, "仿真产物尚未生成")
        fp = task_dir / csv_rel
        if not fp.exists():
            raise HTTPException(404, f"仿真文件不存在: {csv_rel}")
        return FileResponse(fp, filename=fp.name, media_type="text/csv")

    if download == "evaluate":
        csv_rel = _find_artifact(artifacts, "evaluate_csv", prefix="evaluate_csv_")
        if not csv_rel:
            raise HTTPException(404, "评分产物尚未生成")
        fp = task_dir / csv_rel
        if not fp.exists():
            raise HTTPException(404, f"评分文件不存在: {csv_rel}")
        return FileResponse(fp, filename=fp.name, media_type="text/csv")

    if download == "report":
        report_dir = task_dir / "report"
        if not report_dir.exists() or not any(report_dir.iterdir()):
            raise HTTPException(404, "报告产物尚未生成")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in sorted(report_dir.rglob("*")):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(report_dir))
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="report_{eval_task_id}.zip"'},
        )

    raise HTTPException(400, f"无效的 download 值: {download}（可选: input/simulate/evaluate/report）")


def _find_artifact(artifacts: dict, primary_key: str, prefix: str) -> str | None:
    """Find artifact path — single-model uses primary_key, multi-model has prefixed keys."""
    if primary_key in artifacts:
        return artifacts[primary_key]
    matches = [v for k, v in artifacts.items() if k.startswith(prefix)]
    return matches[0] if matches else None
