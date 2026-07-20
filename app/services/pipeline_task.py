"""Pipeline task management — file-based task state for multi-worker safety.

Each task lives in ``data/outputs/eval_tasks/<eval_task_id>/`` with a
``task.json`` that tracks status, stage, params, artifact paths, and errors.
Multiple uvicorn workers can safely read task.json; only the executing
worker writes it (one task = one async coroutine on one worker).
"""

import asyncio
import json
import logging
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from app.paths import DATA_DIR

logger = logging.getLogger(__name__)

TASKS_DIR = DATA_DIR / "outputs" / "eval_tasks"


def generate_task_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = secrets.token_hex(3)
    return f"evt_{ts}_{rand}"


def task_dir_of(task_id: str) -> Path:
    return TASKS_DIR / task_id


def create_task(task_id: str, params: dict[str, Any]) -> Path:
    task_dir = task_dir_of(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "simulate").mkdir(exist_ok=True)
    (task_dir / "evaluate").mkdir(exist_ok=True)
    (task_dir / "report").mkdir(exist_ok=True)

    now = datetime.now().isoformat(timespec="seconds")
    state = {
        "eval_task_id": task_id,
        "status": "running",
        "stage": "pending",
        "created_at": now,
        "updated_at": now,
        "params": params,
        "artifacts": {},
        "error": None,
    }
    _write_state(task_dir, state)
    return task_dir


def read_task(task_id: str) -> dict[str, Any] | None:
    task_dir = task_dir_of(task_id)
    json_path = task_dir / "task.json"
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_task(task_dir: Path, **updates: Any) -> None:
    json_path = task_dir / "task.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    if "artifacts" in updates:
        state.setdefault("artifacts", {}).update(updates.pop("artifacts"))
    state.update(updates)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_state(task_dir, state)


def _write_state(task_dir: Path, state: dict) -> None:
    json_path = task_dir / "task.json"
    tmp = json_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(json_path)


async def execute_pipeline(task_dir: Path, task_id: str, params: dict) -> None:
    """Run the full pipeline asynchronously. Called via asyncio.create_task()."""
    from app.scripts.run_pipeline import stage_simulate, stage_evaluate
    from app.services.eval_summary_service import summarize
    from app.services.eval_compare_service import compare_grouped

    input_xlsx = task_dir / "input.xlsx"
    model = params.get("model") or ""
    models_str = params.get("models") or ""
    is_multi = bool(models_str)

    try:
        if is_multi:
            await _execute_multi_model(
                task_dir, task_id, params, input_xlsx, models_str,
            )
        else:
            await _execute_single_model(
                task_dir, task_id, params, input_xlsx, model,
            )
    except Exception as exc:
        logger.exception("[%s] pipeline failed", task_id)
        update_task(task_dir, status="failed", error=f"{type(exc).__name__}: {exc}")


async def _execute_single_model(
    task_dir: Path, task_id: str, params: dict,
    input_xlsx: Path, model: str,
) -> None:
    from app.scripts.run_pipeline import stage_simulate, stage_evaluate
    from app.services.eval_summary_service import summarize
    from app.config import settings

    effective_model = model or settings.MODEL_NAME
    sim_dir = task_dir / "simulate"
    eval_dir = task_dir / "evaluate"
    report_dir = task_dir / "report"

    # Stage 1
    update_task(task_dir, stage="simulate")
    sim_csv = await stage_simulate(
        input_xlsx.name,
        model=effective_model,
        suffix=effective_model,
        max_turns=params.get("max_turns", 20),
        concurrency=params.get("concurrency", 4),
        sandbox=params.get("sandbox", False),
        streaming=params.get("streaming", False),
        thinking=params.get("thinking", False),
        output_dir=sim_dir,
        input_dir=task_dir,
    )
    update_task(task_dir, artifacts={"simulate_csv": str(sim_csv.relative_to(task_dir))})

    # Stage 2
    update_task(task_dir, stage="evaluate")
    eval_csv = await stage_evaluate(
        sim_csv,
        cases_input=input_xlsx,
        suffix=effective_model,
        eval_concurrency=params.get("eval_concurrency", 2),
        language=params.get("language", "chinese"),
        enable_verification=params.get("enable_verification", False),
        enable_meta_judge=params.get("enable_meta_judge", False),
        output_dir=eval_dir,
    )
    update_task(task_dir, artifacts={"evaluate_csv": str(eval_csv.relative_to(task_dir))})

    # Stage 3
    update_task(task_dir, stage="report")
    products = summarize(
        xlsx_path=eval_csv,
        output_dir=report_dir,
        top_n=5,
        visualize="both",
    )
    report_path = products.html_path or products.txt_path
    update_task(
        task_dir,
        stage="done",
        status="completed",
        artifacts={"report_path": str(report_path.relative_to(task_dir)) if report_path else "report/"},
    )
    logger.info("[%s] pipeline completed", task_id)


async def _execute_multi_model(
    task_dir: Path, task_id: str, params: dict,
    input_xlsx: Path, models_str: str,
) -> None:
    from app.scripts.run_pipeline import stage_simulate, stage_evaluate
    from app.services.eval_compare_service import compare_grouped

    models = [m.strip() for m in models_str.split(",") if m.strip()]
    sim_dir = task_dir / "simulate"
    eval_dir = task_dir / "evaluate"
    report_dir = task_dir / "report"

    eval_csvs: dict[str, list[Path]] = {}

    for i, model in enumerate(models, 1):
        logger.info("[%s] model %d/%d: %s", task_id, i, len(models), model)
        update_task(task_dir, stage=f"simulate ({model})")

        sim_csv = await stage_simulate(
            input_xlsx.name,
            model=model,
            suffix=model,
            max_turns=params.get("max_turns", 20),
            concurrency=params.get("concurrency", 4),
            sandbox=params.get("sandbox", False),
            streaming=params.get("streaming", False),
            thinking=params.get("thinking", False),
                output_dir=sim_dir,
            input_dir=task_dir,
        )
        update_task(task_dir, artifacts={f"simulate_csv_{model}": str(sim_csv.relative_to(task_dir))})

        update_task(task_dir, stage=f"evaluate ({model})")
        eval_csv = await stage_evaluate(
            sim_csv,
            cases_input=input_xlsx,
            suffix=model,
            eval_concurrency=params.get("eval_concurrency", 2),
            language=params.get("language", "chinese"),
            enable_verification=params.get("enable_verification", False),
            enable_meta_judge=params.get("enable_meta_judge", False),
            output_dir=eval_dir,
        )
        eval_csvs.setdefault(model, []).append(eval_csv)
        update_task(task_dir, artifacts={f"evaluate_csv_{model}": str(eval_csv.relative_to(task_dir))})

    update_task(task_dir, stage="report")
    products = compare_grouped(eval_csvs, report_dir)
    update_task(
        task_dir,
        stage="done",
        status="completed",
        artifacts={
            "report_md": str(products.md_path.relative_to(task_dir)),
            "report_json": str(products.json_path.relative_to(task_dir)),
        },
    )
    logger.info("[%s] multi-model pipeline completed", task_id)
