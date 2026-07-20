import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
from app.config import USER_SIMULATOR_MODEL, settings
from app.core.agent.llm.openai_chat import OpenAICompatProvider
from app.core.simulator.user_simulator import UserSimulator
from app.schemas.dialogue_simulator import DialogueCase


def build_user_simulator(
    case: DialogueCase,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> UserSimulator:
    """根据用例构建 UserSimulator 实例。

    Args:
        case: 评测用例
        base_url: LLM API base_url，默认从 settings 读取
        api_key: LLM API key，默认从 settings 读取
        model: 已废弃 — UserSimulator 强制锁定 ``USER_SIMULATOR_MODEL``，
            传值会被忽略并打 WARNING。被测 agent 在切换模型评测时，
            用户仿真侧必须保持稳定才不污染对照实验。

    Returns:
        初始化好的 UserSimulator
    """
    if model and model != USER_SIMULATOR_MODEL:
        logger.warning(
            "[UserSimulator] 收到 model=%r，已忽略；强制锁定到 USER_SIMULATOR_MODEL=%r",
            model, USER_SIMULATOR_MODEL,
        )

    # 用户仿真专用凭证：BASE_URL_USER / AI_STUDIO_TOKEN_USER（与 agent / Judge 解耦）。
    # 严格模式：缺失即报错，避免被测 agent 切网关时 user simulator 隐式漂移。
    resolved_base_url = base_url or settings.BASE_URL_USER
    resolved_api_key = api_key or settings.AI_STUDIO_TOKEN_USER
    if not resolved_base_url:
        raise RuntimeError(
            "BASE_URL_USER 未配置（.env 或 Diamond），UserSimulator 拒绝隐式回退"
            " BASE_URL，请显式设置以保证用户仿真侧凭证独立。"
        )
    if not resolved_api_key:
        raise RuntimeError(
            "AI_STUDIO_TOKEN_USER 未配置（.env 或 Diamond），UserSimulator 拒绝"
            "隐式回退 AI_STUDIO_TOKEN，请显式设置以保证用户仿真侧凭证独立。"
        )
    llm = OpenAICompatProvider(
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        model=USER_SIMULATOR_MODEL,
    )

    # 从 context JSON 中解析 user_loc_name 作为 current_location
    current_location = case.location or ""
    if case.context:
        try:
            ctx = json.loads(case.context)
            current_location = ctx.get("user_loc_name", current_location)
        except json.JSONDecodeError:
            pass

    simulator_input = {
        "current_time": case.time or datetime.now().strftime("%Y-%m-%d %H:%M"),
        "current_location": current_location,
        "persona": case.persona or "",
        "query": case.query,
        "full_intent": case.full_intent or case.query,
        "language": "chinese",
    }
    logger.info(f"[UserSimulator] 入口入参 | {json.dumps(simulator_input, ensure_ascii=False)}")
    case.user_simulator_input = json.dumps(simulator_input, ensure_ascii=False)

    return UserSimulator(
        llm_provider=llm,
        current_time=simulator_input["current_time"],
        current_location=simulator_input["current_location"],
        persona=simulator_input["persona"],
        query=simulator_input["query"],
        full_intent=simulator_input["full_intent"],
        language=simulator_input["language"],
    )
