from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator


class Settings(BaseSettings):
    """
    Pydantic Settings类，会默认将环境变量映射到类属性中，大小写不敏感
    该类中的属性值在运行时会被环境变量替换
    优先级：环境变量 > .env文件 > 默认值
    """
    #这里主要是做一些Settings类相关的设置
    model_config = SettingsConfigDict(
        env_file=".env",           # 默认从 .env 文件读取配置，但不会覆盖已有环境变量配置，所以如果从diamond中load_dotenv，会覆盖.env文件中的配置
        env_ignore_empty=True,     # 忽略空的环境变量
        extra="ignore",            # 忽略额外的字段
    )

    MODEL_NAME: str = "gpt-5.3-chat-0303-global"

    # 提示词默认语言（对应 prompt yaml 中的语言键，如 chinese/english）
    DEFAULT_LANGUAGE: str = "chinese"

    # LLM API Token
    AI_STUDIO_TOKEN: str = ""

    # 调用模型时的 BASE_URL，自行切换为百炼/Whale 等
    BASE_URL: str = ""

    # 用户仿真模块（UserSimulator）专用 LLM 凭证：与被测 agent / Judge 解耦，
    # 避免被测 agent 切到 amap-magic-tool 网关时，user_simulator 也跟着漂。
    # 严格策略：未配置时 user_simulator_factory 会 raise，强制运维明确配置。
    # 默认空字符串而非合理值，确保"忘记设置"不会被静默走 fallback。
    AI_STUDIO_TOKEN_USER: str = ""
    BASE_URL_USER: str = ""

    #embedding模型参数
    EMBEDDING_MODEL_URL: Optional[str] = None
    EMBEDDING_MODEL_NAME: Optional[str] = None
    EMBEDDING_DIMENSIONS: Optional[int] = 0
    EMBEDDING_ENCODING_FORMAT: Optional[str] = None

#从diamond中加载环境变量，覆盖.env文件中的配置
from app.diamond import env_content
from dotenv import load_dotenv
from io import StringIO

load_dotenv(stream=StringIO(env_content))

# 本地环境：diamond 返回空时回退加载 .env 到 os.environ，确保 LLM key 等可用
if not env_content.strip():
    load_dotenv(".env")


settings = Settings()  # type: ignore


# ─────────────────────────────────────────────────────────────────────
# 评测体系强制锁定的模型（**不**走 Settings/.env/Diamond override 通道）
# ─────────────────────────────────────────────────────────────────────
# 设计意图：被测 agent 切换不同模型评测时，user 仿真模型 / 评分模型必须
# **稳定不变**，否则对照实验的两侧（agent 与 user/judge）同时漂移，跑出
# 来的指标既归因不到 agent 也归因不到 user，污染整个评测体系。
#
# 这里**故意**选择模块级常量而非 Settings 字段：
#   - Settings 字段会被 .env / Diamond / 环境变量随便覆盖，不符合「锁死」
#   - 模块级常量需要修改源码 + code review 才能改，符合「关键评测基线」
#     的变更门槛
#
# 切换前需评估：新模型是否流式 OK / tool_use 适配 OK /
# pricing 是否可接受 / 与历史 baseline 的可比性。
USER_SIMULATOR_MODEL = "gpt-5.3-chat-0303-global"
JUDGE_MODEL = "gpt-5.3-chat-0303-global"
# Tool 响应仿造模型：strict 沙箱模式下，沙箱未命中时由它仿造一条工具响应；
# 与 USER_SIMULATOR_MODEL / JUDGE_MODEL 平级，独立可调，避免被测 agent 换模型
# 时仿造侧也漂移。
TOOL_SIMULATOR_MODEL = "gpt-5.3-chat-0303-global"
