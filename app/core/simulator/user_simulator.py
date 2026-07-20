"""
标准用户模拟器实现
支持空间决策场景的用户模拟，无工具调用能力。

核心逻辑：
  每次接收完整对话历史（list[dict]），结合系统提示（当前时间/位置/用户画像/完整意图），
  调用 LLM 生成下一条用户回复。
"""
from typing import Optional
from loguru import logger

from .base import BaseUser, STOP
from app.core.agent.prompt import get_prompts
from ..agent.llm.base import BaseLLMProvider


class UserSimulator(BaseUser):
    """
    基于 LLM 的标准用户模拟器

    核心特性：
    - 提示词模板化：从 YAML 加载系统提示，支持多语言
    - 无工具调用：用户没有任何执行能力，只能对话
    - 历史驱动：每次接收完整对话历史，生成下一条回复
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        current_time: str = "",
        current_location: str = "",
        persona: Optional[str] = None,
        query: str = "",
        full_intent: str = "",
        language: str = "chinese",
    ):
        """
        初始化用户模拟器

        Args:
            llm_provider: 已初始化的 LLM 提供者实例（BaseLLMProvider 子类）
            current_time: 当前时间（如 "2026-05-08 14:30"）
            current_location: 当前位置（如 "北京市朝阳区"）
            persona: 用户画像，描述用户的性格、背景等信息（可为空）
            query: 用户的初始查询（对话第一轮的原始发言，作为系统提示的参考上下文）
            full_intent: 用户的完整意图（内心真实需求，含隐性偏好和硬性约束）
            language: 语言设置（"chinese" 或 "english"）
        """
        self.llm_provider = llm_provider
        self.current_time = current_time
        self.current_location = current_location
        self.persona = persona or ""
        self.query = query
        self.full_intent = full_intent
        self.language = language
        self.last_llm_messages: list[dict] = []
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0
        self.last_response_id: str = ""

    @property
    def system_prompt(self) -> str:
        """
        构建系统提示词。

        从 YAML 文件加载模板并填充上下文参数。
        注意：{query} 作为参考上下文保留在系统提示中；
             实际对话历史（含首轮 query）以 messages 形式独立传入 LLM。

        Returns:
            完整的系统提示词字符串
        """
        prompts = get_prompts(self.language)
        try:
            return prompts.user_system_prompt.format(
                current_time=self.current_time,
                current_location=self.current_location,
                persona=self.persona,
                query=self.query,
                full_intent=self.full_intent,
            )
        except KeyError as e:
            raise ValueError(f"Missing placeholder in prompt template: {e}")

    async def generate_next_message(
        self,
        conversation_history: list[dict[str, str]],
    ) -> str:
        """
        根据完整对话历史生成下一条用户消息。

        流程：
        1. 将系统提示拼接到对话历史前面
        2. 调用 LLM 生成回复（无工具调用）
        3. 返回生成的文本内容

        Args:
            conversation_history: 当前完整对话历史，格式为
                [
                    {"role": "user",      "content": "初始 query ..."},
                    {"role": "assistant", "content": "助手回复 ..."},
                    {"role": "user",      "content": "用户第一次回复 ..."},
                    ...
                ]
            正确的调用时机：最后一条消息必须是 assistant，即助手已回复后才轮到用户。

        Returns:
            下一条用户消息的文本内容（可能包含 "[Finish Conversation]" 终止标记）
        """
        # ── 前置校验：最后一条消息必须是 assistant ──────────────────────────
        # 如果最后一条是 user，说明助手尚未回复，不应该再生成用户消息，直接终止。
        if conversation_history and conversation_history[-1].get("role") == "user":
            logger.warning(
                "[UserSimulator] Last message is from 'user', assistant has not replied yet. "
                "Returning stop marker directly."
            )
            return STOP

        # ── 硬性上限：对话总轮数达到 10 条则强制终止 ────────────────────────
        # turn_count = len(conversation_history)（user + assistant 累计消息数）。
        # 防止 LLM 忽略 prompt 中的终止条件导致对话无限进行。
        turn_count = len(conversation_history)
        if turn_count >= 20:
            logger.info(
                f"[UserSimulator] Reached max turn_count={turn_count} (>=20), "
                "forcing termination."
            )
            return STOP

        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + conversation_history

        self.last_llm_messages = messages

        logger.debug(
            f"[UserSimulator] Calling LLM: 1 system + {len(conversation_history)} history messages"
        )

        # UserSimulator 强制 temperature=1.0：用户仿真需要稳定的随机性 baseline，
        # 不随被测 agent 模型切换而变化（对照实验前提）。同时避免依赖各模型 server
        # 的默认温度（不同路由后端默认值不一致）。
        response = await self.llm_provider.achat(messages, temperature=1.0)
        content = response.content.strip()
        self.last_prompt_tokens = response.prompt_tokens
        self.last_completion_tokens = response.completion_tokens
        self.last_response_id = response.response_id or ""

        logger.debug(f"[UserSimulator] Generated reply: {content[:120]}...")
        return content
