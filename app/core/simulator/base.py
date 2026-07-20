"""
用户模拟器抽象基类
定义统一的接口规范：基于对话历史（list[dict]）生成下一条用户消息
"""
from abc import ABC, abstractmethod

# 终止标记常量
STOP = "[Finish Conversation]"


class BaseUser(ABC):
    """
    用户模拟器基类

    接口约定：
    - generate_next_message: 接收完整对话历史，生成下一条用户消息内容
    - is_stop: 判断生成内容是否包含终止标记
    """

    @abstractmethod
    async def generate_next_message(
        self,
        conversation_history: list[dict[str, str]],
    ) -> str:
        """
        根据完整对话历史生成下一条用户消息。

        Args:
            conversation_history: 当前完整对话历史，格式为
                [
                    {"role": "user",      "content": "初始 query ..."},
                    {"role": "assistant", "content": "助手回复 ..."},
                    {"role": "user",      "content": "用户第一次回复 ..."},
                    ...
                ]

        Returns:
            下一条用户消息的文本内容
        """
        pass

    @classmethod
    def is_stop(cls, content: str) -> bool:
        """
        判断消息内容是否包含终止标记。

        Args:
            content: 用户消息文本

        Returns:
            True 表示应终止对话
        """
        if not content:
            return False
        return STOP in content
