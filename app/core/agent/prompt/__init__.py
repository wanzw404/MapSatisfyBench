"""Centralized prompt templates.

所有 prompt 均以 YAML 格式存放在本目录下（*.yaml）。
其他地方通过 `get_prompts()` 或模块级变量读取指定 prompt 内容。

用法示例:
    from app.core.agent.prompt import get_prompts, PLANNER_PROMPT

    # 方式1：通过 get_prompts 动态访问（支持多语言切换）
    prompts = get_prompts("chinese")
    planner = prompts.planner          # 读取 planner.yaml 的 chinese 内容
    reporter = prompts.reporter        # 读取 reporter.yaml 的 chinese 内容
    judge = prompts.judge              # 读取 judge.yaml 的 chinese 内容
    user_system = prompts.user_system_prompt

    # 方式2：直接导入模块级变量（默认语言，向后兼容）
    text = PLANNER_PROMPT
"""

import os
import yaml
from typing import Dict, Optional


def _load_prompts(language: str) -> Dict[str, str]:
    """加载 prompts 目录下所有 .yaml 文件，返回 {prompt_name: content}。"""
    prompts_dir = os.path.dirname(__file__)
    prompts: Dict[str, str] = {}

    for filename in os.listdir(prompts_dir):
        if not filename.endswith(".yaml"):
            continue

        filepath = os.path.join(prompts_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            prompt_name = filename.replace(".yaml", "")
            if language in data:
                prompts[prompt_name] = data[language]
            else:
                raise ValueError(
                    f"Language '{language}' not found in {filename}. "
                    f"Available languages: {list(data.keys())}"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt file {filename}: {e}")

    return prompts


class Prompts:
    """
    提示词管理器

    功能：
    - 自动加载 prompts 目录下的所有 .yaml 文件
    - 根据语言设置提取对应的提示词内容
    - 支持运行时动态切换语言
    - 通过属性访问方式获取提示词（如 prompts.user_system_prompt）
    """

    def __init__(self, language: str = "chinese"):
        self.language = language
        self._prompts: Dict[str, str] = _load_prompts(language)

    def __getattr__(self, name: str) -> str:
        """动态访问提示词"""
        if name in self._prompts:
            return self._prompts[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'. "
            f"Available prompts: {list(self._prompts.keys())}"
        )

    def set_language(self, language: str):
        """切换语言并重新加载提示词"""
        if self.language != language:
            self.language = language
            self._prompts = _load_prompts(language)

    def get_available_prompts(self) -> list:
        """获取可用的提示词名称列表"""
        return list(self._prompts.keys())

    def get(self, name: str) -> str:
        """通过名称获取提示词内容。"""
        if name in self._prompts:
            return self._prompts[name]
        raise KeyError(
            f"Prompt '{name}' not found. "
            f"Available prompts: {list(self._prompts.keys())}"
        )


# ── 全局单例实例 ──
_prompts_instance: Optional[Prompts] = None


def get_prompts(language: Optional[str] = None) -> Prompts:
    """
    获取提示词实例

    Args:
        language: 语言设置，如果为 None 则使用默认语言

    Returns:
        Prompts 实例
    """
    global _prompts_instance

    if language is None:
        if _prompts_instance is None:
            from app.config import settings

            _prompts_instance = Prompts(settings.DEFAULT_LANGUAGE)
        return _prompts_instance
    else:
        return Prompts(language)


# ── 模块级变量：默认语言（chinese）的 prompt 内容，向后兼容 ──
_default_prompts = _load_prompts("chinese")

PLANNER_PROMPT = _default_prompts.get("planner", "")
REPORTER_PROMPT = _default_prompts.get("reporter", "")
JUDGE_PROMPT = _default_prompts.get("judge", "")

__all__ = [
    "Prompts",
    "get_prompts",
    "PLANNER_PROMPT",
    "REPORTER_PROMPT",
    "JUDGE_PROMPT",
]
