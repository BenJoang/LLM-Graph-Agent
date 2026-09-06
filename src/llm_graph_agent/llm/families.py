"""模型家族抽象。每种模型家族是一个类，封装自己的 thinking 参数逻辑。

设计说明：
- 每个模型家族继承 ModelFamily，实现 apply_thinking。
- 注册到 FAMILIES 字典，按模型名前缀匹配。
- 新增模型家族：写一个子类 + 注册一行，不需要改 build_chat_model。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class ModelFamily(ABC):
    """模型家族基类。每个家族定义自己的 thinking 参数设置方式。"""

    # 模型名前缀，用于匹配家族（子类覆盖）
    prefixes: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def apply_thinking(
        self,
        extra_body: dict[str, Any],
        thinking: bool,
    ) -> None:
        """把 thinking 设置写入 extra_body。"""
        raise NotImplementedError

    @classmethod
    def matches(cls, model_name: str) -> bool:
        """判断模型名是否属于本家族。"""
        lowered = model_name.lower()
        return any(lowered.startswith(p) for p in cls.prefixes)


class QwenFamily(ModelFamily):
    """Qwen 系列：通过 chat_template_kwargs.enable_thinking 控制。"""

    prefixes: ClassVar[tuple[str, ...]] = ("qwen",)

    def apply_thinking(
        self,
        extra_body: dict[str, Any],
        thinking: bool,
    ) -> None:
        chat_template_kwargs = extra_body.setdefault(
            "chat_template_kwargs", {}
        )
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError(
                "extra_body.chat_template_kwargs 必须是一个对象"
            )
        chat_template_kwargs["enable_thinking"] = thinking


class DeepSeekFamily(ModelFamily):
    """DeepSeek 系列：通过 extra_body.thinking.type 控制。"""

    prefixes: ClassVar[tuple[str, ...]] = ("deepseek",)

    def apply_thinking(
        self,
        extra_body: dict[str, Any],
        thinking: bool,
    ) -> None:
        extra_body["thinking"] = {
            "type": "enabled" if thinking else "disabled",
        }


class OpenAICompatibleFamily(ModelFamily):
    """通用 OpenAI 兼容：把 thinking 放到 extra_body.thinking（带 type 的格式）。

    很多兼容服务（vllm、ollama 等）接受这种格式。作为兜底家族，matches 永远 False，
    需要显式注册前缀或用 get_family 指定。
    """

    prefixes: ClassVar[tuple[str, ...]] = ()

    def apply_thinking(
        self,
        extra_body: dict[str, Any],
        thinking: bool,
    ) -> None:
        extra_body["thinking"] = {
            "type": "enabled" if thinking else "disabled",
        }


# 注册表：按顺序匹配，先注册的先匹配
FAMILIES: list[ModelFamily] = [
    QwenFamily(),
    DeepSeekFamily(),
    OpenAICompatibleFamily(),
]

# 便捷查找：按前缀
_PREFIX_INDEX: dict[str, ModelFamily] = {}
for _family in FAMILIES:
    for _prefix in _family.prefixes:
        _PREFIX_INDEX[_prefix] = _family


def get_family(model_name: str) -> ModelFamily:
    """根据模型名返回对应家族；未知家族抛异常。"""
    lowered = model_name.lower()
    for family in FAMILIES:
        if family.matches(lowered):
            return family
    raise ValueError(f"未知模型家族：{model_name}")


def register_family(family: ModelFamily) -> None:
    """运行时注册新家族（插件式扩展）。"""
    FAMILIES.append(family)
    for prefix in family.prefixes:
        _PREFIX_INDEX[prefix] = family
