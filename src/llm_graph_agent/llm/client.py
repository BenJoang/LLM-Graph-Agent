"""模型客户端连接。只负责创建 OpenAI / ChatOpenAI 实例，不含配置加载和日志。

thinking 参数差异由 families.py 的模型家族抽象处理。
"""
from __future__ import annotations

from copy import deepcopy

from openai import OpenAI, AsyncOpenAI
from langchain_openai import ChatOpenAI

from llm_graph_agent.llm.families import get_family


def build_client(profile: dict, timeout: int = 60) -> OpenAI:
    """构建同步 OpenAI 客户端。"""
    return OpenAI(
        base_url=profile["base_url"],
        api_key=profile["api_key"],
        timeout=timeout,
    )


def build_async_client(profile: dict, timeout: int = 60) -> AsyncOpenAI:
    """构建异步 OpenAI 客户端。"""
    return AsyncOpenAI(
        base_url=profile["base_url"],
        api_key=profile["api_key"],
        timeout=timeout,
    )


def build_chat_model(profile: dict, thinking: bool | None = True) -> ChatOpenAI:
    """构建 LangChain ChatOpenAI，处理 thinking 参数差异。"""
    generation = profile.get("generation", {})
    kwargs = {
        "model": profile["model"],
        "base_url": profile["base_url"],
        "api_key": profile["api_key"],
    }

    # OpenAI/ChatOpenAI 标准生成参数
    for key in ("temperature", "top_p", "presence_penalty"):
        value = generation.get(key)
        if value is not None:  # 配置为 null 时不向模型发送
            kwargs[key] = value

    # 必须复制，否则可能修改 profile 中的原始配置
    extra_body = deepcopy(profile.get("extra_body") or {})

    # 判断 profile 是否已显式配置 thinking（不覆盖用户设置）
    if _profile_explicitly_configures_thinking(extra_body):
        thinking_configured = True
    else:
        thinking_configured = False

    # profile 没配置时，才使用 graph 传入的 thinking
    if thinking is not None and not thinking_configured:
        family = get_family(profile["model"])
        family.apply_thinking(extra_body, thinking)

    if extra_body:
        kwargs["extra_body"] = extra_body

    return ChatOpenAI(**kwargs)


def _profile_explicitly_configures_thinking(extra_body: dict) -> bool:
    """判断 extra_body 里是否已经显式配置了 thinking（qwen 或 deepseek 格式）。"""
    chat_template_kwargs = extra_body.get("chat_template_kwargs")
    qwen_configured = (
        isinstance(chat_template_kwargs, dict)
        and "enable_thinking" in chat_template_kwargs
    )
    deepseek_configured = "thinking" in extra_body
    return qwen_configured or deepseek_configured


def test_connection(client: OpenAI) -> bool:
    """测试模型连接是否可用。"""
    try:
        client.models.list()
        return True
    except Exception as e:
        print("连接失败")
        print("错误类型:", type(e).__name__)
        print("错误信息:", e)
        return False
