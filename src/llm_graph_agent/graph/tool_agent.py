"""通用工具 Agent 图（收敛后的单一循环）。

原项目有 tool_agent / tts / qq_main 等多个功能几乎相同的图，只是工具集
与 profile 不同。收敛为这一个 build_graph：通过 tool_names + 配置驱动，
即可获得任意工具组合的 Agent 循环。

结构：
    assistant → (tools_condition) → tools → assistant … → END

- 压缩/重试/上下文组装收敛在 common.CompressedAssistant
- 工具节点是 turn-aware 的（工具消息会打上 turn_id 标签）
"""
from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict, NotRequired

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition

from llm_graph_agent.graph.common import (
    CompressedAssistant,
    build_system_prompt,
    make_message_manage,
)
from llm_graph_agent.llm import build_chat_model, load_profile, load_prompt
from llm_graph_agent.context.messages import build_turn_aware_tool_node
from llm_graph_agent.tools import registry
from llm_graph_agent.context.compression.session import CompressionSession


class ToolAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    turn_id: int
    compression_session: NotRequired[CompressionSession]


def get_available_tool_names(
    tool_names: list[str] | None,
    default_tool_names: list[str],
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    """工具名列表：未指定时用默认集，去重，可选排除项。"""
    source = tool_names if tool_names is not None else default_tool_names
    exclude = exclude or set()

    result: list[str] = []
    for name in source:
        if name in exclude:
            continue
        if name not in result:
            result.append(name)
    return result


# Agenttool 是"递归子代理"，子代理内不再暴露，避免无限嵌套
SUBAGENT_EXCLUDE = {"agenttool"}

# 默认工具集：能覆盖大多数业务场景的通用能力
DEFAULT_TOOL_NAMES = [
    "read_file",
    "get_file",
    "grep",
    "imageread",
    "agenttool",
    "shell_tool",
]


def build_graph(
    *,
    profile_name: str = "qwen3.6",
    vision_profile_name: str = "qwen3-vl",
    working_dir: str | None = None,
    checkpointer=None,
    context_window_tokens: int = 32768,
    tool_names: list[str] | None = None,
    skill_names: list[str] | None = None,
    instruction: str | None = None,
):
    """构建通用工具 Agent 图。

    参数说明：
    - tool_names: 需要挂载的工具名；不传用默认集
    - skill_names: 额外的 skills（如 "degoog-search"）
    - instruction: 额外的系统指令覆盖（旧项目里不同图用不同 system prompt）
    """
    profile = load_profile(profile_name)
    prompt = load_prompt("tool_agent") if instruction is None else {"system": instruction}
    collapse_prompt = load_prompt("collapse_compact")

    selected_tools = get_available_tool_names(
        tool_names,
        DEFAULT_TOOL_NAMES,
    )
    subagent_tool_names = get_available_tool_names(
        selected_tools,
        DEFAULT_TOOL_NAMES,
        exclude=SUBAGENT_EXCLUDE,
    )

    tools = registry.get_langchain_tools_by_names(
        selected_tools,
        injected_by_tool={
            "agenttool": {
                "_profile_name": profile_name,
                "_vision_profile_name": vision_profile_name,
                "_working_dir": working_dir,
                "_context_window_tokens": context_window_tokens,
                "_recursion_limit": 200,
                "_tool_names": subagent_tool_names,
            },
            "imageread": {
                "_profile_name": vision_profile_name,
            },
            "shell_tool": {
                "_working_dir": working_dir,
            },
        },
    )

    llm = build_chat_model(profile)
    llm_with_tools = llm.bind_tools(tools)

    message_manage = make_message_manage(
        llm=llm,
        collapse_prompt=collapse_prompt["system"],
        context_window_tokens=context_window_tokens,
    )
    system_content = build_system_prompt(
        prompt["system"],
        working_dir=working_dir,
        skill_names=skill_names,
    )

    async def assistant_node(state: ToolAgentState) -> dict:
        assistant = CompressedAssistant(
            message_manage=message_manage,
            plane=llm,                      # 收官用，避免继续调工具
            tooled=llm_with_tools,
            system_content=system_content,
            turn_id=state["turn_id"],
            compression_session=state.get("compression_session"),
        )

        response = await assistant.invoke(state["messages"])

        return {
            "messages": [response],
            "compression_session": assistant.compression_session,
        }

    builder = StateGraph(ToolAgentState)

    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", build_turn_aware_tool_node(tools))

    builder.add_edge(START, "assistant")

    builder.add_conditional_edges(
        "assistant",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END,
        },
    )

    builder.add_edge("tools", "assistant")

    return builder.compile(checkpointer=checkpointer)
