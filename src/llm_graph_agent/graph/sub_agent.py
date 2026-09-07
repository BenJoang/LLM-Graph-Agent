"""子代理图：有执行预算的深度调查代理。

- 保留 RemainingSteps 预算跟踪
- 接近预算（<=4 步）时插入"收官指令"并切换到未绑定工具的模型，
  强制产出阶段性报告
- 状态含 status / stop_reason，供外层判断"完整完成 / 部分完成"
"""
from __future__ import annotations

from typing import Annotated, Literal
from typing_extensions import TypedDict, NotRequired

from langgraph.managed import RemainingSteps
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition

from langchain_core.messages import HumanMessage

from llm_graph_agent.graph.common import (
    CompressedAssistant,
    build_system_prompt,
    make_message_manage,
)
from llm_graph_agent.llm import build_chat_model, load_profile, load_prompt
from llm_graph_agent.context.compression.session import CompressionSession
from llm_graph_agent.context.messages import build_turn_aware_tool_node
from llm_graph_agent.tools import registry


class SubAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    turn_id: int
    remaining_steps: RemainingSteps
    status: NotRequired[Literal["running", "completed", "partial"]]
    stop_reason: NotRequired[str]
    compression_session: NotRequired[CompressionSession]


FINALIZE_THRESHOLD = 4


def build_subagent_injection_message() -> HumanMessage:
    """接近预算时注入的收官指令。"""
    return HumanMessage(
        content=(
            "系统通知：子 agent 的执行预算即将耗尽。\n"
            "现在禁止调用任何工具。\n"
            "请立即根据此前已有的工具结果输出阶段性调查报告。\n"
            "即使任务尚未完成，也必须输出非空内容。\n\n"
            "报告必须包含：\n"
            "1. 已确认的结论\n"
            "2. 对应证据、文件路径或工具结果\n"
            "3. 尚未确认的部分\n"
            "4. 建议的后续调查步骤"
        ),
        name="subagent_budget_controller",
        additional_kwargs={
            "synthetic": True,
            "reason": "step_limit",
        },
    )


def build_graph(
    *,
    profile_name: str = "qwen3.6",
    vision_profile_name: str = "qwen3-vl",
    working_dir: str | None = None,
    context_window_tokens: int = 32768,
    tool_names: list[str] | None = None,
):
    profile = load_profile(profile_name)
    prompt = load_prompt("subagent")
    collapse_prompt = load_prompt("collapse_compact")

    tools = registry.get_subagent_langchain_tools(
        parent_tool_names=tool_names,
        injected_by_tool={
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
        skill_names=[],
    )

    async def assistant_node(state: SubAgentState) -> dict:
        remaining_steps = state["remaining_steps"]
        should_finalize = remaining_steps <= FINALIZE_THRESHOLD

        assistant = CompressedAssistant(
            message_manage=message_manage,
            plane=llm,                      # 收官禁工具
            tooled=llm_with_tools,
            system_content=system_content,
            turn_id=state["turn_id"],
            compression_session=state.get("compression_session"),
        )

        # 收官时在末尾注入"禁止调用工具"的指令
        finalize_message = (
            build_subagent_injection_message()
            if should_finalize
            else None
        )

        messages = state["messages"]
        if finalize_message is not None:
            messages = [*messages, finalize_message]

        response = await assistant.invoke(
            messages,
            should_finalize=should_finalize,
        )

        new_messages = []
        if finalize_message is not None:
            new_messages.append(finalize_message)
        new_messages.append(response)

        if should_finalize:
            status = "partial"
            stop_reason = "step_limit"
        elif getattr(response, "tool_calls", None):
            status = "running"
            stop_reason = ""
        else:
            status = "completed"
            stop_reason = ""

        return {
            "messages": new_messages,
            "status": status,
            "stop_reason": stop_reason,
            "compression_session": assistant.compression_session,
        }

    builder = StateGraph(SubAgentState)

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

    return builder.compile()
