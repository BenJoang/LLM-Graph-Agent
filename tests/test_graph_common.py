"""graph 公共编排入口（CompressedAssistant 等）的结构测试。

不调用真实 LLM：用 RunnableLambda 假模型验证编排路径与会话状态流转。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from llm_graph_agent.graph.common import (
    CompressedAssistant,
    build_system_prompt,
    make_message_manage,
)


def test_build_system_prompt_assembles_parts():
    result = build_system_prompt(
        "角色提示",
        working_dir=None,
        skill_names=[],
    )
    assert "角色提示" in result
    # 动态上下文（skills/工作目录）会拼在后面
    assert "skills" in result.lower() or "当前没有可用 skills" in result


def test_make_message_manage_wires_summarizer():
    llm = RunnableLambda(lambda msgs: AIMessage(content="sum"))
    manage = make_message_manage(
        llm=llm,
        collapse_prompt="总结历史",
        context_window_tokens=1024,
    )
    assert manage.thresholds.max_tokens == 1024
    assert manage.asummarize_fn is not None

    # 摘要回调应能调用（走假 LLM）
    import asyncio
    result = asyncio.run(manage.asummarize_fn("历史文本"))
    assert result == "sum"


def test_compressed_assistant_invokes_plane_when_finalize():
    async def plane_fn(msgs):
        # 收官时用 plane（未绑定工具）
        return AIMessage(content="final", id="ai-1")

    async def tooled_fn(msgs):
        raise AssertionError("finalize 时不应调用 tooled")

    assistant = CompressedAssistant(
        message_manage=make_message_manage(
            llm=RunnableLambda(lambda msgs: AIMessage(content="s")),
            collapse_prompt="总结",
            context_window_tokens=1024,
        ),
        plane=RunnableLambda(plane_fn),
        tooled=RunnableLambda(tooled_fn),
        system_content="系统",
        turn_id=1,
    )

    import asyncio
    response = asyncio.run(
        assistant.invoke([HumanMessage(content="hi")], should_finalize=True)
    )
    assert response.content == "final"


def test_compressed_assistant_uses_tooled_by_default():
    async def tooled_fn(msgs):
        return AIMessage(content="tooled", id="ai-2")

    assistant = CompressedAssistant(
        message_manage=make_message_manage(
            llm=RunnableLambda(lambda msgs: AIMessage(content="s")),
            collapse_prompt="总结",
            context_window_tokens=1024,
        ),
        plane=RunnableLambda(lambda msgs: AIMessage(content="plane")),
        tooled=RunnableLambda(tooled_fn),
        system_content="系统",
        turn_id=2,
    )

    import asyncio
    response = asyncio.run(assistant.invoke([HumanMessage(content="hi")]))
    assert response.content == "tooled"


def test_compressed_assistant_persists_session_across_invokes():
    import asyncio

    calls = {"n": 0}

    async def tooled_fn(msgs):
        calls["n"] += 1
        return AIMessage(content=f"r{calls['n']}", id=f"ai-{calls['n']}")

    assistant = CompressedAssistant(
        message_manage=make_message_manage(
            llm=RunnableLambda(lambda msgs: AIMessage(content="s")),
            collapse_prompt="总结",
            context_window_tokens=1024,
        ),
        plane=RunnableLambda(lambda msgs: AIMessage(content="p")),
        tooled=RunnableLambda(tooled_fn),
        system_content="系统",
        turn_id=3,
    )

    asyncio.run(assistant.invoke([HumanMessage(content="a")]))
    session_after_first = dict(assistant.compression_session)
    asyncio.run(assistant.invoke([HumanMessage(content="b")]))
    assert assistant.compression_session == session_after_first
    assert calls["n"] == 2
