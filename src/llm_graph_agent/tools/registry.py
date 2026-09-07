"""工具注册表：统一收集、查询、包装项目里的所有工具。

工具模块的公开契约（由脚手架 create_tool 约定）：
- TOOL_NAME: 工具名
- IS_READ_ONLY / IS_DESTRUCTIVE: 元数据
- InputSchema: pydantic 输入模型
- call / acall: 同步/异步执行，返回 OutputSchema.model_dump() 字典
- render_result_for_llm(result): 把结果 dict 转成模型可读文本

registry 负责把这些工具包装成 LangChain StructuredTool。
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool

from llm_graph_agent.tools.prompt_loader import read_tool_prompt
from llm_graph_agent.tools.Agenttool import tool as agenttool
from llm_graph_agent.tools.GetFile import tool as get_file
from llm_graph_agent.tools.Grep import tool as grep
from llm_graph_agent.tools.Imageread import tool as imageread
from llm_graph_agent.tools.MemorySearch import tool as memory_search
from llm_graph_agent.tools.MemoryWrite import tool as memory_write
from llm_graph_agent.tools.PythonTool import tool as python_tool
from llm_graph_agent.tools.PythonToolweaker import tool as python_tool_weaker
from llm_graph_agent.tools.ReadFile import tool as read_file
from llm_graph_agent.tools.ShellTool import tool as shell_tool
from llm_graph_agent.tools.SkillTool import tool as skill_tool


TOOL_ENTRIES = {
    read_file.TOOL_NAME: read_file,
    get_file.TOOL_NAME: get_file,
    imageread.TOOL_NAME: imageread,
    agenttool.TOOL_NAME: agenttool,
    memory_search.TOOL_NAME: memory_search,
    memory_write.TOOL_NAME: memory_write,
    python_tool.TOOL_NAME: python_tool,
    shell_tool.TOOL_NAME: shell_tool,
    skill_tool.TOOL_NAME: skill_tool,
    python_tool_weaker.TOOL_NAME: python_tool_weaker,
    grep.TOOL_NAME: grep,
}


def get_tool_dir(tool_module) -> Path:
    """工具 prompt.md 所在目录：跟随工具模块文件。"""
    return Path(tool_module.__file__).resolve().parent


def get_all_tools() -> list:
    """注册表内全部工具模块（保持注册顺序）。"""
    return list(TOOL_ENTRIES.values())


def get_tool(tool_name: str):
    """按名称查找工具模块。"""
    tool = TOOL_ENTRIES.get(tool_name)
    if tool is None:
        raise KeyError(f"Tool not found: {tool_name}")
    return tool


def get_tool_modules_by_names(tool_names: list[str]) -> list:
    """按名称列表取工具模块（保持传入顺序）。"""
    return [
        get_tool(name)
        for name in tool_names
    ]


def get_read_only_tools() -> list:
    """只读工具（无副作用，可安全用于任意环境）。"""
    return [
        tool
        for tool in get_all_tools()
        if tool.IS_READ_ONLY
    ]


def to_langchain_tool(
    tool_module,
    injected_kwargs: dict | None = None,
) -> StructuredTool:
    """把工具模块包装成可绑定到模型的 StructuredTool。

    injected_kwargs 会在每次调用时注入（如 working_dir、profile_name 等
    由 graph 构建期决定的上下文参数）。
    """
    injected_kwargs = injected_kwargs or {}

    def run_tool(**kwargs):
        result = tool_module.call(**kwargs, **injected_kwargs)
        return tool_module.render_result_for_llm(result)

    coroutine = None

    if hasattr(tool_module, "acall"):

        async def arun_tool(**kwargs):
            result = await tool_module.acall(**kwargs, **injected_kwargs)
            return tool_module.render_result_for_llm(result)

        coroutine = arun_tool

    return StructuredTool.from_function(
        func=run_tool,
        coroutine=coroutine,
        name=tool_module.TOOL_NAME,
        description=read_tool_prompt(get_tool_dir(tool_module)),
        args_schema=tool_module.InputSchema,
    )


def get_langchain_tools() -> list:
    """全部工具 → LangChain 工具列表。"""
    return [
        to_langchain_tool(tool)
        for tool in get_all_tools()
    ]


def get_langchain_tools_by_names(
    tool_names: list[str],
    injected_by_tool: dict[str, dict] | None = None,
) -> list:
    """按名称列表取 LangChain 工具；每个工具可带独立注入参数。"""
    injected_by_tool = injected_by_tool or {}

    return [
        to_langchain_tool(
            tool_module,
            injected_kwargs=injected_by_tool.get(tool_module.TOOL_NAME),
        )
        for tool_module in get_tool_modules_by_names(tool_names)
    ]


# 子代理工具集：去掉"递归子代理"本身，避免无限嵌套
SUBAGENT_EXCLUDED_TOOLS = {"agenttool"}


def get_subagent_tool_names(
    parent_tool_names: list[str] | None = None,
) -> list[str]:
    """子代理可用工具名：默认全部工具，去掉 agenttool（防无限递归）。"""
    if parent_tool_names is None:
        source_names = [
            tool.TOOL_NAME
            for tool in get_all_tools()
        ]
    else:
        source_names = parent_tool_names

    result: list[str] = []
    for name in source_names:
        if name in SUBAGENT_EXCLUDED_TOOLS:
            continue
        if name not in result:
            result.append(name)
    return result


def get_subagent_langchain_tools(
    parent_tool_names: list[str] | None = None,
    injected_by_tool: dict[str, dict] | None = None,
) -> list:
    """子代理可用的 LangChain 工具（排除 agenttool）。"""
    tool_names = get_subagent_tool_names(parent_tool_names)
    return get_langchain_tools_by_names(
        tool_names,
        injected_by_tool=injected_by_tool,
    )
