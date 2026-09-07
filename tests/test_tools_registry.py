"""工具层公共契约：registry 注册、LangChain 包装、只读集、子代理工具集。"""
from __future__ import annotations

from llm_graph_agent.tools import (
    get_all_tools,
    get_langchain_tools,
    get_langchain_tools_by_names,
    get_read_only_tools,
    get_subagent_langchain_tools,
    get_subagent_tool_names,
    get_tool,
    get_tool_modules_by_names,
    read_tool_prompt,
    to_langchain_tool,
)
from llm_graph_agent.tools.ReadFile import tool as read_file


def test_registry_has_generic_tools_only():
    """QQ 专用工具已被剔除，通用工具齐全。"""
    names = {tool.TOOL_NAME for tool in get_all_tools()}
    assert "qq_memory_search" not in names
    assert {
        "read_file",
        "get_file",
        "grep",
        "shell_tool",
        "python_tool",
        "memory_search",
        "memory_write",
        "imageread",
        "agenttool",
        "skill_tool",
        "python_tool_weaker",
    } <= names


def test_get_tool_finds_and_raises():
    assert get_tool("read_file") is read_file
    try:
        get_tool("not_a_tool")
        raise AssertionError("应抛 KeyError")
    except KeyError:
        pass


def test_get_tool_modules_by_names_keeps_order():
    modules = get_tool_modules_by_names(["read_file", "grep", "read_file"])
    names = [m.TOOL_NAME for m in modules]
    assert names == ["read_file", "grep", "read_file"]


def test_to_langchain_tool_wraps_with_prompt_and_schema():
    lc_tool = to_langchain_tool(read_file)
    assert lc_tool.name == "read_file"
    assert lc_tool.args_schema is read_file.InputSchema
    assert "WHEN_TO_USE" in lc_tool.description


def test_get_langchain_tools_all_present():
    tools = get_langchain_tools()
    assert {t.name for t in tools} == {m.TOOL_NAME for m in get_all_tools()}


def test_get_langchain_tools_by_names_with_injection():
    injected = {"shell_tool": {"_working_dir": "/tmp"}}
    tools = get_langchain_tools_by_names(["shell_tool"], injected_by_tool=injected)
    assert len(tools) == 1
    assert tools[0].name == "shell_tool"


def test_get_read_only_tools_mark_flags():
    tools = get_read_only_tools()
    assert all(tool.IS_READ_ONLY for tool in tools)
    assert "shell_tool" not in {t.TOOL_NAME for t in tools}


def test_subagent_tool_names_exclude_agenttool():
    names = get_subagent_tool_names()
    assert "agenttool" not in names
    # 全部工具（默认）减去 agenttool
    assert len(names) == len(get_all_tools()) - 1


def test_subagent_tool_names_respect_parent_selection():
    names = get_subagent_tool_names(["read_file", "grep", "agenttool", "read_file"])
    assert names == ["read_file", "grep"]


def test_subagent_langchain_tools_exclude_agenttool():
    tools = get_subagent_langchain_tools(["read_file", "grep", "agenttool"])
    assert {t.name for t in tools} == {"read_file", "grep"}


def test_prompt_loader_reads_sections():
    tool_dir = read_file.TOOL_DIR
    description = read_tool_prompt(tool_dir)
    assert description.startswith("###")
