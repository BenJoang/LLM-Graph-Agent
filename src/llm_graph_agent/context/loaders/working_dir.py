from __future__ import annotations

from pathlib import Path


from llm_graph_agent.paths import PROJECT_ROOT
from llm_graph_agent.context.loaders._paths import resolve_working_dir



def build_working_dir_system_message(
    working_dir: str | Path | None = None,
    working_dir_need_switch: bool = False,
) -> str:
    if working_dir_need_switch:
        resolved_working_dir = resolve_working_dir(working_dir)

        return (
            "本次运行的工作目录如下。涉及相对路径、项目文件读取、项目规则判断时，"
            "优先以该目录作为当前项目目录。\n"
            f'<working-directory>'
            f'{resolved_working_dir}'
            f'</working-directory>'
        )
    else:
        return 