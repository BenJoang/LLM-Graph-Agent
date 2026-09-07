from __future__ import annotations

from pathlib import Path

from llm_graph_agent.context.loaders.project_instruction import build_project_instruction_system_message
from llm_graph_agent.context.loaders.skills import build_skill_system_message
from llm_graph_agent.context.loaders.working_dir import build_working_dir_system_message


def build_system_context(
    working_dir: str | Path | None = None,
    skill_names: list[str] | None = None,
    include_all_skills: bool = False,
    working_dir_need: bool = False,
    instruction_need: bool = False
) -> str:
    parts = [
        build_working_dir_system_message(working_dir, working_dir_need_switch=working_dir_need),
        build_project_instruction_system_message(
            working_dir=working_dir,
            mode="nearest",
            instruction_need=instruction_need
        ),
        build_skill_system_message(
            skill_names=skill_names,
            include_all_skills=include_all_skills
        ),
    ]

    return "\n\n".join(part for part in parts if part)