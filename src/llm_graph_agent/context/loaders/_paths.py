"""loaders 内部共享的路径解析工具。"""
from __future__ import annotations

from pathlib import Path

from llm_graph_agent.paths import PROJECT_ROOT


def resolve_working_dir(working_dir: str | Path | None = None) -> Path:
    """把传入的工作目录解析为绝对 Path。

    - None → 项目根
    - 指向文件 → 返回其父目录
    - 其它 → expanduser + resolve
    """
    if working_dir is None:
        return PROJECT_ROOT

    path = Path(working_dir).expanduser().resolve()

    if path.is_file():
        return path.parent

    return path
