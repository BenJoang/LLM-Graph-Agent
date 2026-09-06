"""配置发现层：模仿 opencode 的分层加载。

查找顺序（优先级从高到低）：
1. 环境变量  LLM_GRAPH_<NAME>_PATH  （显式指定单个文件，例如 user_config.json -> LLM_GRAPH_USER_CONFIG_PATH）
2. 用户目录  %APPDATA%/llm_graph_agent/  （Windows）或 ~/.config/llm_graph_agent/
3. 项目默认  <PROJECT_ROOT>/config/

设计目的：代码和配置解耦。使用者可以在自己机器上覆盖配置，
不需要改仓库里的任何代码。
"""
from __future__ import annotations

import os
from pathlib import Path

from llm_graph_agent.paths import PROJECT_ROOT


def user_config_dir() -> Path:
    """用户级配置目录（XDG 风格；Windows 用 APPDATA）。"""
    env = os.getenv("LLM_GRAPH_CONFIG_DIR")
    if env:
        return Path(env)
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "llm_graph_agent"


def _env_key_for(name: str) -> str:
    """user_config.json -> LLM_GRAPH_USER_CONFIG_PATH；去掉扩展名和点。"""
    stem = name
    for suffix in (".json", ".jsonc", ".yaml", ".yml", ".toml"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return f"LLM_GRAPH_{stem.upper().replace('-', '_')}_PATH"


def find_config_file(name: str) -> Path | None:
    """按 环境变量 → 用户目录 → 项目默认 的顺序找配置文件。"""
    # 1. 环境变量显式指定
    env = os.getenv(_env_key_for(name))
    if env:
        p = Path(env)
        if p.exists():
            return p

    # 2. 用户目录
    user = user_config_dir() / name
    if user.exists():
        return user

    # 3. 项目默认
    project = PROJECT_ROOT / "config" / name
    if project.exists():
        return project

    return None


def resolve_config_dir() -> Path:
    """找到实际使用的配置目录（用于生成用户配置时知道往哪写）。"""
    for p in (user_config_dir(), PROJECT_ROOT / "config"):
        if p.exists():
            return p
    # 都不存在时默认用户目录（将来给别人用：首次运行自动生成）
    return user_config_dir()
