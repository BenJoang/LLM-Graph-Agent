"""提示词加载。使用分层配置发现（见 llm_graph_agent.config）。"""
from __future__ import annotations

import json
from pathlib import Path

from llm_graph_agent.config import find_config_file


def _load_prompt_config() -> dict:
    """按分层规则找到 prompt_config.json 并读取。"""
    path = find_config_file("prompt_config.json")
    if path is None:
        raise FileNotFoundError(
            "找不到 prompt_config.json。请把配置文件放到项目 config/ 目录，"
            "或设置 LLM_GRAPH_PROMPT_CONFIG_PATH 环境变量指向它。"
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _prompts_base_dir() -> Path:
    """prompts/*.md 所在目录：跟随 config 所在目录。"""
    path = find_config_file("prompt_config.json")
    if path is None:
        raise FileNotFoundError("找不到 prompt_config.json，无法定位 prompts 目录。")
    return path.parent


def load_prompt(prompt_name: str) -> dict:
    """读取 prompt 配置，并把 system_file 内容读入 prompt["system"]。"""
    config = _load_prompt_config()
    if prompt_name not in config.get("prompts", {}):
        raise KeyError(
            f"prompt '{prompt_name}' 不存在。可选：{sorted(config.get('prompts', {}).keys())}"
        )

    prompt = dict(config["prompts"][prompt_name])

    system_file = prompt.get("system_file")
    if system_file:
        system_path = Path(system_file)
        if not system_path.is_absolute():
            system_path = _prompts_base_dir() / system_path
        prompt["system"] = system_path.read_text(encoding="utf-8")

    return prompt
