"""模型 profile 加载。使用分层配置发现（见 llm_graph_agent.config）。"""
from __future__ import annotations

import json
import os

from llm_graph_agent.config import find_config_file


def _load_user_config() -> dict:
    """按分层规则找到 user_config.json 并读取。找不到抛清晰错误。"""
    path = find_config_file("user_config.json")
    if path is None:
        raise FileNotFoundError(
            "找不到 user_config.json。请把配置文件放到项目 config/ 目录，"
            "或设置 LLM_GRAPH_USER_CONFIG_PATH 环境变量指向它。"
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_profile(profile_name: str) -> dict:
    """读取指定 profile，合并默认生成参数，解析环境变量里的 base_url/api_key。"""
    config = _load_user_config()
    if profile_name not in config.get("profiles", {}):
        raise KeyError(
            f"profile '{profile_name}' 不存在。可选：{sorted(config.get('profiles', {}).keys())}"
        )

    profile = dict(config["profiles"][profile_name])

    default_generation = config.get("defaults", {}).get("generation", {})
    profile_generation = profile.get("generation", {})
    profile["generation"] = {**default_generation, **profile_generation}

    base_url_env = profile.pop("base_url_env", None)
    if base_url_env:
        base_url = os.getenv(base_url_env)
        if not base_url:
            raise ValueError(f"Missing environment variable: {base_url_env}")
        profile["base_url"] = base_url

    api_key_env = profile.pop("api_key_env", None)
    if api_key_env:
        profile["api_key"] = os.getenv(api_key_env)

    return profile
