"""测试配置发现层。"""
from pathlib import Path

from llm_graph_agent.config import (
    _env_key_for,
    find_config_file,
    user_config_dir,
)


def test_env_key_for():
    assert _env_key_for("user_config.json") == "LLM_GRAPH_USER_CONFIG_PATH"
    assert _env_key_for("prompt_config.json") == "LLM_GRAPH_PROMPT_CONFIG_PATH"


def test_user_config_dir_uses_env(monkeypatch):
    monkeypatch.setenv("LLM_GRAPH_CONFIG_DIR", r"C:\\fake\\dir")
    assert user_config_dir() == Path(r"C:\\fake\\dir")


def test_user_config_dir_defaults_to_appdata(monkeypatch):
    monkeypatch.delenv("LLM_GRAPH_CONFIG_DIR", raising=False)
    monkeypatch.setenv("APPDATA", r"C:\\Users\\test\\AppData\\Roaming")
    assert user_config_dir() == Path(r"C:\\Users\\test\\AppData\\Roaming") / "llm_graph_agent"


def test_find_config_file_prefers_env(monkeypatch, tmp_path):
    env_file = tmp_path / "user_config.json"
    env_file.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("LLM_GRAPH_USER_CONFIG_PATH", str(env_file))
    found = find_config_file("user_config.json")
    assert found == env_file


def test_find_config_file_env_missing_falls_back(monkeypatch, tmp_path):
    # env 指向不存在的文件 -> 不匹配，落到用户/项目目录
    monkeypatch.setenv("LLM_GRAPH_USER_CONFIG_PATH", str(tmp_path / "nope.json"))
    # 阻断用户目录 -> 走项目默认（我们的仓库有 config/user_config.json）
    monkeypatch.setenv("LLM_GRAPH_CONFIG_DIR", str(tmp_path / "userconf"))
    found = find_config_file("user_config.json")
    # 仓库里有 config/user_config.json，所以应返回它
    assert found is not None
    assert found.name == "user_config.json"
