"""测试 prompt 加载。"""
import json

import pytest

import llm_graph_agent.llm.prompts as prompts_mod
from llm_graph_agent.llm.prompts import load_prompt


@pytest.fixture
def fake_prompt_config(tmp_path, monkeypatch):
    """造一个最小 prompt_config.json + 对应 md，并 mock prompts 模块的 find_config_file。"""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "my_prompt.md").write_text("这是测试 prompt", encoding="utf-8")

    cfg = tmp_path / "prompt_config.json"
    cfg.write_text(json.dumps({
        "prompts": {
            "inline_prompt": {"system": "inline system text"},
            "file_prompt": {"system_file": "prompts/my_prompt.md"},
        },
    }), encoding="utf-8")

    monkeypatch.setattr(
        prompts_mod, "find_config_file",
        lambda name: cfg if name == "prompt_config.json" else None,
    )
    return cfg


def test_load_inline_prompt(fake_prompt_config):
    p = load_prompt("inline_prompt")
    assert p["system"] == "inline system text"


def test_load_file_prompt(fake_prompt_config):
    p = load_prompt("file_prompt")
    assert "这是测试 prompt" in p["system"]


def test_load_prompt_unknown_raises(fake_prompt_config):
    with pytest.raises(KeyError):
        load_prompt("nonexistent")


def test_load_prompt_missing_config_raises(monkeypatch):
    monkeypatch.setattr(prompts_mod, "find_config_file", lambda name: None)
    with pytest.raises(FileNotFoundError):
        load_prompt("anything")
