"""测试 profile 加载。"""
import json

import pytest

import llm_graph_agent.llm.profiles as profiles_mod
from llm_graph_agent.llm.profiles import load_profile


@pytest.fixture
def fake_user_config(tmp_path, monkeypatch):
    """造一个最小 user_config.json，并让 profiles 模块的 find_config_file 返回它。"""
    cfg = tmp_path / "user_config.json"
    cfg.write_text(json.dumps({
        "defaults": {"generation": {"temperature": 0.7, "top_p": 0.9}},
        "profiles": {
            "test-profile": {
                "model": "test-model",
                "base_url_env": "TEST_BASE_URL",
                "api_key_env": "TEST_API_KEY",
                "generation": {"temperature": 1.5},
            },
            "no-env-profile": {"model": "plain-model"},
        },
    }), encoding="utf-8")

    monkeypatch.setattr(
        profiles_mod, "find_config_file",
        lambda name: cfg if name == "user_config.json" else None,
    )
    return cfg


def test_load_profile_basic(fake_user_config, monkeypatch):
    # 基础行为：读 model + generation 合并。用不依赖 env 的 profile。
    p = load_profile("no-env-profile")
    assert p["model"] == "plain-model"
    # 默认 generation 合并
    assert p["generation"]["temperature"] == 0.7
    assert p["generation"]["top_p"] == 0.9


def test_load_profile_generation_override(fake_user_config):
    # profile 自带 generation 覆盖 defaults
    p = load_profile("no-env-profile")
    # no-env-profile 没有自带 generation，这里验证 defaults 生效
    assert p["generation"]["temperature"] == 0.7
    # 单独验证 profile 覆盖：直接用 test-profile 但有 env
    # 见 test_load_profile_resolves_env


def test_load_profile_resolves_env(monkeypatch, fake_user_config):
    monkeypatch.setenv("TEST_BASE_URL", "http://test:8000/v1")
    monkeypatch.setenv("TEST_API_KEY", "sk-test")
    p = load_profile("test-profile")
    assert p["model"] == "test-model"
    assert p["base_url"] == "http://test:8000/v1"
    assert p["api_key"] == "sk-test"
    assert "base_url_env" not in p
    assert "api_key_env" not in p
    # profile 的 generation 覆盖 defaults
    assert p["generation"]["temperature"] == 1.5
    assert p["generation"]["top_p"] == 0.9


def test_load_profile_missing_env_raises(fake_user_config):
    with pytest.raises(ValueError, match="TEST_BASE_URL"):
        load_profile("test-profile")


def test_load_profile_no_env_keys(fake_user_config):
    p = load_profile("no-env-profile")
    assert "base_url" not in p


def test_load_profile_unknown_raises(fake_user_config):
    with pytest.raises(KeyError):
        load_profile("nonexistent")


def test_load_profile_missing_config_file_raises(monkeypatch):
    monkeypatch.setattr(profiles_mod, "find_config_file", lambda name: None)
    with pytest.raises(FileNotFoundError):
        load_profile("anything")
