"""测试模型家族抽象。"""
import pytest

from llm_graph_agent.llm.families import (
    DeepSeekFamily,
    ModelFamily,
    QwenFamily,
    get_family,
    register_family,
)


def test_qwen_matches():
    family = get_family("qwen3.6-27b")
    assert isinstance(family, QwenFamily)


def test_deepseek_matches():
    family = get_family("deepseek-chat")
    assert isinstance(family, DeepSeekFamily)


def test_qwen_apply_thinking_enabled():
    family = QwenFamily()
    extra_body = {}
    family.apply_thinking(extra_body, True)
    assert extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_qwen_apply_thinking_disabled():
    family = QwenFamily()
    extra_body = {}
    family.apply_thinking(extra_body, False)
    assert extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_deepseek_apply_thinking_enabled():
    family = DeepSeekFamily()
    extra_body = {}
    family.apply_thinking(extra_body, True)
    assert extra_body == {"thinking": {"type": "enabled"}}


def test_deepseek_apply_thinking_disabled():
    family = DeepSeekFamily()
    extra_body = {}
    family.apply_thinking(extra_body, False)
    assert extra_body == {"thinking": {"type": "disabled"}}


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        get_family("unknown-model-xyz")


def test_register_custom_family():
    class FakeFamily(ModelFamily):
        prefixes = ("fake",)

        def apply_thinking(self, extra_body, thinking):
            extra_body["fake_thinking"] = thinking

    register_family(FakeFamily())
    family = get_family("fake-anything")
    assert isinstance(family, FakeFamily)
