"""read_file 工具行为测试（契约保持）。"""
from __future__ import annotations

from llm_graph_agent.tools.ReadFile import tool as read_file


def test_read_text_file(tmp_path):
    target = tmp_path / "example.txt"
    target.write_text("第一行\n第二行\n第三行", encoding="utf-8")

    result = read_file.call(
        file_path=str(target),
        offset=1,
        limit=2,
    )

    assert result["ok"] is True
    assert result["count"] == 2
    assert "1: 第一行" in result["content"]
    assert "2: 第二行" in result["content"]
    assert result["next_offset"] == 3
    assert result["truncated_by_lines"] is True


def test_read_missing_file(tmp_path):
    result = read_file.call(
        file_path=str(tmp_path / "missing.txt"),
    )

    assert result["ok"] is False
    assert "不存在" in result["error"]


def test_read_char_offset_mid_line(tmp_path):
    target = tmp_path / "mid.txt"
    target.write_text("abcdef\nghijk", encoding="utf-8")

    result = read_file.call(
        file_path=str(target),
        offset=1,
        char_offset=3,
        limit=1,
    )

    assert result["ok"] is True
    assert "def" in result["content"]
    assert result["start_char"] == 3
    # 第 1 行只剩 3 个字符，需继续从第 2 行开头读取
    assert result["next_offset"] == 2
    assert result["next_char_offset"] == 0
    assert result["truncated"] is True
