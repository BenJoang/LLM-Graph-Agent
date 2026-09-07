"""工具输出裁剪。

把超长的工具输出压缩成"头 + 省略标记 + 尾"，保留可读的上下文。
纯文本处理，不涉及模型调用。
"""
from __future__ import annotations


def snip_tool_content(
    content: str,
    *,
    tool_head_chars: int,
    tool_tail_lines: int,
    tool_tail_chars: int,
) -> tuple[str, bool]:
    """把超长的工具输出裁剪为头 + 省略标记 + 尾。"""
    head_chars = tool_head_chars
    tail_lines = tool_tail_lines
    tail_chars = tool_tail_chars

    if len(content) <= head_chars + tail_chars:
        return content, False

    head = content[:head_chars]

    lines = content.splitlines(keepends=True)
    tail_candidate = "".join(lines[-tail_lines:])
    tail = tail_candidate[-tail_chars:]

    tail_start = len(content) - len(tail)

    # 头尾已经重叠，没必要插入省略标记。
    if tail_start <= len(head):
        return content, False

    removed_chars = tail_start - len(head)

    marker = (
        "\n"
        f"[tool output snipped: {removed_chars} chars removed; "
        f"last {min(tail_lines, len(lines))} lines retained]"
        "\n"
    )

    snipped = head.rstrip("\n") + marker + tail.lstrip("\n")

    return snipped, snipped != content


def history_snip(
    messages: list,
    *,
    cutoff: int,
    tool_head_chars: int,
    tool_tail_lines: int,
    tool_tail_chars: int,
    is_tool_message,
    get_content,
    set_content,
) -> bool:
    """遍历历史消息，把超长的工具输出逐个裁剪。

    参数中的 is_tool_message / get_content / set_content 是消息访问回调，
    由调用方（MessageManage）注入，避免这里耦合具体的消息类型判断。
    """
    changed = False
    cutoff_index = max(0, len(messages) - cutoff)

    for index, msg in enumerate(messages):
        if index >= cutoff_index:
            continue

        if not is_tool_message(msg):
            continue
        content = get_content(msg)
        if not isinstance(content, str):
            continue

        if len(content) <= tool_head_chars:
            continue

        snipped_content, content_changed = snip_tool_content(
            content,
            tool_head_chars=tool_head_chars,
            tool_tail_lines=tool_tail_lines,
            tool_tail_chars=tool_tail_chars,
        )

        if not content_changed:
            continue

        set_content(msg, snipped_content)
        changed = True
    return changed
