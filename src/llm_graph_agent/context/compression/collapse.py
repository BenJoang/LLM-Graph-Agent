"""上下文折叠：把旧轮次的消息折叠成摘要消息。

纯结构操作：查找可折叠的消息块、生成摘要/压缩消息、判定消息类型。
不直接调用模型——摘要内容由调用方通过回调生成后传入。
"""
from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from langchain_core.messages import HumanMessage, AIMessage

from llm_graph_agent.context.messages import (
    get_message_turn_id,
    mark_ai_message,
    set_context_state,
)


def _apply_context_collapse(messages: list, session: dict):
    result = messages

    for commit in session["collapse_commits"]:
        source_ids = commit.get("source_message_ids", [])

        if not source_ids:
            # 旧版 commit 没有 source_message_ids。
            # 可以选择保留旧逻辑，或者清理旧 checkpoint。
            continue

        result_ids = [
            _get_message_id(msg)
            for msg in result
        ]

        source_count = len(source_ids)
        matched_start = None

        for start in range(
            0,
            len(result_ids) - source_count + 1,
        ):
            if result_ids[start:start + source_count] == source_ids:
                matched_start = start
                break

        if matched_start is None:
            continue

        output_message_type = commit.get(
            "output_message_type",
            "ai",
        )

        if output_message_type == "human":
            summary_message = (
                _make_compressed_turn_message(
                    content=commit["summary_content"],
                    collapse_id=commit["collapse_id"],
                    turn_id=commit["turn_id"],
                    message_id=commit["summary_id"],
                )
            )
        else:
            summary_message = _make_summary_message(
                summary_content=commit[
                    "summary_content"
                ],
                collapse_id=commit["collapse_id"],
                summary_id=commit["summary_id"],
                turn_id=commit.get("turn_id"),
                summary_kind=commit.get(
                    "kind",
                    "normal",
                ),
            )

        result[
            matched_start:matched_start + source_count
        ] = [summary_message]

    return result

def _find_next_collapse_batch(
    messages: list,
    session: dict,
    *,
    collapse_batch_size: int,
    summarize_start: int,
    summarize_end: int,
) -> tuple[int, int, object] | None:
    protected_end = len(messages) - summarize_end

    if protected_end <= summarize_start:
        return None

    human_indexes = [
        index
        for index, msg in enumerate(messages)
        if _is_human_message(msg)
    ]

    if not human_indexes:
        return None

    collapsed_ids = session["collapse_message_ids"]

    for position, human_index in enumerate(human_indexes):
        next_human_index = (
            human_indexes[position + 1]
            if position + 1 < len(human_indexes)
            else len(messages)
        )

        body_start = max(
            human_index + 1,
            summarize_start,
        )
        body_end = min(
            next_human_index,
            protected_end,
        )

        if body_end - body_start <= 1:
            continue

        # 摘要消息和已压缩消息都会成为分隔点，
        # 防止新批次跨过旧摘要。
        block_start = None

        for index in range(body_start, body_end + 1):
            reached_end = index == body_end

            if not reached_end:
                msg = messages[index]
                msg_id = _get_message_id(msg)

                compressible = (
                    not _is_human_message(msg)
                    and not _is_summary_message(msg, session)
                    and msg_id not in collapsed_ids
                )
            else:
                compressible = False

            if compressible and block_start is None:
                block_start = index
                continue

            if compressible:
                continue

            if block_start is None:
                continue

            block_end = index
            block_length = block_end - block_start

            if block_length > 1:
                batch_end = min(
                    block_start + collapse_batch_size,
                    block_end,
                )

                batch_end = _adjust_batch_for_tool_calls(
                    messages=messages,
                    start=block_start,
                    end=batch_end,
                    hard_end=block_end,
                )

                if batch_end is not None and batch_end - block_start > 1:
                    return (
                        block_start,
                        batch_end,
                        messages[human_index],
                    )

            block_start = None

    return None

def _find_contiguous_summary_groups(
    messages: list,
    start_index: int,
    end_index: int,
    session: dict,
) -> list[list[str]]:
    groups = []
    current_group = []

    for message in messages[start_index:end_index]:
        if _is_summary_message(message, session):
            message_id = _get_message_id(message)

            if message_id is not None:
                current_group.append(message_id)

            continue

        if len(current_group) > 1:
            groups.append(current_group)

        current_group = []

    if len(current_group) > 1:
        groups.append(current_group)

    return groups

def _find_contiguous_message_ids(
    messages: list,
    source_ids: list[str],
) -> tuple[int, int] | None:
    message_ids = [
        _get_message_id(message)
        for message in messages
    ]

    source_count = len(source_ids)

    for start in range(
        len(message_ids) - source_count + 1
    ):
        end = start + source_count

        if message_ids[start:end] == source_ids:
            return start, end

    return None

def _plan_retry_summary_merges(
    messages: list,
    session: dict,
) -> list[dict]:
    human_indexes = [
        index
        for index, message in enumerate(messages)
        if _is_human_message(message)
    ]

    if not human_indexes:
        return []

    plans = []

    for position, human_index in enumerate(human_indexes):
        next_human_index = (
            human_indexes[position + 1]
            if position + 1 < len(human_indexes)
            else len(messages)
        )

        anchor_human = messages[human_index]
        anchor_human_id = _get_message_id(anchor_human)
        turn_id = get_message_turn_id(anchor_human)

        body_start = human_index + 1
        body_end = next_human_index

        # 在当前 HumanMessage 和下一个 HumanMessage 之间，
        # 查找包含至少两个摘要的连续摘要块。
        summary_groups = _find_contiguous_summary_groups(
            messages=messages,
            start_index=body_start,
            end_index=body_end,
            session=session,
        )

        for source_ids in summary_groups:
            plans.append({
                "turn_id": turn_id,
                "anchor_human_id": anchor_human_id,
                "source_message_ids": source_ids,
            })

    return plans

def _plan_retry_turn_collapses(
    messages: list,
    current_turn_id: int,
) -> list[dict]:
    human_indexes = [
        index
        for index, message in enumerate(messages)
        if _is_human_message(message)
    ]

    plans = []

    for position, human_index in enumerate(
        human_indexes
    ):
        human_message = messages[human_index]
        turn_id = get_message_turn_id(
            human_message
        )

        # 当前轮绝对不能处理。
        if turn_id == current_turn_id:
            continue

        # 如果你要求只处理明确的旧轮次，
        # 缺少 turn_id 的 HumanMessage 也跳过。
        if turn_id is None:
            continue

        # 正常情况下旧轮 turn_id 应小于当前轮。
        if turn_id >= current_turn_id:
            continue

        if _is_compressed_turn_message(
            human_message
        ):
            continue

        next_human_index = (
            human_indexes[position + 1]
            if position + 1 < len(human_indexes)
            else len(messages)
        )

        turn_body = messages[
            human_index + 1:next_human_index
        ]

        # 必须恰好只有一个对应 AIMessage。
        if len(turn_body) != 1:
            continue

        ai_message = turn_body[0]

        if not _is_ai_message(ai_message):
            continue

        human_id = _get_message_id(
            human_message
        )
        ai_id = _get_message_id(
            ai_message
        )

        if human_id is None or ai_id is None:
            continue

        plans.append({
            "turn_id": turn_id,
            "human_id": human_id,
            "ai_id": ai_id,
            "source_message_ids": [
                human_id,
                ai_id,
            ],
        })

    # human_indexes 本身是时间顺序，
    # 所以 plans 默认从最旧轮次开始。
    return plans

def _adjust_batch_for_tool_calls(
    messages: list,
    start: int,
    end: int,
    hard_end: int,
) -> int | None:
    # 批次不能以 ToolMessage 开头，否则可能留下孤立工具结果。
    if _is_tool_message(messages[start]):
        return None

    adjusted_end = end

    # 如果最后选择的是带 tool_calls 的 AIMessage，
    # 把它后面的 ToolMessage 一并包含进来。
    if _has_tool_calls(messages[adjusted_end - 1]):
        while (
            adjusted_end < hard_end
            and _is_tool_message(messages[adjusted_end])
        ):
            adjusted_end += 1

    # 如果批次已经包含一个 ToolMessage，
    # 将紧随其后的并行 ToolMessage 一起包含。
    while (
        adjusted_end < hard_end
        and _is_tool_message(messages[adjusted_end - 1])
        and _is_tool_message(messages[adjusted_end])
    ):
        adjusted_end += 1

    # 最终仍以带 tool_calls 的消息结尾，说明对应结果在保护区外。
    if _has_tool_calls(messages[adjusted_end - 1]):
        return None

    # 下一条还是 ToolMessage，说明并行工具结果没有收完整。
    if (
        adjusted_end < len(messages)
        and _is_tool_message(messages[adjusted_end - 1])
        and _is_tool_message(messages[adjusted_end])
    ):
        return None

    return adjusted_end

def _make_summary_message(
    summary_content: str,
    collapse_id: str,
    summary_id: str | None = None,
    turn_id: int | None = None,
    summary_kind: str = "normal",
) -> AIMessage:
    
    message = AIMessage(
        content=summary_content,
        id=summary_id or str(uuid4()),
        response_metadata={
            "context_compression": {
                "is_summary": True,
                "collapse_id": collapse_id,
                "summary_kind": summary_kind,
            }
        },
    )

    if turn_id is not None:
        mark_ai_message(message, turn_id=turn_id)

    return message

def _make_compressed_turn_message(
    content: str,
    collapse_id: str,
    turn_id: int,
    message_id: str | None = None,
) -> HumanMessage:
    message = HumanMessage(
        content=content,
        id=message_id or str(uuid4()),
        response_metadata={
            "context_compression": {
                "is_compressed_turn": True,
                "collapse_id": collapse_id,
                "summary_kind": "retry_turn_collapse",
            }
        },
    )

    set_context_state(
        message,
        turn_id=turn_id,
    )

    return message

def _message_role(msg):
    if isinstance(msg, dict):
        return msg.get("role", "unknown")

    name = msg.__class__.__name__
    if name == "HumanMessage":
        return "user"
    if name == "AIMessage":
        return "assistant"
    if name == "ToolMessage":
        return "tool"
    if name == "SystemMessage":
        return "system"

    return name

def _get_message_id(msg):
    if isinstance(msg, dict):
        return msg.get("id")
    return getattr(msg, "id", None)

def _is_human_message(msg) -> bool:
    if isinstance(msg, dict):
        return msg.get("role") in {"user", "human"}

    return msg.__class__.__name__ == "HumanMessage"

def _is_ai_message(message) -> bool:
    if isinstance(message, dict):
        return message.get("role") in {
            "assistant",
            "ai",
        }

    return (
        message.__class__.__name__
        == "AIMessage"
    )

def _is_summary_message(msg, session: dict) -> bool:
    msg_id = MessageManage._get_message_id(msg)

    summary_ids = {
        commit.get("summary_id")
        for commit in session["collapse_commits"]
    }

    if msg_id in summary_ids:
        return True

    if isinstance(msg, dict):
        metadata = msg.get("response_metadata", {}) or {}
    else:
        metadata = getattr(msg, "response_metadata", {}) or {}

    compression = metadata.get("context_compression", {}) or {}
    return compression.get("is_summary") is True

def _is_compressed_turn_message(message) -> bool:
    if isinstance(message, dict):
        metadata = message.get(
            "response_metadata",
            {},
        ) or {}
    else:
        metadata = getattr(
            message,
            "response_metadata",
            {},
        ) or {}

    compression = metadata.get(
        "context_compression",
        {},
    ) or {}

    return (
        compression.get("is_compressed_turn")
        is True
    )

def _has_tool_calls(msg) -> bool:
    if isinstance(msg, dict):
        return bool(msg.get("tool_calls"))

    return bool(getattr(msg, "tool_calls", None))
