"""上下文压缩引擎（编排层）。

只负责决策与调度：
- 判断当前 token 估算命中了哪个阶段（overflow 模块）
- 依次尝试：应用已提交压缩 → 裁剪工具输出 → 折叠旧轮次
- 处理 LLM 调用失败后的降级压缩（level 1/2/3）

纯结构操作与消息判定在 collapse / segment / snip / serialize 模块，
摘要调用通过注入的 asummarize_fn 完成。全项目已是 async 通道，
不再保留 sync 副本。
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import deepcopy
from uuid import uuid4

from llm_graph_agent.context.compression.collapse import (
    _apply_context_collapse,
    _find_contiguous_message_ids,
    _find_next_collapse_batch,
    _make_compressed_turn_message,
    _make_summary_message,
    _plan_retry_summary_merges,
    _plan_retry_turn_collapses,
)
from llm_graph_agent.context.compression.overflow import (
    CompactThresholds,
    at_or_below_retry_target,
    compute_thresholds,
    should_snip,
    should_summarize,
    still_over_max,
)
from llm_graph_agent.context.compression.prompts import (
    ERROR_COLLAPSE_BATCH_SIZE,
    ERROR_NO_ASUMMARIZE_FN,
    ERROR_RETRY_LEVEL,
    MERGE_SUMMARY_PREFIX,
    SUMMARY_PREFIX,
)
from llm_graph_agent.context.compression.serialize import (
    format_messages_for_summary,
)
from llm_graph_agent.context.compression.session import (
    CompressionSession,
    dump_compression_session,
    load_compression_session,
)
from llm_graph_agent.context.compression.snip import (
    history_snip as _history_snip_fn,
    snip_tool_content as _snip_tool_content_fn,
)
from llm_graph_agent.context.compression.tokens import (
    estimate_tokens,
)
from llm_graph_agent.context.messages import (
    get_message_turn_id,
)
from llm_graph_agent.context.segment import (
    get_content,
    get_message_id,
    is_ai_message,
    is_compressed_turn_message,
    is_human_message,
    is_summary_message,
    is_tool_message,
    set_content,
)


class MessageManage:
    """上下文压缩引擎。

    用法：
        manager = MessageManage(
            max_tokens=context_window_tokens,
            asummarize_fn=summarize_with_main_model,
        )
        messages, compressed, session = (
            await manager.prepare_messages_for_query(messages, session)
        )
    """

    def __init__(
        self,
        max_tokens: int = 32768,
        asummarize_fn: Callable[[str], Awaitable[str]] | None = None,
        collapse_batch_size: int = 6,
    ):
        if collapse_batch_size < 2:
            raise ValueError(ERROR_COLLAPSE_BATCH_SIZE)

        self.thresholds: CompactThresholds = compute_thresholds(max_tokens)
        self.asummarize_fn = asummarize_fn
        self.collapse_batch_size = collapse_batch_size

        # 折叠时跳过头部/尾部的轮次数
        self.summarize_start = 2
        self.summarize_end = 4

        # 工具输出裁剪参数
        self.tool_head_chars = 4000
        self.tool_tail_lines = 3
        self.tool_tail_chars = 1000

        # history_snip 跳过最近 cutoff 条消息
        self.cutoff = 2

    def project_committed_context(
        self,
        messages: list,
        compression_session: CompressionSession | None = None,
    ) -> tuple[list, CompressionSession]:
        """只应用已提交的压缩计划，生成可持久化的模型投影。

        不裁剪工具输出、不调用摘要模型、不创建新的压缩 commit，
        因此事实消息仍只保存在 conversation_events 中。
        """
        session = load_compression_session(compression_session)
        projected = _apply_context_collapse(
            deepcopy(messages),
            session,
        )
        return projected, dump_compression_session(session)

    async def prepare_messages_for_query(
        self,
        messages: list,
        compression_session: CompressionSession | None = None,
    ) -> tuple[list, bool, CompressionSession]:
        """把原始消息准备成可送给模型的消息。

        依次尝试：应用已提交压缩 → 裁剪超长工具输出 → 折叠旧轮次摘要。
        返回 (消息, 是否压缩过, 更新后的压缩会话)。
        """
        compressed = False
        session = load_compression_session(compression_session)

        messages_for_query = deepcopy(messages)
        messages_for_query = _apply_context_collapse(messages_for_query, session)
        current_tokens = estimate_tokens(messages_for_query)

        if should_snip(current_tokens, self.thresholds):
            if self._history_snip(messages_for_query):
                compressed = True
                current_tokens = estimate_tokens(messages_for_query)

        if should_summarize(current_tokens, self.thresholds):
            if await self._create_context_collapse(messages_for_query, session):
                compressed = True
                current_tokens = estimate_tokens(messages_for_query)

        return (
            messages_for_query,
            compressed,
            dump_compression_session(session),
        )

    def estimate_tokens(self, messages: list) -> int:
        """估算 token 数（委托 tokens 模块）。"""
        return estimate_tokens(messages)

    async def compress_for_retry(
        self,
        messages: list,
        original_messages: list,
        level: int,
        compression_session: CompressionSession | None,
        current_turn_id: int,
    ) -> tuple[list, CompressionSession, bool]:
        """LLM 调用超限后的降级压缩。

        level 1：合并相邻摘要
        level 2：裁剪所有工具输出（纯本地，不调用 LLM）
        level 3：折叠旧轮次
        """
        retry_messages = deepcopy(messages)
        session = load_compression_session(compression_session)
        changed = False

        if level == 1:
            changed = await self._retry_merge_summary_messages(
                retry_messages,
                session,
            )
        elif level == 2:
            changed = self._retry_snip_all_tool_messages(retry_messages)
        elif level == 3:
            changed = await self._retry_collapse_old_turns(
                messages=retry_messages,
                session=session,
                current_turn_id=current_turn_id,
            )
        else:
            raise ValueError(ERROR_RETRY_LEVEL.format(level=level))

        return (
            retry_messages,
            dump_compression_session(session),
            changed,
        )

    def _history_snip(self, messages: list) -> bool:
        """裁剪历史中的超长工具输出（委托 snip 模块）。"""
        return _history_snip_fn(
            messages,
            cutoff=self.cutoff,
            tool_head_chars=self.tool_head_chars,
            tool_tail_lines=self.tool_tail_lines,
            tool_tail_chars=self.tool_tail_chars,
            is_tool_message=is_tool_message,
            get_content=get_content,
            set_content=set_content,
        )

    def _snip_tool_content(
        self,
        content: str,
    ) -> tuple[str, bool]:
        """裁剪单个工具输出（委托 snip 模块）。"""
        return _snip_tool_content_fn(
            content,
            tool_head_chars=self.tool_head_chars,
            tool_tail_lines=self.tool_tail_lines,
            tool_tail_chars=self.tool_tail_chars,
        )

    async def _create_context_collapse(
        self,
        messages: list,
        session: dict,
    ) -> bool:
        """把旧轮次折叠成摘要，直到不再超过 max_tokens 或无候选可折。"""
        changed = False
        first_compression = True

        while (
            first_compression
            or still_over_max(estimate_tokens(messages), self.thresholds)
        ):
            first_compression = False

            candidate = _find_next_collapse_batch(
                messages=messages,
                session=session,
                collapse_batch_size=self.collapse_batch_size,
                summarize_start=self.summarize_start,
                summarize_end=self.summarize_end,
            )

            if candidate is None:
                break

            start_index, end_index, anchor_human = candidate
            source_messages = messages[start_index:end_index]
            turn_id = get_message_turn_id(anchor_human)

            # 压缩一条消息通常没有意义。
            if len(source_messages) <= 1:
                break

            source_ids = [
                get_message_id(msg)
                for msg in source_messages
            ]

            # 当前 commit 的恢复逻辑依赖 ID。
            if any(msg_id is None for msg_id in source_ids):
                break

            before_tokens = estimate_tokens(messages)

            # HumanMessage 只作为摘要背景，不会被替换。
            summary = await self._build_simple_summary([
                anchor_human,
                *source_messages,
            ])

            if not summary:
                break

            collapse_id = str(uuid4())
            summary_content = SUMMARY_PREFIX.format(summary)

            summary_message = _make_summary_message(
                summary_content=summary_content,
                collapse_id=collapse_id,
                turn_id=turn_id,
                summary_kind="normal",
            )

            # 先临时替换，检查摘要是否真的减少 token。
            messages[start_index:end_index] = [summary_message]
            after_tokens = estimate_tokens(messages)

            if after_tokens >= before_tokens:
                messages[start_index:start_index + 1] = source_messages
                break

            commit = {
                "collapse_id": collapse_id,
                "kind": "normal",
                "turn_id": turn_id,
                "summary_id": summary_message.id,
                "source_message_ids": source_ids,
                "first_archived_id": source_ids[0],
                "last_archived_id": source_ids[-1],
                "summary": summary,
                "summary_content": summary_message.content,
            }

            session["collapse_commits"].append(commit)
            session["collapse_message_ids"].update(source_ids)

            changed = True

            # 第一次压缩后，只有仍超过 max_tokens 才继续。
            if not still_over_max(after_tokens, self.thresholds):
                break

        return changed

    async def _retry_merge_summary_messages(
        self,
        messages: list,
        session: dict,
    ) -> bool:
        """合并两段及以上相邻摘要为一条新摘要。"""
        # 必须先冻结计划。
        plans = _plan_retry_summary_merges(
            messages,
            session,
        )

        if not plans:
            return False

        changed = False

        for plan in plans:
            if await self._execute_retry_summary_merge(
                messages,
                session,
                plan,
            ):
                changed = True

        return changed

    def _retry_snip_all_tool_messages(
        self,
        messages: list,
    ) -> bool:
        """裁剪所有超长工具输出（纯本地，不调用 LLM）。"""
        changed = False

        for message in messages:
            if not is_tool_message(message):
                continue

            content = get_content(message)

            if not isinstance(content, str):
                continue

            snipped_content, content_changed = (
                self._snip_tool_content(content)
            )

            if not content_changed:
                continue

            set_content(message, snipped_content)
            changed = True

        return changed

    async def _retry_collapse_old_turns(
        self,
        messages: list,
        session: dict,
        current_turn_id: int,
    ) -> bool:
        """折叠旧轮次（Human+AI 各一条），直到达到目标 token 或没有可折轮次。"""
        plans = _plan_retry_turn_collapses(
            messages=messages,
            current_turn_id=current_turn_id,
        )

        if not plans:
            return False

        changed = False
        first_success = False

        for plan in plans:
            # 第一次成功压缩后，如果已经达到目标，
            # 就不再损失更多历史信息。
            if (
                first_success
                and at_or_below_retry_target(
                    estimate_tokens(messages), self.thresholds
                )
            ):
                break

            success = await self._execute_retry_turn_collapse(
                messages=messages,
                session=session,
                plan=plan,
            )

            if not success:
                # 某轮摘要没有变短或结构失效，
                # 继续尝试下一轮，不能直接 break。
                continue

            first_success = True
            changed = True

        return changed

    async def _build_simple_summary(
        self,
        messages: list,
    ) -> str:
        """把消息序列化后交给 asummarize_fn 生成摘要文本。"""
        text = format_messages_for_summary(messages)

        if not text:
            return ""

        if self.asummarize_fn is None:
            raise RuntimeError(ERROR_NO_ASUMMARIZE_FN)

        return await self.asummarize_fn(text)

    async def _execute_retry_summary_merge(
        self,
        messages: list,
        session: dict,
        plan: dict,
    ) -> bool:
        """执行一条摘要合并计划；摘要未变短则回滚并返回 False。"""
        source_ids = plan["source_message_ids"]

        if len(source_ids) <= 1:
            return False

        matched = _find_contiguous_message_ids(
            messages,
            source_ids,
        )

        if matched is None:
            return False

        start_index, end_index = matched
        source_messages = messages[start_index:end_index]

        if not all(
            is_summary_message(message, session)
            for message in source_messages
        ):
            return False

        before_tokens = estimate_tokens(messages)

        summary = await self._build_simple_summary(source_messages)

        if not summary:
            return False

        collapse_id = str(uuid4())
        summary_content = MERGE_SUMMARY_PREFIX.format(summary)

        summary_message = _make_summary_message(
            summary_content=summary_content,
            collapse_id=collapse_id,
            turn_id=plan["turn_id"],
            summary_kind="retry_merge",
        )

        messages[start_index:end_index] = [summary_message]

        after_tokens = estimate_tokens(messages)

        # 摘要没有变短，恢复原消息。
        if after_tokens >= before_tokens:
            messages[start_index:start_index + 1] = source_messages
            return False

        commit = {
            "collapse_id": collapse_id,
            "kind": "retry_merge",
            "turn_id": plan["turn_id"],
            "anchor_human_id": plan["anchor_human_id"],

            # Level 1 的 source 就是旧摘要的 ID。
            "source_message_ids": source_ids,
            "source_summary_ids": source_ids.copy(),

            "summary_id": summary_message.id,
            "summary": summary,
            "summary_content": summary_message.content,
        }

        session["collapse_commits"].append(commit)

        # 旧摘要已被新摘要消费。
        session["collapse_message_ids"].update(source_ids)

        # 不要将 summary_message.id 加入这里。
        # 它以后仍可再次参与合并。

        return True

    async def _execute_retry_turn_collapse(
        self,
        messages: list,
        session: dict,
        plan: dict,
    ) -> bool:
        """执行一条旧轮次折叠计划；摘要未变短则回滚并返回 False。"""
        source_ids = plan["source_message_ids"]

        matched = _find_contiguous_message_ids(
            messages,
            source_ids,
        )

        if matched is None:
            return False

        start_index, end_index = matched
        source_messages = messages[start_index:end_index]

        if len(source_messages) != 2:
            return False

        human_message, ai_message = source_messages

        if not is_human_message(human_message):
            return False

        if not is_ai_message(ai_message):
            return False

        if is_compressed_turn_message(human_message):
            return False

        before_tokens = estimate_tokens(messages)
        summary = await self._build_simple_summary(source_messages)

        if not summary:
            return False

        collapse_id = str(uuid4())

        compressed_content = (
            "[历史轮次压缩摘要]\n"
            f"{summary}"
        )

        compressed_human = (
            _make_compressed_turn_message(
                content=compressed_content,
                collapse_id=collapse_id,
                turn_id=plan["turn_id"],
            )
        )

        messages[start_index:end_index] = [compressed_human]
        after_tokens = estimate_tokens(messages)

        if after_tokens >= before_tokens:
            messages[start_index:start_index + 1] = source_messages
            return False

        commit = {
            "collapse_id": collapse_id,
            "kind": "retry_turn_collapse",
            "turn_id": plan["turn_id"],
            "source_message_ids": source_ids,
            "summary_id": compressed_human.id,
            "summary": summary,
            "summary_content": compressed_human.content,
            "output_message_type": "human",
        }

        session["collapse_commits"].append(commit)
        session["collapse_message_ids"].update(source_ids)

        # compressed_human.id 不加入 consumed IDs。
        # 是否禁止重复压缩由 is_compressed_turn 标签负责。

        return True
