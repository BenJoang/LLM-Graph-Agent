from copy import deepcopy
from langchain_core.messages import HumanMessage, AIMessage
from collections.abc import Awaitable, Callable
from typing_extensions import TypedDict
from uuid import uuid4
import json
from llm_graph_agent.context.messages import (
    get_message_turn_id,
    mark_ai_message,
)
from llm_graph_agent.context.messages import set_context_state
from llm_graph_agent.context.compression.session import (
    CompressionSession,
    dump_compression_session,
    load_compression_session,
    make_empty_compression_session,
)
from llm_graph_agent.context.compression.tokens import estimate_tokens as _estimate_tokens
from llm_graph_agent.context.compression.snip import (
    history_snip as _history_snip_fn,
    snip_tool_content as _snip_tool_content_fn,
)
from llm_graph_agent.context.compression.prompts import (
    COMPRESSED_TURN_PREFIX,
    ERROR_COLLAPSE_BATCH_SIZE,
    ERROR_NO_ASUMMARIZE_FN,
    ERROR_NO_SUMMARIZE_FN,
    ERROR_RETRY_LEVEL,
    MERGE_SUMMARY_PREFIX,
    SUMMARY_PREFIX,
)
from llm_graph_agent.context.compression.collapse import (
    _adjust_batch_for_tool_calls,
    _apply_context_collapse,
    _find_contiguous_message_ids,
    _find_contiguous_summary_groups,
    _find_next_collapse_batch,
    _make_compressed_turn_message,
    _make_summary_message,
    _plan_retry_summary_merges,
    _plan_retry_turn_collapses,
)
from llm_graph_agent.context.segment import (
    get_content,
    get_message_id,
    has_tool_calls,
    is_ai_message,
    is_compressed_turn_message,
    is_human_message,
    is_summary_message,
    is_tool_message,
    message_role,
    set_content,
)


class MessageManage:
    def __init__(
            self, 
            max_tokens: int = 32768,
            summarize_fn: Callable[[str], str] | None = None,
            asummarize_fn: (Callable[[str], Awaitable[str]] | None) = None,
            collapse_batch_size: int = 6,
        ):

        if collapse_batch_size < 2:
            raise ValueError(ERROR_COLLAPSE_BATCH_SIZE)
        
        self.max_tokens = max_tokens
        self.summarize_fn = summarize_fn
        self.asummarize_fn = asummarize_fn
        self.collapse_batch_size = collapse_batch_size

        self._snip_tokens = int(max_tokens * 0.50)
        self._summarize_tokens = int(max_tokens * 0.70)
        self.retry_target_tokens = int(max_tokens * 0.85)

        self.summarize_start = 2
        self.summarize_end = 4

        self.tool_head_chars = 4000
        self.tool_tail_lines = 3
        self.tool_tail_chars = 1000

        self.cutoff = 2

    def project_committed_context(
        self,
        messages: list,
        compression_session: CompressionSession | None = None,
    ) -> tuple[list, CompressionSession]:
        """只应用已经提交的压缩计划，生成可持久化的模型投影。

        这里不会裁剪工具输出，也不会调用摘要模型或创建新的压缩
        commit，因此事实消息仍只保存在 conversation_events 中。
        """

        session = load_compression_session(compression_session)
        projected = _apply_context_collapse(
            deepcopy(messages),
            session,
        )
        return projected, dump_compression_session(session)

    def prepare_messages_for_query(self, messages: list, compression_session: CompressionSession | None = None) -> tuple[list, bool, CompressionSession]:
        compressed = False
        current_tokens = self.estimate_tokens(messages)

        session = load_compression_session(compression_session)

        #if current_tokens <= self._snip_tokens:
        #    return messages, compressed
        
        messages_for_query = deepcopy(messages)

        messages_for_query = _apply_context_collapse(messages_for_query, session)
        current_tokens = self.estimate_tokens(messages_for_query)

        if current_tokens > self._snip_tokens:
            if self._history_snip(messages_for_query):
                compressed = True
                current_tokens = self.estimate_tokens(messages_for_query)

        if current_tokens > self._summarize_tokens:
            if self._create_context_collapse(messages_for_query, session):
                compressed = True
                current_tokens = self.estimate_tokens(messages_for_query)

        return (
            messages_for_query, 
            compressed,
            dump_compression_session(session),
        )
    
    async def aprepare_messages_for_query(self, messages: list, compression_session: CompressionSession | None = None) -> tuple[list, bool, CompressionSession]:
            compressed = False
            current_tokens = self.estimate_tokens(messages)
    
            session = load_compression_session(compression_session)
    
            #if current_tokens <= self._snip_tokens:
            #    return messages, compressed
            
            messages_for_query = deepcopy(messages)
    
            messages_for_query = _apply_context_collapse(messages_for_query, session)
            current_tokens = self.estimate_tokens(messages_for_query)
    
            if current_tokens > self._snip_tokens:
                if self._history_snip(messages_for_query):
                    compressed = True
                    current_tokens = self.estimate_tokens(messages_for_query)
    
            if current_tokens > self._summarize_tokens:
                if await self._acreate_context_collapse(messages_for_query, session):
                    compressed = True
                    current_tokens = self.estimate_tokens(messages_for_query)
    
            return (
                messages_for_query, 
                compressed,
                dump_compression_session(session),
            )
    
    def estimate_tokens(self, messages: list) -> int:
        """估算 token 数（委托给 tokens 模块）。"""
        return _estimate_tokens(messages)
    
    def compress_for_retry(
            self, 
            messages: list,  
            original_messages: list, 
            level: int,
            compression_session: CompressionSession | None,
            current_turn_id: int,
    ) -> tuple[list, CompressionSession, bool]:
        
        retry_messages = deepcopy(messages)
        session = load_compression_session(compression_session)
        changed = False
        '''已完成'''
        if level == 1:
            changed = self._retry_merge_summary_messages(
                retry_messages,
                session,
            )

        elif level == 2:
            changed = self._retry_snip_all_tool_messages(
                retry_messages,
            )

        elif level == 3:
            changed = self._retry_collapse_old_turns(
                messages=retry_messages,
                session=session,
                current_turn_id=current_turn_id,
            )
        else:
            raise ValueError(
                ERROR_RETRY_LEVEL.format(level=level)
            )
        return (
            retry_messages,
            dump_compression_session(session),
            changed,
        )
    async def acompress_for_retry(
        self,
        messages: list,
        original_messages: list,
        level: int,
        compression_session: (
            CompressionSession | None
        ),
        current_turn_id: int,
    ) -> tuple[list, CompressionSession, bool]:
        retry_messages = deepcopy(messages)

        session = load_compression_session(
            compression_session
        )

        changed = False

        if level == 1:
            changed = await self._aretry_merge_summary_messages(
                retry_messages,
                session,
            )

        elif level == 2:
            # 这里只是本地裁剪，不调用 LLM
            changed = self._retry_snip_all_tool_messages(
                retry_messages
            )

        elif level == 3:
            changed = await self._aretry_collapse_old_turns(
                messages=retry_messages,
                session=session,
                current_turn_id=current_turn_id,
            )

        else:
            raise ValueError(
                ERROR_RETRY_LEVEL.format(level=level)
            )

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
    

    
    def _create_context_collapse(
            self, 
            messages: list, 
            session: dict
    ) -> bool:
        changed = False
        first_compression = True

        while (
            first_compression
            or self.estimate_tokens(messages) > self.max_tokens
        ):
            first_compression = False

            candidate = _find_next_collapse_batch(messages=messages, session=session, collapse_batch_size=self.collapse_batch_size, summarize_start=self.summarize_start, summarize_end=self.summarize_end)

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

            before_tokens = self.estimate_tokens(messages)

            # HumanMessage 只作为摘要背景，不会被替换。
            summary = self._build_simple_summary([
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
            after_tokens = self.estimate_tokens(messages)

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
            if after_tokens <= self.max_tokens:
                break

        return changed

    async def _acreate_context_collapse(
                self, 
                messages: list, 
                session: dict
        ) -> bool:
            changed = False
            first_compression = True
    
            while (
                first_compression
                or self.estimate_tokens(messages) > self.max_tokens
            ):
                first_compression = False
    
                candidate = _find_next_collapse_batch(messages=messages, session=session, collapse_batch_size=self.collapse_batch_size, summarize_start=self.summarize_start, summarize_end=self.summarize_end)
    
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
    
                before_tokens = self.estimate_tokens(messages)
    
                # HumanMessage 只作为摘要背景，不会被替换。
                summary = await self._abuild_simple_summary([
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
                after_tokens = self.estimate_tokens(messages)
    
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
                if after_tokens <= self.max_tokens:
                    break
    
            return changed
    
    def _retry_merge_summary_messages(
        self,
        messages: list,
        session: dict,
    ) -> bool:
        # 必须先冻结计划。
        plans = _plan_retry_summary_merges(
            messages,
            session,
        )

        if not plans:
            return False

        changed = False

        for plan in plans:
            if self._execute_retry_summary_merge(
                messages,
                session,
                plan,
            ):
                changed = True

        return changed

    async def _aretry_merge_summary_messages(
            self,
            messages: list,
            session: dict,
        ) -> bool:
            # 必须先冻结计划。
            plans = _plan_retry_summary_merges(
                messages,
                session,
            )
    
            if not plans:
                return False
    
            changed = False
    
            for plan in plans:
                if await self._aexecute_retry_summary_merge(
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

            set_content(
                message,
                snipped_content,
            )
            changed = True

        return changed
    
    def _retry_collapse_old_turns(
        self,
        messages: list,
        session: dict,
        current_turn_id: int,
    ) -> bool:
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
                and self.estimate_tokens(messages)
                <= self.retry_target_tokens
            ):
                break

            success = (
                self._execute_retry_turn_collapse(
                    messages=messages,
                    session=session,
                    plan=plan,
                )
            )

            if not success:
                # 某轮摘要没有变短或结构失效，
                # 继续尝试下一轮，不能直接 break。
                continue

            first_success = True
            changed = True

        return changed

    async def _aretry_collapse_old_turns(
            self,
            messages: list,
            session: dict,
            current_turn_id: int,
        ) -> bool:
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
                    and self.estimate_tokens(messages)
                    <= self.retry_target_tokens
                ):
                    break
    
                success =await self._aexecute_retry_turn_collapse(
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
    
    
    
    
    


    def _build_simple_summary(self, messages:list) -> str:
        text = self._format_messages_for_summary(messages)

        if not text:
            return ""

        if self.summarize_fn is None:
            raise RuntimeError(ERROR_NO_SUMMARIZE_FN)
        
        return self.summarize_fn(text)

    async def _abuild_simple_summary(
        self,
        messages: list,
    ) -> str:
        text = self._format_messages_for_summary(
            messages
        )

        if not text:
            return ""

        if self.asummarize_fn is None:
            raise RuntimeError(
                ERROR_NO_ASUMMARIZE_FN
            )

        return await self.asummarize_fn(text)
    
    def _format_messages_for_summary(self, messages: list) -> str:
        parts = []

        for msg in messages:
            content = get_content(msg)
            if not content:
                continue

            role = message_role(msg)
            parts.append(
                {
                    "role": role,
                    "content": str(content)
                }
            )

        return json.dumps(parts, ensure_ascii=False, indent=2)



    
    
    def _execute_retry_summary_merge(
        self,
        messages: list,
        session: dict,
        plan: dict,
    ) -> bool:
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

        before_tokens = self.estimate_tokens(messages)

        summary = self._build_simple_summary(
            source_messages
        )

        if not summary:
            return False

        collapse_id = str(uuid4())
        summary_content = (
            MERGE_SUMMARY_PREFIX.format(summary)
        )

        summary_message = _make_summary_message(
            summary_content=summary_content,
            collapse_id=collapse_id,
            turn_id=plan["turn_id"],
            summary_kind="retry_merge",
        )

        messages[start_index:end_index] = [
            summary_message
        ]

        after_tokens = self.estimate_tokens(messages)

        # 摘要没有变短，恢复原消息。
        if after_tokens >= before_tokens:
            messages[
                start_index:start_index + 1
            ] = source_messages

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
        session["collapse_message_ids"].update(
            source_ids
        )

        # 不要将 summary_message.id 加入这里。
        # 它以后仍可再次参与合并。

        return True
    async def _aexecute_retry_summary_merge(
            self,
            messages: list,
            session: dict,
            plan: dict,
        ) -> bool:
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
    
            before_tokens = self.estimate_tokens(messages)
    
            summary = await self._abuild_simple_summary(
                source_messages
            )
    
            if not summary:
                return False
    
            collapse_id = str(uuid4())
            summary_content = (
                MERGE_SUMMARY_PREFIX.format(summary)
            )
    
            summary_message = _make_summary_message(
                summary_content=summary_content,
                collapse_id=collapse_id,
                turn_id=plan["turn_id"],
                summary_kind="retry_merge",
            )
    
            messages[start_index:end_index] = [
                summary_message
            ]
    
            after_tokens = self.estimate_tokens(messages)
    
            # 摘要没有变短，恢复原消息。
            if after_tokens >= before_tokens:
                messages[
                    start_index:start_index + 1
                ] = source_messages
    
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
            session["collapse_message_ids"].update(
                source_ids
            )
    
            # 不要将 summary_message.id 加入这里。
            # 它以后仍可再次参与合并。
    
            return True
    
    def _execute_retry_turn_collapse(
        self,
        messages: list,
        session: dict,
        plan: dict,
    ) -> bool:
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

        before_tokens = self.estimate_tokens(messages)
        summary = self._build_simple_summary(source_messages)

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
        after_tokens = self.estimate_tokens(messages)

        if after_tokens >= before_tokens:
            messages[
                start_index:start_index + 1
            ] = source_messages

            return False

        commit = {
            "collapse_id": collapse_id,
            "kind": "retry_turn_collapse",
            "turn_id": plan["turn_id"],
            "source_message_ids": source_ids,
            "summary_id": compressed_human.id,
            "summary": summary,
            "summary_content": (compressed_human.content),
            "output_message_type": "human",
        }

        session["collapse_commits"].append(commit)

        session["collapse_message_ids"].update(source_ids)

        # compressed_human.id 不加入 consumed IDs。
        # 是否禁止重复压缩由 is_compressed_turn 标签负责。

        return True

    async def _aexecute_retry_turn_collapse(
            self,
            messages: list,
            session: dict,
            plan: dict,
        ) -> bool:
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
    
            before_tokens = self.estimate_tokens(messages)
            summary = await self._abuild_simple_summary(source_messages)
    
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
            after_tokens = self.estimate_tokens(messages)
    
            if after_tokens >= before_tokens:
                messages[
                    start_index:start_index + 1
                ] = source_messages
    
                return False
    
            commit = {
                "collapse_id": collapse_id,
                "kind": "retry_turn_collapse",
                "turn_id": plan["turn_id"],
                "source_message_ids": source_ids,
                "summary_id": compressed_human.id,
                "summary": summary,
                "summary_content": (compressed_human.content),
                "output_message_type": "human",
            }
    
            session["collapse_commits"].append(commit)
    
            session["collapse_message_ids"].update(source_ids)
    
            # compressed_human.id 不加入 consumed IDs。
            # 是否禁止重复压缩由 is_compressed_turn 标签负责。
    
            return True
