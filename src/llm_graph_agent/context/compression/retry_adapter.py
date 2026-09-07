"""重试用压缩适配器：把 engine 的降级压缩包装成一个可调入重试循环的回调。

调用方把 adapter 传进 retry 循环后，每次需要压缩时就调它的 async acall，
它内部持有当前压缩会话，压缩后自动更新，供下一次降级继续使用。
全项目已是 async 通道，只保留 acall。
"""
from __future__ import annotations

from llm_graph_agent.context.compression.engine import (
    CompressionSession,
    MessageManage,
)


class CompressionRetryAdapter:
    def __init__(
        self,
        message_manage: MessageManage,
        compression_session: CompressionSession | None,
        current_turn_id: int,
    ):
        self.message_manage = message_manage
        self.compression_session = compression_session
        self.current_turn_id = current_turn_id

        self.last_level: int | None = None
        self.last_changed = False
        self.last_estimated_tokens: int | None = None

    async def acall(
        self,
        current_messages: list,
        original_messages: list,
        level: int,
    ) -> list:
        (
            retry_messages,
            updated_session,
            changed,
        ) = await self.message_manage.compress_for_retry(
            messages=current_messages,
            original_messages=original_messages,
            level=level,
            compression_session=self.compression_session,
            current_turn_id=self.current_turn_id,
        )

        self.compression_session = updated_session
        self.last_level = level
        self.last_changed = changed
        self.last_estimated_tokens = self.message_manage.estimate_tokens(
            retry_messages
        )

        return retry_messages
