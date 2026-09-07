"""LLM 调用重试（async 通道）。

统一处理两类失败：
- APITimeoutError：指数退避后重试（依次等待 1、2、4、8、16 秒）
- 上下文超限（BadRequestError/400 且命中 context length）：调用
  compress_fn 压缩消息后重试，最多 max_context_retries 级降级

全项目已是 async 通道，不再保留 sync 副本。
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from langchain_core.messages import AIMessage
from openai import APITimeoutError, BadRequestError

from llm_graph_agent.context.messages import mark_ai_message


def is_context_overflow_error(e: Exception) -> bool:
    """判断错误是否是"模型上下文超限"（可触发压缩重试）。"""
    body = getattr(e, "body", None)

    if isinstance(body, dict):
        message = str(body.get("message", "")) or str(e)
        param = body.get("param")
    else:
        message = str(body) if body else str(e)
        param = None

    normalized_message = message.lower()

    return (
        isinstance(e, BadRequestError)
        and getattr(e, "status_code", None) == 400
        and (
            param == "input_tokens"
            or "maximum context length" in normalized_message
            or "context length" in normalized_message
        )
    )


async def invoke_with_retry(
    invoke_fn: Callable[[list], Awaitable],
    messages: list,
    original_messages: list,
    compress_fn: Callable[[list, list, int], Awaitable[list]],
    *,
    turn_id: int | None = None,
    max_timeout_retries: int = 5,
    max_context_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
):
    """带重试的模型调用。

    compress_fn 接收 (current_messages, original_messages, level)，
    返回压缩后的消息列表（level 从 1 开始，逐级升级）。
    """
    current_messages = messages

    timeout_retries = 0
    context_retries = 0

    while True:
        try:
            response = await invoke_fn(current_messages)

            if (
                turn_id is not None
                and isinstance(response, AIMessage)
            ):
                response = mark_ai_message(
                    response,
                    turn_id=turn_id,
                )

            return response

        except asyncio.CancelledError:
            raise

        except APITimeoutError as e:
            if timeout_retries >= max_timeout_retries:
                raise RuntimeError(
                    "模型调用超时，重试 "
                    f"{max_timeout_retries} 次后仍失败"
                ) from e

            delay = min(
                max_delay,
                base_delay * (2 ** timeout_retries),
            )

            timeout_retries += 1
            await asyncio.sleep(delay)

        except Exception as e:
            if not is_context_overflow_error(e):
                raise

            if context_retries >= max_context_retries:
                raise RuntimeError(
                    "模型上下文超限，压缩重试 "
                    f"{max_context_retries} 次后仍失败"
                ) from e

            context_retries += 1

            current_messages = await compress_fn(
                current_messages,
                original_messages,
                context_retries,
            )
