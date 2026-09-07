"""持久化层的纯工具/转换函数。

不含任何数据库连接逻辑，只做时间转换、URL 解析、行记录转换。
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from urllib.parse import unquote

from llm_graph_agent.persistence.models import EventRecord, RunRecord


def sqlite_path_from_url(database_url: str) -> str:
    """将 sqlite:/// URL 转换为 sqlite3 可接受的路径。"""

    prefix = "sqlite:///"
    if not database_url.lower().startswith(prefix):
        raise ValueError(
            "SQLite URL 必须以 sqlite:/// 开头"
        )

    raw_path = unquote(database_url[len(prefix):])
    if not raw_path:
        raise ValueError("SQLite URL 缺少数据库路径")
    if "?" in raw_path or "#" in raw_path:
        raise ValueError("SQLite URL 暂不支持 query 或 fragment")
    if raw_path == ":memory:":
        return raw_path

    if re.match(r"^[A-Za-z]:[/\\]", raw_path):
        return str(Path(raw_path))

    return str(Path(raw_path).expanduser().resolve())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _datetime_to_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _datetime_from_text(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _run_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        session_id=row["session_id"],
        turn_no=int(row["turn_no"]),
        status=row["status"],
        base_event_id=int(row["base_event_id"]),
        checkpoint_thread_id=row["checkpoint_thread_id"],
        heartbeat_at=_datetime_from_text(row["heartbeat_at"]),
        started_at=cast(
            datetime,
            _datetime_from_text(row["started_at"]),
        ),
        finished_at=_datetime_from_text(row["finished_at"]),
        error=row["error"],
    )


def _event_record(row: sqlite3.Row) -> EventRecord:
    payload = json.loads(row["payload"])
    if not isinstance(payload, dict):
        raise ValueError(
            f"event_id={row['event_id']} 的 payload 不是 JSON object"
        )

    return EventRecord(
        event_id=int(row["event_id"]),
        session_id=row["session_id"],
        run_id=row["run_id"],
        turn_no=int(row["turn_no"]),
        ordinal=int(row["ordinal"]),
        message_id=row["message_id"],
        event_type=row["event_type"],
        payload=payload,
        created_at=cast(
            datetime,
            _datetime_from_text(row["created_at"]),
        ),
    )


