"""数据库建表 DDL（schema）。

SQLite 和 PostgreSQL 各自独立的建表语句。
由对应的 store 在 setup() 时执行。
"""
from __future__ import annotations


SQLITE_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS conversation_sessions (
        session_id TEXT PRIMARY KEY,
        next_turn_no INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_runs (
        run_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL
            REFERENCES conversation_sessions(session_id)
            ON DELETE CASCADE,
        turn_no INTEGER NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN (
                'running',
                'completed',
                'cancelled',
                'interrupted',
                'failed'
            )
        ),
        base_event_id INTEGER NOT NULL DEFAULT 0,
        checkpoint_thread_id TEXT,
        heartbeat_at TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        error TEXT,
        UNIQUE (session_id, turn_no),
        UNIQUE (session_id, run_id)
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS
        uq_conversation_runs_active_session
    ON conversation_runs(session_id)
    WHERE status = 'running'
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        turn_no INTEGER NOT NULL,
        ordinal INTEGER NOT NULL,
        message_id TEXT NOT NULL,
        event_type TEXT NOT NULL CHECK (
            event_type IN (
                'user',
                'assistant',
                'tool',
                'system'
            )
        ),
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id, run_id)
            REFERENCES conversation_runs(session_id, run_id)
            ON DELETE CASCADE,
        UNIQUE (run_id, ordinal),
        UNIQUE (run_id, message_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS
        idx_conversation_events_session
    ON conversation_events(session_id, event_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_context (
        session_id TEXT PRIMARY KEY
            REFERENCES conversation_sessions(session_id)
            ON DELETE CASCADE,
        through_event_id INTEGER NOT NULL DEFAULT 0,
        version INTEGER NOT NULL DEFAULT 1,
        projected_messages TEXT NOT NULL DEFAULT '[]',
        compression_session TEXT NOT NULL DEFAULT '{}',
        updated_at TEXT NOT NULL
    )
    """
)


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS conversation_sessions (
        session_id TEXT PRIMARY KEY,
        next_turn_no BIGINT NOT NULL DEFAULT 1,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_runs (
        run_id TEXT PRIMARY KEY,

        session_id TEXT NOT NULL
            REFERENCES conversation_sessions(session_id)
            ON DELETE CASCADE,

        turn_no BIGINT NOT NULL,

        status TEXT NOT NULL CHECK (
            status IN (
                'running',
                'completed',
                'cancelled',
                'interrupted',
                'failed'
            )
        ),

        base_event_id BIGINT NOT NULL DEFAULT 0,
        checkpoint_thread_id TEXT,

        heartbeat_at TIMESTAMPTZ,
        started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        finished_at TIMESTAMPTZ,
        error TEXT,

        UNIQUE (session_id, turn_no),
        UNIQUE (session_id, run_id)
    )
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS
        uq_conversation_runs_active_session
    ON conversation_runs(session_id)
    WHERE status = 'running'
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_events (
        event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

        session_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        turn_no BIGINT NOT NULL,
        ordinal INTEGER NOT NULL,

        message_id TEXT NOT NULL,

        event_type TEXT NOT NULL CHECK (
            event_type IN (
                'user',
                'assistant',
                'tool',
                'system'
            )
        ),

        payload JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

        FOREIGN KEY (session_id, run_id)
            REFERENCES conversation_runs(session_id, run_id)
            ON DELETE CASCADE,

        UNIQUE (run_id, ordinal),
        UNIQUE (run_id, message_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS
        idx_conversation_events_session
    ON conversation_events(session_id, event_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_context (
        session_id TEXT PRIMARY KEY
            REFERENCES conversation_sessions(session_id)
            ON DELETE CASCADE,
        through_event_id BIGINT NOT NULL DEFAULT 0,
        version BIGINT NOT NULL DEFAULT 1,
        projected_messages JSONB NOT NULL DEFAULT '[]'::jsonb,
        compression_session JSONB NOT NULL DEFAULT '{}'::jsonb,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """
)
