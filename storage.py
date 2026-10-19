"""SQLite storage layer.

Tables:
    channels     -- known channels and whether they're watched
    last_seen    -- last fetched message id per channel (for incremental fetch)
    shown_posts  -- posts already shown to the user (dedupe across runs)
    settings     -- simple key/value store for future GUI settings
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    channel_id   INTEGER PRIMARY KEY,
    username     TEXT,
    title        TEXT,
    watch        INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS last_seen (
    channel_id   INTEGER PRIMARY KEY,
    last_message_id INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS shown_posts (
    channel_id   INTEGER NOT NULL,
    message_id   INTEGER NOT NULL,
    score        REAL,
    shown_at     TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (channel_id, message_id)
);

CREATE TABLE IF NOT EXISTS channel_profile (
    channel_id   INTEGER PRIMARY KEY,
    topics_sig   TEXT NOT NULL,
    vec          TEXT NOT NULL,
    updated      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS channel_category (
    channel_id INTEGER PRIMARY KEY,
    topic      TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'auto',  -- auto | manual
    updated    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    text_hash TEXT PRIMARY KEY,
    text      TEXT NOT NULL,
    topic     TEXT NOT NULL,   -- topic name, or '' = "none of the topics"
    ts        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS image_feedback (
    img_hash TEXT PRIMARY KEY,   -- md5 of the picture file
    path     TEXT NOT NULL,      -- thumbnail on disk (re-embedded on load)
    topic    TEXT NOT NULL,      -- topic name, or '' = "none of the topics"
    ts       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


@dataclass
class ChannelRow:
    channel_id: int
    username: Optional[str]
    title: Optional[str]
    watch: bool


class Storage:
    """Thin synchronous SQLite wrapper.

    Kept synchronous (not async) intentionally: SQLite access here is fast
    and local; wrapping it in asyncio would add complexity with no real
    benefit, and this stays trivially portable to Termux.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # -- channels ---------------------------------------------------
    def upsert_channel(
        self,
        channel_id: int,
        username: Optional[str],
        title: Optional[str],
        watch: bool = True,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO channels (channel_id, username, title, watch)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    username=excluded.username,
                    title=excluded.title
                """,
                (channel_id, username, title, int(watch)),
            )

    def set_channel_watch(self, channel_id: int, watch: bool) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE channels SET watch=? WHERE channel_id=?",
                (int(watch), channel_id),
            )

    def list_channels(self) -> list[ChannelRow]:
        cur = self._conn.execute("SELECT * FROM channels ORDER BY title")
        return [
            ChannelRow(
                channel_id=r["channel_id"],
                username=r["username"],
                title=r["title"],
                watch=bool(r["watch"]),
            )
            for r in cur.fetchall()
        ]

    # -- manual corrections: "this post belongs to topic X" -------------
    def add_feedback(self, text: str, topic: str) -> None:
        import hashlib
        import time

        h = hashlib.md5(text.strip().encode("utf-8")).hexdigest()
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO feedback(text_hash, text, topic, ts) VALUES (?,?,?,?)",
                (h, text.strip()[:600], topic, time.time()),
            )

    def list_feedback(self, limit: int = 200) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT text, topic FROM feedback ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [(r["text"], r["topic"]) for r in rows]

    def add_image_feedback(self, path: str, topic: str) -> bool:
        """Remember a manual verdict for a picture (picture-only posts have no text to learn from)."""
        import hashlib
        import time

        try:
            with open(path, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
        except OSError:
            return False
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO image_feedback(img_hash, path, topic, ts) VALUES (?,?,?,?)",
                (h, path, topic, time.time()),
            )
        return True

    def list_image_feedback(self, limit: int = 200) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT path, topic FROM image_feedback ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [(r["path"], r["topic"]) for r in rows]

    # -- channel home category: which topic a channel belongs to ------
    def get_channel_category(self, channel_id: int):
        """(topic, source) or None."""
        r = self._conn.execute(
            "SELECT topic, source FROM channel_category WHERE channel_id=?", (channel_id,)
        ).fetchone()
        return (r["topic"], r["source"]) if r else None

    def set_channel_category(self, channel_id: int, topic: str, source: str = "auto") -> None:
        import time

        with self._conn:
            if source == "auto":  # never overwrite the user's own choice
                self._conn.execute(
                    "INSERT INTO channel_category(channel_id, topic, source, updated) VALUES (?,?,?,?) "
                    "ON CONFLICT(channel_id) DO UPDATE SET topic=excluded.topic, updated=excluded.updated "
                    "WHERE channel_category.source != 'manual'",
                    (channel_id, topic, source, time.time()),
                )
            else:
                self._conn.execute(
                    "INSERT OR REPLACE INTO channel_category(channel_id, topic, source, updated) VALUES (?,?,?,?)",
                    (channel_id, topic, source, time.time()),
                )

    def clear_channel_category(self, channel_id: int) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM channel_category WHERE channel_id=?", (channel_id,))

    # -- channel topic profile (what a channel usually posts about) ----
    def get_profile(self, channel_id: int):
        """(topics_signature, vector, updated_epoch) or None."""
        import json

        row = self._conn.execute(
            "SELECT topics_sig, vec, updated FROM channel_profile WHERE channel_id=?",
            (channel_id,),
        ).fetchone()
        if not row:
            return None
        return row["topics_sig"], json.loads(row["vec"]), row["updated"]

    def set_profile(self, channel_id: int, sig: str, vec: list, updated: float) -> None:
        import json

        self._conn.execute(
            "INSERT OR REPLACE INTO channel_profile(channel_id, topics_sig, vec, updated) "
            "VALUES (?,?,?,?)",
            (channel_id, sig, json.dumps(vec), updated),
        )
        self._conn.commit()

    # -- last_seen ----------------------------------------------------
    def get_last_seen(self, channel_id: int) -> int:
        cur = self._conn.execute(
            "SELECT last_message_id FROM last_seen WHERE channel_id=?",
            (channel_id,),
        )
        row = cur.fetchone()
        return int(row["last_message_id"]) if row else 0

    def set_last_seen(self, channel_id: int, message_id: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO last_seen (channel_id, last_message_id)
                VALUES (?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    last_message_id=MAX(last_message_id, excluded.last_message_id)
                """,
                (channel_id, message_id),
            )

    # -- shown_posts --------------------------------------------------
    def mark_shown(self, channel_id: int, message_id: int, score: float) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR IGNORE INTO shown_posts (channel_id, message_id, score)
                VALUES (?, ?, ?)
                """,
                (channel_id, message_id, score),
            )

    def is_shown(self, channel_id: int, message_id: int) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM shown_posts WHERE channel_id=? AND message_id=?",
            (channel_id, message_id),
        )
        return cur.fetchone() is not None

    # -- settings -------------------------------------------------------
    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        cur = self._conn.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = cur.fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, value),
            )
