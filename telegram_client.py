"""Telethon-based Telegram client: login, channel discovery, incremental fetch.

Async throughout. Same module is used on Windows and Termux (Android) --
Telethon itself is cross-platform, and nothing here is OS-specific.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

from telethon import TelegramClient
from telethon.tl.types import Channel, Chat

from config import TelegramCredentials, resolve_path


@dataclass
class DiscoveredChannel:
    channel_id: int
    username: Optional[str]
    title: str
    broadcast: bool = True  # True = real channel, False = group/megagroup
    read_max_id: int = 0  # last message id you have read in Telegram
    unread_count: int = 0  # unread messages in this chat
    archived: bool = False  # lives in the Telegram archive folder
    entity: object = None  # cached Telethon entity, avoids an extra lookup


@dataclass
class Post:
    channel_id: int
    channel_title: str
    channel_username: Optional[str]
    message_id: int
    date: datetime
    text: str
    has_media: bool
    has_photo: bool = False
    has_video: bool = False
    image_path: Optional[str] = None  # local cached thumbnail, filled in by the pipeline
    message: object = field(default=None, repr=False, compare=False)  # Telethon message

    @property
    def link(self) -> str:
        if self.channel_username:
            return f"https://t.me/{self.channel_username}/{self.message_id}"
        # Private/no-username channel: use the internal c/<id>/<msg> form.
        internal_id = str(self.channel_id).removeprefix("-100")
        return f"https://t.me/c/{internal_id}/{self.message_id}"

    @property
    def preview(self) -> str:
        text = " ".join(self.text.split())
        return text[:250]


def _build_proxy_tuple(creds: TelegramCredentials):
    """Convert ProxyConfig into the (proxy_type, host, port, ...) tuple
    Telethon/PySocks expect, or None if no proxy is configured.

    Required for connecting from places where Telegram's servers aren't
    directly reachable (e.g. Russia) -- same SOCKS5/HTTP proxy setup you'd
    use in Telegram Desktop or any other Telegram client.
    """
    if creds.proxy is None:
        return None

    import socks  # PySocks; only imported when actually needed

    proxy_type_map = {
        "socks5": socks.SOCKS5,
        "socks4": socks.SOCKS4,
        "http": socks.HTTP,
    }
    proxy_type = proxy_type_map.get(creds.proxy.proxy_type)
    if proxy_type is None:
        raise ValueError(
            f"Unsupported TG_PROXY_TYPE '{creds.proxy.proxy_type}'. "
            f"Use socks5, socks4, or http."
        )
    return (
        proxy_type,
        creds.proxy.host,
        creds.proxy.port,
        True,  # rdns
        creds.proxy.username,
        creds.proxy.password,
    )


class TelegramDigestClient:
    """Thin async wrapper around Telethon for this app's needs."""

    def __init__(self, creds: TelegramCredentials) -> None:
        self._creds = creds
        session_path = resolve_path(creds.session_path)
        # The session is an SQLite file. On Windows keep it on the local
        # system drive (%LOCALAPPDATA%): SQLite locking is unreliable on
        # some other drives (mapped/virtual/cloud-synced), which shows up
        # as "database is locked" while logging in.
        import os

        local = os.environ.get("LOCALAPPDATA")
        if os.name == "nt" and local:
            session_path = Path(local) / "TelegramDigest" / session_path.name
        session_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = TelegramClient(
            str(session_path),
            creds.api_id,
            creds.api_hash,
            proxy=_build_proxy_tuple(creds),
        )

    async def __aenter__(self) -> "TelegramDigestClient":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    async def start(
        self,
        phone_callback=None,
        code_callback=None,
        password_callback=None,
    ) -> None:
        """Connect and ensure the user is logged in.

        By default (CLI / Termux) Telethon prompts via stdin. The GUI passes
        callbacks instead, because a frozen GUI exe has no usable stdin
        ("input(): lost sys.stdin"). Callbacks are plain callables that
        return a string.
        """
        kwargs = {}
        phone = self._creds.phone
        kwargs["phone"] = phone if phone else (phone_callback or None)
        if code_callback:
            kwargs["code_callback"] = code_callback
        if password_callback:
            kwargs["password"] = password_callback
        if not kwargs["phone"]:
            del kwargs["phone"]
        await self.client.start(**kwargs)  # type: ignore[arg-type]

    async def download_thumb(self, post: "Post", cache_dir: Path, max_side: int = 1280):
        """Download a post's picture (or a video's preview frame), shrink it
        and cache it on disk. Returns the file path, or None if the post has
        no picture. Never raises."""
        try:
            if post.message is None or not (post.has_photo or post.has_video):
                return None
            cache_dir.mkdir(parents=True, exist_ok=True)
            out = cache_dir / f"{post.channel_id}_{post.message_id}.jpg"
            if out.exists():
                return str(out)
            thumb = -1 if (post.has_video and not post.has_photo) else None
            data = await self.client.download_media(post.message, file=bytes, thumb=thumb)
            if not data:
                return None
            import io

            from PIL import Image

            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.thumbnail((max_side, max_side))
            img.save(out, "JPEG", quality=88)
            return str(out)
        except Exception:
            return None

    async def stop(self) -> None:
        await self.client.disconnect()

    async def discover_channels(self) -> list[DiscoveredChannel]:
        """List ALL channels/supergroups the user is a member of, including
        archived ones (folder 1) -- main list (folder 0) and archive are
        requested explicitly and merged."""
        results: list[DiscoveredChannel] = []
        seen: set[int] = set()
        for folder, archived in ((0, False), (1, True)):
            try:
                dialogs = self.client.iter_dialogs(folder=folder)
                async for dialog in dialogs:
                    if dialog.id in seen:
                        continue
                    entity = dialog.entity
                    if isinstance(entity, Channel) or (
                        isinstance(entity, Chat) and getattr(entity, "megagroup", False)
                    ):
                        seen.add(dialog.id)
                        results.append(
                            DiscoveredChannel(
                                channel_id=dialog.id,
                                username=getattr(entity, "username", None),
                                title=dialog.name or "",
                                broadcast=bool(getattr(entity, "broadcast", False)),
                                read_max_id=int(getattr(dialog.dialog, "read_inbox_max_id", 0) or 0),
                                unread_count=int(getattr(dialog, "unread_count", 0) or 0),
                                entity=entity,
                                archived=archived,
                            )
                        )
            except Exception:
                if folder == 0:
                    raise  # main list failing is a real error
                # archive listing is best-effort
        return results

    async def fetch_new_posts(
        self,
        channel_id: int,
        channel_title: str,
        channel_username: Optional[str],
        last_seen_id: int,
        limit: int = 200,
        entity=None,
    ) -> AsyncIterator[Post]:
        """Yield posts newer than last_seen_id, oldest first.

        `limit` caps how far back we look per channel per run, to avoid a
        huge backfill after a long time offline.
        """
        if entity is None:
            entity = await self.client.get_entity(channel_id)
        new_messages = []
        async for message in self.client.iter_messages(entity, limit=limit):
            if message.id <= last_seen_id:
                break
            if not (message.text or message.media):
                continue
            new_messages.append(message)

        for message in reversed(new_messages):
            yield Post(
                channel_id=channel_id,
                channel_title=channel_title,
                channel_username=channel_username,
                message_id=message.id,
                date=message.date,
                text=message.text or "",
                has_media=message.media is not None,
                has_photo=bool(getattr(message, "photo", None)),
                has_video=bool(getattr(message, "video", None) or getattr(message, "gif", None)),
                message=message,
            )
