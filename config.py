"""Configuration loading for the Telegram Digest tool.

Reads secrets from .env and settings from config.yaml, and exposes them as
typed dataclasses. Same code path is used on Windows and Termux (Android).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv


def _detect_project_root() -> Path:
    """Where config.yaml/.env/session/data live.

    When running as a normal .py file, that's next to this file. When
    running as a PyInstaller-built .exe, `__file__` points into a
    temporary extraction directory (sys._MEIPASS) that's wiped after the
    process exits -- so config/session/db would silently "reset" on every
    launch. In that case, use the folder the .exe itself lives in instead,
    so settings and your Telegram session persist across runs.
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        # Onedir build lives in <project>/dist/TelegramDigest/ -- look upward
        # for the folder that actually holds .env so settings/session/db
        # stay in the project root.
        for candidate in [exe_dir, *list(exe_dir.parents)[:3]]:
            if (candidate / ".env").exists():
                return candidate
        return exe_dir
    return Path(__file__).resolve().parent


PROJECT_ROOT = _detect_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


@dataclass
class ChannelConfig:
    """Per-channel switch. Only channels the user switched OFF need to be
    listed (watch: false); everything else is watched by default, so new
    channels you join are picked up automatically."""

    username: str = ""
    watch: bool = True
    channel_id: Optional[int] = None


@dataclass
class TopicConfig:
    """A user-defined topic: a name plus comma-separated keywords/phrases
    that describe it. Posts are grouped under the topic they match best."""

    name: str
    keywords: list[str] = field(default_factory=list)

    def as_example(self) -> str:
        """Text embedded by the local model to represent this topic."""
        if self.keywords:
            return f"{self.name}: {', '.join(self.keywords)}"
        return self.name


def parse_topics_text(text: str) -> list[TopicConfig]:
    """'Название: слово1, слово2' per line -> topics (keywords optional)."""
    topics: list[TopicConfig] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        name, sep, rest = line.partition(":")
        name = name.strip()
        if not name:
            continue
        keywords = [k.strip() for k in rest.split(",") if k.strip()] if sep else []
        topics.append(TopicConfig(name=name, keywords=keywords))
    return topics


def topics_to_text(topics: list["TopicConfig"]) -> str:
    return "\n".join(
        f"{t.name}: {', '.join(t.keywords)}" if t.keywords else t.name for t in topics
    )


@dataclass
class ProxyConfig:
    # proxy_type: "socks5", "socks4", or "http". Matches what PySocks /
    # Telethon expect.
    proxy_type: str
    host: str
    port: int
    username: str | None = None
    password: str | None = None


@dataclass
class TelegramCredentials:
    api_id: int
    api_hash: str
    phone: str | None = None
    session_path: str = "session/telegram_digest"
    proxy: Optional["ProxyConfig"] = None


@dataclass
class AppConfig:
    check_interval_minutes: int = 40
    similarity_threshold: float = 0.48
    db_path: str = "data/digest.db"
    # "auto" (default): use CUDA on PC if available, CPU on Termux/Android
    # (no CUDA there). "cuda" / "cpu" force a specific device.
    embedding_device: str = "auto"
    topics: list[TopicConfig] = field(default_factory=list)
    interest_examples: list[str] = field(default_factory=list)  # extra free-form examples
    global_exclude_keywords: list[str] = field(default_factory=list)
    channels: list[ChannelConfig] = field(default_factory=list)
    # Speed-related knobs:
    only_unread: bool = True  # only posts that are unread in Telegram
    only_broadcast: bool = True  # skip group chats/megagroups, watch channels only
    first_run_limit: int = 30  # posts per channel on the very first scan
    max_concurrent: int = 6  # channels fetched in parallel
    # Optional local LLM (Ollama) for topic summaries; "" = disabled.
    llm_model: str = ""
    llm_url: str = "http://localhost:11434"
    channel_context_weight: float = 1.0  # 0 = off; how strongly a channel's usual topics bias its posts
    show_images: bool = True  # download pictures from matching posts
    llm_classify: bool = True  # let the LLM verify/assign topics when llm_model is set


def _all_examples(self) -> list[str]:
    return [t.as_example() for t in self.topics] + list(self.interest_examples)


def _topic_name_for(self, example) -> str:
    """Map the matched example text back to a short topic name."""
    if not example:
        return ""
    for t in self.topics:
        if t.as_example() == example:
            return t.name
    return example


AppConfig.all_examples = _all_examples  # type: ignore[attr-defined]
AppConfig.topic_name_for = _topic_name_for  # type: ignore[attr-defined]


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _detect_system_proxy() -> Optional["ProxyConfig"]:
    """Best-effort auto-detect of a proxy to use, when none is set
    explicitly in .env (TG_PROXY_TYPE etc).

    Windows: reads the same system proxy setting used by Internet Options
    / most VPN and proxy client apps (Settings -> Network -> Proxy), via
    the registry. This is exactly what you'd see under "Use a proxy
    server" there. Recognizes a SOCKS entry if present (proxy strings can
    look like "socks=127.0.0.1:1080" or combine multiple protocols
    separated by ";"), otherwise falls back to the plain
    "host:port"/"http=host:port" form.

    Not Windows, or nothing configured there: returns None (connects
    directly, same as before -- set TG_PROXY_* in .env manually if you're
    on Termux/Linux/macOS and need a proxy).
    """
    if sys.platform != "win32":
        return None

    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        )
        try:
            enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
            if not enabled:
                return None
            proxy_server, _ = winreg.QueryValueEx(key, "ProxyServer")
        finally:
            winreg.CloseKey(key)

        if not proxy_server:
            return None

        # proxy_server is either "host:port" (applies to all protocols) or
        # "protocol=host:port;protocol=host:port;...".
        entries: dict[str, str] = {}
        if "=" in proxy_server:
            for part in proxy_server.split(";"):
                part = part.strip()
                if "=" in part:
                    proto, addr = part.split("=", 1)
                    entries[proto.strip().lower()] = addr.strip()
        else:
            entries["*"] = proxy_server.strip()

        chosen_type = "http"
        addr = entries.get("socks")
        if addr:
            chosen_type = "socks5"
        else:
            addr = entries.get("http") or entries.get("*")

        if not addr or ":" not in addr:
            return None
        host, port_raw = addr.rsplit(":", 1)
        try:
            port = int(port_raw)
        except ValueError:
            return None

        logger_msg = (
            f"Auto-detected system proxy ({chosen_type} {host}:{port}) "
            f"from Windows network settings."
        )
        print(f"INFO: {logger_msg}")
        return ProxyConfig(proxy_type=chosen_type, host=host, port=port)
    except Exception:
        return None


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"Missing required environment variable '{name}'. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


def load_credentials(env_path: Path = DEFAULT_ENV_PATH) -> TelegramCredentials:
    """Load Telegram API credentials from a .env file (or existing env)."""
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Fall back to whatever is already in the process environment
        # (useful on Termux where .env may be exported via shell profile).
        load_dotenv()

    api_id_raw = _require_env("TG_API_ID")
    try:
        api_id = int(api_id_raw)
    except ValueError as exc:
        raise ConfigError("TG_API_ID must be an integer") from exc

    api_hash = _require_env("TG_API_HASH")
    phone = os.environ.get("TG_PHONE") or None
    session_path = os.environ.get("TG_SESSION", "session/telegram_digest")

    proxy = None
    proxy_type = os.environ.get("TG_PROXY_TYPE")
    if not proxy_type:
        # Nothing set manually -- try to auto-detect a system proxy
        # (Windows only for now) before falling back to a direct
        # connection. Manual TG_PROXY_* in .env always wins if present.
        proxy = _detect_system_proxy()
    elif proxy_type:
        proxy_host = os.environ.get("TG_PROXY_HOST")
        proxy_port_raw = os.environ.get("TG_PROXY_PORT")
        if not proxy_host or not proxy_port_raw:
            raise ConfigError(
                "TG_PROXY_TYPE is set but TG_PROXY_HOST/TG_PROXY_PORT are "
                "missing. Fill in all three (and optionally "
                "TG_PROXY_USERNAME/TG_PROXY_PASSWORD) in .env, or remove "
                "TG_PROXY_TYPE to connect directly."
            )
        try:
            proxy_port = int(proxy_port_raw)
        except ValueError as exc:
            raise ConfigError("TG_PROXY_PORT must be an integer") from exc
        proxy = ProxyConfig(
            proxy_type=proxy_type.lower(),
            host=proxy_host,
            port=proxy_port,
            username=os.environ.get("TG_PROXY_USERNAME") or None,
            password=os.environ.get("TG_PROXY_PASSWORD") or None,
        )

    return TelegramCredentials(
        api_id=api_id,
        api_hash=api_hash,
        phone=phone,
        session_path=session_path,
        proxy=proxy,
    )


def load_app_config(config_path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load app settings (thresholds, interests, channels) from YAML."""
    if not config_path.exists():
        raise ConfigError(
            f"Config file not found at {config_path}. "
            f"Copy config.example.yaml to config.yaml and edit it."
        )

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    channels_raw = raw.get("channels") or []
    channels = [
        ChannelConfig(
            username=str(c.get("username") or ""),
            watch=bool(c.get("watch", True)),
            channel_id=int(c["channel_id"]) if c.get("channel_id") is not None else None,
        )
        for c in channels_raw
        if c and (c.get("username") or c.get("channel_id") is not None)
    ]

    topics_raw = raw.get("topics") or []
    topics = []
    for t in topics_raw:
        if isinstance(t, dict) and t.get("name"):
            kws = t.get("keywords") or []
            if isinstance(kws, str):
                kws = [k.strip() for k in kws.split(",") if k.strip()]
            topics.append(TopicConfig(name=str(t["name"]), keywords=[str(k) for k in kws]))
        elif isinstance(t, str) and t.strip():
            topics.extend(parse_topics_text(t))

    return AppConfig(
        topics=topics,
        check_interval_minutes=int(raw.get("check_interval_minutes", 40)),
        similarity_threshold=float(raw.get("similarity_threshold", 0.48)),
        db_path=str(raw.get("db_path", "data/digest.db")),
        embedding_device=str(raw.get("embedding_device", "auto")),
        interest_examples=list(raw.get("interest_examples") or []),
        global_exclude_keywords=list(raw.get("global_exclude_keywords") or []),
        channels=channels,
        only_unread=bool(raw.get("only_unread", True)),
        only_broadcast=bool(raw.get("only_broadcast", True)),
        first_run_limit=int(raw.get("first_run_limit", 30)),
        max_concurrent=max(1, int(raw.get("max_concurrent", 6))),
        llm_model=str(raw.get("llm_model", "") or ""),
        llm_url=str(raw.get("llm_url", "http://localhost:11434")),
        llm_classify=bool(raw.get("llm_classify", True)),
        show_images=bool(raw.get("show_images", True)),
        channel_context_weight=float(raw.get("channel_context_weight", 1.0)),
    )


def resolve_path(relative: str) -> Path:
    """Resolve a config-relative path (e.g. db_path, session_path) to absolute."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def save_app_config(config: AppConfig, config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Persist an AppConfig back to config.yaml.

    Used by the GUI's Settings editor (interest examples, threshold,
    interval, channel list) so edits survive a restart. Comments in a
    hand-edited config.yaml are NOT preserved -- this rewrites the file
    from the AppConfig fields as plain YAML.
    """
    raw = {
        "check_interval_minutes": config.check_interval_minutes,
        "similarity_threshold": config.similarity_threshold,
        "db_path": config.db_path,
        "embedding_device": config.embedding_device,
        "topics": [{"name": t.name, "keywords": list(t.keywords)} for t in config.topics],
        "interest_examples": list(config.interest_examples),
        "global_exclude_keywords": list(config.global_exclude_keywords),
        "channels": [
            {
                k: v
                for k, v in (
                    ("channel_id", c.channel_id),
                    ("username", c.username or None),
                    ("watch", c.watch),
                )
                if v is not None
            }
            for c in config.channels
        ],
        "only_unread": config.only_unread,
        "only_broadcast": config.only_broadcast,
        "first_run_limit": config.first_run_limit,
        "max_concurrent": config.max_concurrent,
        "llm_model": config.llm_model,
        "llm_url": config.llm_url,
        "llm_classify": config.llm_classify,
        "show_images": config.show_images,
        "channel_context_weight": config.channel_context_weight,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
