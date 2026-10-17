"""Termux/Android entry point.

Same core pipeline as main.py (config.py, telegram_client.py,
embed_filter.py, storage.py, pipeline.py) -- nothing Windows-specific is
imported here. Two modes:

    python termux_main.py           # loop forever, checking every
                                     # check_interval_minutes, notifying
                                     # via termux-notification
    python termux_main.py --once    # single pass, then exit (for
                                     # scheduling via cron/termux-job-scheduler
                                     # instead of a long-lived loop)

Notifications use termux-notification (from the termux-api package, paired
with the Termux:API app) with a tap action that opens the post directly in
Telegram via termux-open-url. If termux-notification isn't available (e.g.
running this on a plain Linux box for testing), notify.py degrades to
plyer or just logs, so this still runs fine.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time

from config import ConfigError, load_app_config, load_credentials, resolve_path
from embed_filter import build_filter
from notify import is_termux, notify
from pipeline import ScoredPost, check_once
from storage import Storage
from telegram_client import TelegramDigestClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("termux_main")


def print_digest(scored_posts: list[ScoredPost], app_config) -> None:
    if not scored_posts:
        print("Новых релевантных постов нет.")
        return
    print(f"\n=== {len(scored_posts)} релевантных пост(ов) ===\n")

    by_topic: dict[str, list[ScoredPost]] = {}
    for sp in scored_posts:
        topic = app_config.topic_name_for(sp.result.matched_example) or "(без темы)"
        by_topic.setdefault(topic, []).append(sp)

    for topic, posts in by_topic.items():
        print(f"--- {topic} ({len(posts)}) ---")
        for sp in posts:
            post = sp.post
            print(f"[{sp.result.score:.2f}] {post.channel_title} - {post.date:%Y-%m-%d %H:%M}")
            print(f"  {post.preview}")
            print(f"  {post.link}")
            print()


def notify_posts(scored_posts: list[ScoredPost]) -> None:
    """One notification per post (Termux notifications are cheap and each
    one can carry its own tap-to-open link), capped to avoid spamming the
    notification shade after a long time offline."""
    MAX_NOTIFICATIONS = 10
    if not scored_posts:
        return
    for sp in scored_posts[:MAX_NOTIFICATIONS]:
        post = sp.post
        title = f"{post.channel_title} ({sp.result.score:.2f})"
        notify(title, post.preview, url=post.link)
    remaining = len(scored_posts) - MAX_NOTIFICATIONS
    if remaining > 0:
        notify("Telegram Digest", f"...и ещё {remaining} пост(ов), см. вывод программы")


async def run_once(limit_per_channel: int) -> list[ScoredPost]:
    creds = load_credentials()
    app_config = load_app_config()

    if not app_config.all_examples():
        raise ConfigError(
            "config.yaml has no interest_examples. Add 5-20 example texts "
            "describing what's interesting to you."
        )

    db_path = resolve_path(app_config.db_path)
    storage = Storage(db_path)

    logger.info("Loading local relevance filter (this may take a moment on first run)...")
    relevance_filter = build_filter(
        app_config.all_examples(),
        threshold=app_config.similarity_threshold,
        device=app_config.embedding_device,
                model_name=app_config.embedding_model,
        topics=app_config.topics,
        llm_model=app_config.llm_model if app_config.llm_classify else "",
        llm_url=app_config.llm_url,
    )

    async with TelegramDigestClient(creds) as client:
        scored_posts = await check_once(
            client=client,
            storage=storage,
            app_config=app_config,
            relevance_filter=relevance_filter,
            limit_per_channel=limit_per_channel,
        )

    storage.close()
    return scored_posts


async def run_loop(limit_per_channel: int) -> None:
    """Long-lived loop: login once, then check on config's interval.

    Keep this alive on Android with `termux-wake-lock` (prevents Doze from
    suspending Termux) and/or the Termux:Boot app to auto-start it, since
    Android will otherwise kill background processes.
    """
    creds = load_credentials()
    app_config = load_app_config()

    if not app_config.all_examples():
        raise ConfigError(
            "config.yaml has no interest_examples. Add 5-20 example texts "
            "describing what's interesting to you."
        )

    db_path = resolve_path(app_config.db_path)
    storage = Storage(db_path)

    logger.info("Loading local relevance filter (this may take a moment on first run)...")
    relevance_filter = build_filter(
        app_config.all_examples(),
        threshold=app_config.similarity_threshold,
        device=app_config.embedding_device,
                model_name=app_config.embedding_model,
        topics=app_config.topics,
        llm_model=app_config.llm_model if app_config.llm_classify else "",
        llm_url=app_config.llm_url,
    )

    async with TelegramDigestClient(creds) as client:
        logger.info(
            "Running. Checking every %d minute(s). Ctrl+C to stop.",
            app_config.check_interval_minutes,
        )
        while True:
            try:
                scored_posts = await check_once(
                    client=client,
                    storage=storage,
                    app_config=app_config,
                    relevance_filter=relevance_filter,
                    limit_per_channel=limit_per_channel,
                )
                print_digest(scored_posts, app_config)
                notify_posts(scored_posts)
            except Exception:
                logger.exception("Check failed; will retry next interval.")

            # Re-read config.yaml each cycle so edits (interests, threshold,
            # interval, channels) apply without restarting -- handy on a
            # phone where you might tweak config.yaml with a text editor
            # app while this keeps running.
            try:
                app_config = load_app_config()
            except ConfigError:
                logger.warning("config.yaml became invalid; keeping previous settings.")

            await asyncio.sleep(max(60, app_config.check_interval_minutes * 60))


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram channel digest (Termux/Android)")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max messages to scan back per channel per run (default: 200)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single check and exit (for cron / termux-job-scheduler)",
    )
    args = parser.parse_args()

    if is_termux():
        logger.info("Termux environment detected (termux-notification found).")
    else:
        logger.info(
            "termux-notification not found; notifications will fall back to "
            "plyer or be skipped. This is fine for testing outside Termux."
        )

    try:
        if args.once:
            scored_posts = asyncio.run(run_once(limit_per_channel=args.limit))
            print_digest(scored_posts, app_config)
            notify_posts(scored_posts)
        else:
            asyncio.run(run_loop(limit_per_channel=args.limit))
    except ConfigError as e:
        logger.error(str(e))
        raise SystemExit(1)
    except KeyboardInterrupt:
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
