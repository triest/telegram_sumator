"""CLI MVP: login -> fetch new posts -> local embed filter -> print sorted by score.

Delivery Order step 1-2 (see agents.md): this is the first runnable slice.
No cloud LLM is used anywhere in this pipeline. Core logic lives in
pipeline.py, shared with the GUI (gui/).
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from config import ConfigError, load_app_config, load_credentials, resolve_path
from embed_filter import build_filter
from pipeline import ScoredPost, check_once
from storage import Storage
from telegram_client import TelegramDigestClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("main")


async def run(limit_per_channel: int) -> None:
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
    print_digest(scored_posts, app_config)


def print_digest(scored_posts: list[ScoredPost], app_config) -> None:
    if not scored_posts:
        print("No new relevant posts found.")
        return

    print(f"\n=== {len(scored_posts)} relevant post(s) ===\n")

    by_topic: dict[str, list[ScoredPost]] = {}
    for sp in scored_posts:
        topic = app_config.topic_name_for(sp.result.matched_example) or "(no topic)"
        by_topic.setdefault(topic, []).append(sp)

    for topic, posts in by_topic.items():
        print(f"--- {topic} ({len(posts)}) ---")
        for sp in posts:
            post = sp.post
            print(f"[{sp.result.score:.2f}] {post.channel_title} - {post.date:%Y-%m-%d %H:%M}")
            print(f"  {post.preview}")
            print(f"  {post.link}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram channel digest (local model)")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max messages to scan back per channel per run (default: 200)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run(limit_per_channel=args.limit))
    except ConfigError as e:
        logger.error(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
