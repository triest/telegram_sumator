"""Shared core pipeline: fetch new posts -> local filter -> scored results.

This is the one place that ties telegram_client + embed_filter + storage
together, so the CLI (main.py) and the future GUI (gui/) run the exact
same logic instead of drifting apart.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from config import AppConfig, resolve_path
from embed_filter import FilterResult, contains_excluded_keyword
from storage import Storage
from telegram_client import Post, TelegramDigestClient

logger = logging.getLogger(__name__)


@dataclass
class ScoredPost:
    post: Post
    result: FilterResult


async def check_once(
    client: TelegramDigestClient,
    storage: Storage,
    app_config: AppConfig,
    relevance_filter,
    limit_per_channel: int = 200,
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[ScoredPost]:
    """Run one fetch+filter pass over all watched channels.

    `client` must already be started (logged in). `relevance_filter` is
    whatever `embed_filter.build_filter()` returned (EmbeddingFilter or
    KeywordFilter -- duck-typed, both expose `.score_batch(texts)`).

    Speed: channels are fetched concurrently (max_concurrent), the first
    scan of a channel is capped (first_run_limit) instead of pulling a huge
    backfill, and each channel's posts are scored in ONE batch on a worker
    thread instead of one model call per post.

    Returns matching posts sorted by score, descending. Advances
    `last_seen` in storage for every channel scanned, and records shown
    posts, as a side effect.
    """
    # Channels the user switched OFF (by id or username). Everything else is
    # watched, so newly joined channels are picked up automatically.
    off_ids = {c.channel_id for c in app_config.channels if not c.watch and c.channel_id is not None}
    off_names = {c.username for c in app_config.channels if not c.watch and c.username}

    channels = await client.discover_channels()
    logger.info("Discovered %d channels/supergroups.", len(channels))

    todo = []
    for ch in channels:
        # Keep the full channel list in storage so Settings can show every
        # channel (not just those with unread posts right now).
        if app_config.only_broadcast and not ch.broadcast:
            continue
        storage.upsert_channel(ch.channel_id, ch.username, ch.title, watch=True)
        if ch.channel_id in off_ids or (ch.username and ch.username in off_names):
            continue
        if app_config.only_unread and ch.unread_count <= 0:
            continue  # nothing unread here
        todo.append(ch)
    logger.info(
        "Scanning %d channels%s (concurrency %d).",
        len(todo),
        " with unread posts" if app_config.only_unread else "",
        app_config.max_concurrent,
    )

    # Channel context: what each channel usually posts about (mean similarity
    # of its recent posts + its title to every topic). Cached in the DB and
    # refreshed weekly or when topics change. Used to nudge ambiguous posts
    # toward the channel's usual topics and as a hint for the LLM.
    ctx_weight = app_config.channel_context_weight
    use_ctx = ctx_weight > 0 and hasattr(relevance_filter, "topic_matrix")
    sig = hashlib.md5("|".join(app_config.all_examples()).encode("utf-8")).hexdigest()
    PROFILE_TTL = 7 * 86400
    PROFILE_SAMPLE = 25

    thumbs_dir = resolve_path("cache/thumbs")
    img_sem = asyncio.Semaphore(4)
    sem = asyncio.Semaphore(app_config.max_concurrent)
    score_lock = asyncio.Lock()  # one model call at a time on the GPU
    scored_posts: list[ScoredPost] = []
    other_posts: list[ScoredPost] = []  # judged not relevant -> shown under 'Прочее'
    done = 0
    total = len(todo)
    stats = {"posts": 0, "excluded": 0, "best_miss": 0.0}

    async def get_prior(ch):
        if not use_ctx:
            return None
        try:
            cached = storage.get_profile(ch.channel_id)
            now = time.time()
            if cached and cached[0] == sig and now - cached[2] < PROFILE_TTL:
                vec = cached[1]
            else:
                async with sem:
                    sample = [
                        p
                        async for p in client.fetch_new_posts(
                            channel_id=ch.channel_id,
                            channel_title=ch.title,
                            channel_username=ch.username,
                            last_seen_id=0,
                            limit=PROFILE_SAMPLE,
                            entity=ch.entity,
                        )
                    ]
                texts = [p.text for p in sample if p.text and p.text.strip()]
                texts.append(ch.title or "")
                if len(texts) < 4:
                    return None  # too little text to say what the channel is about
                async with score_lock:
                    matrix = await asyncio.to_thread(relevance_filter.topic_matrix, texts)
                if matrix is None or len(matrix) == 0:
                    return None
                vec = matrix.mean(axis=0).tolist()
                storage.set_profile(ch.channel_id, sig, vec, now)
            v = np.asarray(vec, dtype=float)
            prior = np.clip((v - v.mean()) * ctx_weight, -0.15, 0.2)
            return prior
        except Exception as e:
            logger.debug("channel profile failed for %s: %s", ch.title, e)
            return None

    async def process(ch) -> None:
        nonlocal done
        try:
            async with sem:
                storage.upsert_channel(ch.channel_id, ch.username, ch.title, watch=True)
                stored_last_seen = storage.get_last_seen(ch.channel_id)
                last_seen_id = storage.get_last_seen(ch.channel_id)
                if app_config.only_unread:
                    # Unread mode: the source of truth is Telegram's own
                    # read marker, NOT our stored last_seen. Every check
                    # re-evaluates all currently-unread posts, so changing
                    # topics/threshold immediately picks up posts that an
                    # earlier pass judged irrelevant. (The GUI de-duplicates.)
                    last_seen_id = ch.read_max_id
                    limit = min(max(ch.unread_count, 1), limit_per_channel)
                else:
                    limit = app_config.first_run_limit if last_seen_id == 0 else limit_per_channel
                posts = [
                    p
                    async for p in client.fetch_new_posts(
                        channel_id=ch.channel_id,
                        channel_title=ch.title,
                        channel_username=ch.username,
                        last_seen_id=last_seen_id,
                        limit=limit,
                        entity=ch.entity,
                    )
                ]
            if posts:
                max_seen = max(p.message_id for p in posts)
                candidates = [
                    p
                    for p in posts
                    if not contains_excluded_keyword(p.text, app_config.global_exclude_keywords)
                ]
                stats["posts"] += len(posts)
                stats["excluded"] += len(posts) - len(candidates)
                if candidates:
                    prior = await get_prior(ch)
                    async with score_lock:
                        results = await asyncio.to_thread(
                            lambda: relevance_filter.score_batch(
                                [p.text for p in candidates],
                                prior=prior,
                                channel_title=ch.title or "",
                            )
                        )
                    fresh = []
                    for post, result in zip(candidates, results):
                        if not result.is_relevant:
                            stats["best_miss"] = max(stats["best_miss"], result.score)
                            other_posts.append(ScoredPost(post=post, result=result))
                        if result.is_relevant:
                            sp = ScoredPost(post=post, result=result)
                            scored_posts.append(sp)
                            fresh.append(sp)
                            if not app_config.only_unread:
                                storage.mark_shown(ch.channel_id, post.message_id, result.score)
                    if app_config.show_images and fresh:
                        async with img_sem:
                            paths = await asyncio.gather(
                                *(client.download_thumb(sp.post, thumbs_dir) for sp in fresh)
                            )
                        for sp, path in zip(fresh, paths):
                            sp.post.image_path = path
                if max_seen > stored_last_seen and not app_config.only_unread:
                    storage.set_last_seen(ch.channel_id, max_seen)
        except Exception as e:  # one bad channel must not abort the whole pass
            logger.warning("Канал «%s» пропущен: %s", ch.title, e)
        finally:
            done += 1
            if progress and (done % 10 == 0 or done == total):
                progress(done, total)

    await asyncio.gather(*(process(ch) for ch in todo))

    scored_posts.sort(key=lambda sp: sp.result.score, reverse=True)
    n_relevant = len(scored_posts)
    if getattr(app_config, "show_unmatched", True) and other_posts:
        other_posts.sort(key=lambda sp: sp.post.date, reverse=True)
        scored_posts.extend(other_posts[:200])
    if hasattr(relevance_filter, "pop_stats"):
        ls = relevance_filter.pop_stats()
        logger.info(
            "ИИ-проверка (LLM): запросов к модели %d, из кэша %d; подтверждено %d, отклонено %d",
            ls["calls"], ls["cached"], ls["accepted"], ls["rejected"],
        )
    logger.info(
        "Непрочитанных постов проверено: %d (исключено стоп-словами: %d), подошло: %d. "
        "Лучший балл среди НЕ подошедших: %.2f (порог %.2f)",
        stats["posts"],
        stats["excluded"],
        n_relevant,
        stats["best_miss"],
        app_config.similarity_threshold,
    )
    return scored_posts
