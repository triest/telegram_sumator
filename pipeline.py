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


# thresholds for a channel's "home" topic (similarity above the neutral baseline)
HOME_MIN_TOP = 0.09
HOME_MIN_GAP = 0.035

NEUTRAL_PROBES = [
    "Сегодня ходил в магазин, купил хлеб и молоко, потом гулял по парку",
    "Скидка 30% на всё до конца недели, промокод в описании, успейте оформить заказ",
    "Рецепт простого супа: нарезать овощи, залить бульоном и варить двадцать минут",
    "Курс валют на сегодня, нефть подешевела, аналитики ждут решения регулятора",
    "Мы переехали в новую квартиру, делаю ремонт, показываю как получилось",
    "Прогноз погоды на выходные: облачно, местами дожди, температура около десяти градусов",
    "Матч закончился со счётом два один, тренер доволен игрой команды",
    "Как справиться со стрессом на работе: пять советов психолога",
    "Новый смартфон получил улучшенную камеру и батарею на два дня работы",
    "Всем привет! Спасибо что вы со мной, завтра будет новый пост, оставайтесь на связи",
    "Доброе утро! Хорошего дня и отличного настроения всем подписчикам",
    "Мэр объявил о ремонте дорог и благоустройстве двора в этом районе",
    "Розыгрыш призов среди подписчиков, условия в закреплённом сообщении",
    "Приходите на встречу в субботу, начало в шесть вечера, вход свободный",
    "Мой кот сегодня опять устроил беспорядок на кухне, смотрите видео",
    "Выставка современного искусства открывается в городском музее на следующей неделе",
    "Сколько стоит содержать автомобиль: страховка, топливо, обслуживание",
    "Вакансия: требуется менеджер по продажам, график пять на два, оформление по ТК",
    "Рассказываю о поездке на выходные: дорога, гостиница, что понравилось",
    "Полезный совет: как выбрать хороший матрас и не переплатить",
]


async def check_once(
    client: TelegramDigestClient,
    storage: Storage,
    app_config: AppConfig,
    relevance_filter,
    limit_per_channel: int = 200,
    progress: Optional[Callable[[int, int], None]] = None,
    unplaced_out: Optional[list[str]] = None,
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
    # (the model is part of the signature: profiles made by another model are not comparable)
    sig = hashlib.md5(
        ("|".join(app_config.all_examples()) + "#" + app_config.embedding_model).encode("utf-8")
    ).hexdigest()
    # Some topics (short generic names like "кино") are similar to almost any text. Their
    # typical similarity to neutral posts is subtracted, otherwise such a topic becomes
    # the "home" of every channel that has no clear subject.
    baseline = None
    if use_ctx:
        try:
            probe = await asyncio.to_thread(relevance_filter.topic_matrix, NEUTRAL_PROBES)
            if probe is not None and len(probe):
                baseline = np.asarray(probe, dtype=float).mean(axis=0)
        except Exception as e:
            logger.debug("topic baseline failed: %s", e)
    PROFILE_TTL = 7 * 86400
    PROFILE_SAMPLE = 25

    topic_names = [t.name for t in app_config.topics]
    example_by_name = {t.name: t.as_example() for t in app_config.topics}
    force = bool(getattr(app_config, "force_category", True)) and bool(topic_names)
    home_auto: dict[int, str] = {}  # channel_id -> category inferred from the channel's profile

    def channel_topic_of(ch) -> str:
        """The channel's own category: the user's choice, else the inferred one."""
        stored = storage.get_channel_category(ch.channel_id)
        if stored and stored[1] == "manual" and stored[0] in example_by_name:
            return stored[0]
        if ch.channel_id in home_auto:
            name = home_auto[ch.channel_id]
        else:
            name = stored[0] if stored else ""
        return name if name in example_by_name else ""

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
            if baseline is not None and len(baseline) == len(v):
                v = v - baseline
            if topic_names and len(v) >= len(topic_names):
                tv = v[: len(topic_names)]
                order = np.argsort(-tv)
                home = topic_names[int(order[0])]
                if baseline is not None and len(order) > 1:
                    # a channel gets a "home" topic only when one topic is clearly ahead;
                    # otherwise (a personal blog, a mix) its posts are not pushed into one
                    if tv[order[0]] < HOME_MIN_TOP or tv[order[0]] - tv[order[1]] < HOME_MIN_GAP:
                        home = ""
                home_auto[ch.channel_id] = home
                storage.set_channel_category(ch.channel_id, home, "auto")
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
                    # picture-only / caption-only posts: fetch the picture first,
                    # so the vision model can look at it
                    need = [
                        c for c in candidates
                        if len(c.text.strip()) < 40 and (c.has_photo or c.has_video)
                    ]
                    if need and (getattr(relevance_filter, "vision_model", "") or app_config.show_images):
                        async with img_sem:
                            got = await asyncio.gather(
                                *(client.download_thumb(c, thumbs_dir) for c in need)
                            )
                        for c, path in zip(need, got):
                            c.image_path = path
                    home_topic = channel_topic_of(ch)
                    async with score_lock:
                        results = await asyncio.to_thread(
                            lambda: relevance_filter.score_batch(
                                [p.text for p in candidates],
                                prior=prior,
                                channel_title=ch.title or "",
                                channel_topic=home_topic,
                                images=[c.image_path for c in candidates],
                            )
                        )
                    fresh = []
                    if unplaced_out is not None:
                        # posts no topic fit (before the "every post gets a category" fallback)
                        unplaced_out.extend(
                            p.text for p, r in zip(candidates, results)
                            if not r.is_relevant and r.source != "вручную" and p.text.strip()
                        )
                    if force:
                        # Every channel fits one of the topics: whatever no engine could
                        # place goes to the channel's own topic (or the closest one).
                        fixed = []
                        for r in results:
                            if r.is_relevant or r.source == "вручную":
                                fixed.append(r)
                                continue
                            # only a channel with a clear home topic forces its posts there;
                            # "closest topic" is NOT used (a generic topic attracts everything)
                            ex = example_by_name.get(home_topic)
                            if ex:
                                fixed.append(FilterResult(True, max(r.score, 0.35), ex, "канал"))
                                stats["forced"] = stats.get("forced", 0) + 1
                            else:
                                fixed.append(r)
                        results = fixed
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
        scored_posts.extend(other_posts[:100])
    if hasattr(relevance_filter, "pop_stats"):
        ls = relevance_filter.pop_stats()
        logger.info(
            "ИИ-проверка (LLM): запросов к модели %d, из кэша %d; подтверждено %d, отклонено %d",
            ls["calls"], ls["cached"], ls["accepted"], ls["rejected"],
        )
    logger.info(
        "Отнесено к категории канала (без явного совпадения): %d", stats.get("forced", 0)
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
