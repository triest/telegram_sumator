"""Background worker that runs the fetch+filter pipeline on a schedule.

Runs in its own thread with its own asyncio event loop, so it doesn't
block the CustomTkinter GUI's mainloop (Telethon needs asyncio; Tkinter
needs its own loop in the main thread -- these can't share one loop).

Cross-thread communication is one-way and queue-based: this worker only
ever pushes WorkerEvent objects onto a thread-safe `queue.Queue`; the GUI
polls that queue periodically via `Tk.after(...)`. The worker never calls
into Tkinter directly, which is the only safe way to mix threads with Tk.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from config import ConfigError, load_app_config, load_credentials, resolve_path
from embed_filter import build_filter
from pipeline import ScoredPost, check_once
from storage import Storage
from telegram_client import TelegramDigestClient

logger = logging.getLogger(__name__)

WorkerEventType = Literal["status", "posts", "error", "ask"]


@dataclass
class WorkerEvent:
    type: WorkerEventType
    status: Optional[str] = None
    posts: list[ScoredPost] = field(default_factory=list)
    error: Optional[str] = None
    # For type == "ask": text shown in the input dialog, and whether to mask.
    prompt: Optional[str] = None
    secret: bool = False


class DigestWorker(threading.Thread):
    """Background thread: login once, then poll on a schedule + on demand."""

    def __init__(self, limit_per_channel: int = 200) -> None:
        super().__init__(name="DigestWorker", daemon=True)
        self.events: "queue.Queue[WorkerEvent]" = queue.Queue()
        self._limit_per_channel = limit_per_channel

        self._stop_event = threading.Event()
        self._check_now_event = threading.Event()
        self._reload_event = threading.Event()
        self._answers: "queue.Queue[Optional[str]]" = queue.Queue()
        self._code_attempts = 0

    # -- public control API, safe to call from the GUI (main) thread -----
    def stop(self) -> None:
        self._stop_event.set()

    def check_now(self) -> None:
        self._check_now_event.set()

    def answer(self, value: Optional[str]) -> None:
        """GUI -> worker: reply to an "ask" event (None = user cancelled)."""
        self._answers.put(value)

    def request_reload(self) -> None:
        """Call after saving config.yaml to pick up new settings (interest
        examples, threshold, interval, channels, device) without
        restarting the whole app / re-logging into Telegram."""
        self._reload_event.set()

    # -- interactive login (blocks only this worker thread) -----------------
    def _ask(self, prompt: str, secret: bool = False) -> str:
        while not self._answers.empty():  # drop stale answers
            self._answers.get_nowait()
        logger.info("Жду ввод от пользователя: %s", prompt.splitlines()[0])
        self.events.put(WorkerEvent(type="ask", prompt=prompt, secret=secret))
        while not self._stop_event.is_set():
            try:
                value = self._answers.get(timeout=0.5)
            except queue.Empty:
                continue
            if value is None:
                raise RuntimeError("Вход в Telegram отменён пользователем")
            return value.strip()
        raise RuntimeError("Остановлено")

    def _phone_cb(self) -> str:
        return self._ask("Введите номер телефона Telegram\n(в формате +79991234567):")

    def _code_cb(self) -> str:
        self._code_attempts += 1
        if self._code_attempts == 1:
            msg = "Telegram отправил код подтверждения.\nОн пришёл в приложение Telegram (чат «Telegram»).\n\nВведите код:"
        else:
            msg = "Код неверный или устарел.\nВведите код ещё раз:"
        return self._ask(msg)

    def _password_cb(self) -> str:
        return self._ask(
            "Включена двухэтапная проверка.\nВведите облачный пароль Telegram:",
            secret=True,
        )

    # -- thread body -------------------------------------------------------
    def run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as e:  # pragma: no cover - top-level safety net
            logger.exception("Worker crashed")
            self.events.put(WorkerEvent(type="error", error=str(e)))

    async def _main(self) -> None:
        self._emit_status("Запуск...")
        try:
            creds = load_credentials()
            app_config = load_app_config()
        except ConfigError as e:
            self.events.put(WorkerEvent(type="error", error=str(e)))
            return

        db_path = resolve_path(app_config.db_path)
        storage = Storage(db_path)

        # Load the local model and log in to Telegram AT THE SAME TIME
        # (the model takes ~10-20 s; login/connection is mostly waiting).
        self._emit_status("Загрузка модели и подключение к Telegram...")
        examples = app_config.all_examples()
        model_task = asyncio.create_task(
            asyncio.to_thread(
                build_filter,
                examples,
                threshold=app_config.similarity_threshold,
                device=app_config.embedding_device,
                topics=app_config.topics,
                llm_model=app_config.llm_model if app_config.llm_classify else "",
                llm_url=app_config.llm_url,
            )
        )
        tg = TelegramDigestClient(creds)
        await tg.start(
            phone_callback=self._phone_cb,
            code_callback=self._code_cb,
            password_callback=self._password_cb,
        )
        self._emit_status("Ожидание загрузки модели...")
        # The model import can be slow on a cold start (big PyTorch libraries
        # on a slow disk, antivirus, a busy GPU). Wait patiently, show progress.
        waited = 0
        while not model_task.done() and waited < 1200:
            await asyncio.wait({model_task}, timeout=5)
            waited += 5
            if waited % 15 == 0 and not model_task.done():
                self._emit_status(f"Загрузка модели… {waited} с (первый запуск после перезагрузки бывает долгим)")
        if model_task.done() and not model_task.cancelled() and model_task.exception() is None:
            relevance_filter = model_task.result()
        else:
            logger.error(
                "Модель не загрузилась (%s) — работаю по ключевым словам тем.",
                model_task.exception() if model_task.done() else "таймаут 20 минут",
            )
            self._emit_status("Модель не загрузилась, работаю по ключевым словам")
            from embed_filter import KeywordFilter, TopicKeywordBoost

            relevance_filter = TopicKeywordBoost(
                KeywordFilter(examples), app_config.topics, app_config.similarity_threshold
            )
        logger.info("Анализ постов: %s", getattr(relevance_filter, "describe", lambda: "?")())
        logger.info("Вход в Telegram выполнен")
        client = tg
        # Fill the channel list for Settings right away (all channels,
        # archived included), without waiting for the first scan.
        try:
            self._emit_status("Загрузка списка каналов...")
            all_channels = await tg.discover_channels()
            if app_config.only_broadcast:
                all_channels = [c for c in all_channels if c.broadcast]
            for ch in all_channels:
                storage.upsert_channel(ch.channel_id, ch.username, ch.title, watch=True)
            archived = sum(1 for c in all_channels if c.archived)
            logger.info("Каналов всего: %d (в архиве: %d)", len(all_channels), archived)
            self._emit_status(f"Найдено каналов: {len(all_channels)} (в архиве {archived})")
        except Exception as e:
            logger.warning("Не удалось загрузить список каналов: %s", e)
        try:
            self._emit_status("Готово")
            last_check = 0.0

            while not self._stop_event.is_set():
                interval_seconds = app_config.check_interval_minutes * 60
                due = (time.monotonic() - last_check) >= interval_seconds
                if due or self._check_now_event.is_set():
                    self._check_now_event.clear()
                    self._emit_status("Проверка новых постов...")
                    try:
                        scored = await check_once(
                            client=client,
                            storage=storage,
                            app_config=app_config,
                            relevance_filter=relevance_filter,
                            limit_per_channel=self._limit_per_channel,
                            progress=lambda d, t: self._emit_status(
                                f"Проверка каналов: {d}/{t}"
                            ),
                        )
                        last_check = time.monotonic()
                        if scored:
                            self.events.put(WorkerEvent(type="posts", posts=scored))
                        self._emit_status("Ожидание")
                    except Exception as e:
                        logger.exception("Check failed")
                        self.events.put(WorkerEvent(type="error", error=str(e)))
                        self._emit_status("Ошибка, повтор позже")
                        last_check = time.monotonic()

                if self._reload_event.is_set():
                    self._reload_event.clear()
                    try:
                        app_config = load_app_config()
                        self._emit_status("Перезагрузка фильтра...")
                        relevance_filter = build_filter(
                            app_config.all_examples(),
                            threshold=app_config.similarity_threshold,
                            device=app_config.embedding_device,
                            topics=app_config.topics,
                            llm_model=app_config.llm_model if app_config.llm_classify else "",
                            llm_url=app_config.llm_url,
                        )
                        self._emit_status("Готово")
                        self._check_now_event.set()  # rescan with the new topics
                    except Exception as e:
                        logger.exception("Reload failed")
                        self.events.put(WorkerEvent(type="error", error=str(e)))

                await asyncio.sleep(1.0)

        finally:
            await tg.stop()

        storage.close()
        self._emit_status("Остановлено")

    def _emit_status(self, status: str) -> None:
        logger.info("Статус: %s", status)
        self.events.put(WorkerEvent(type="status", status=status))
