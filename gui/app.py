"""CustomTkinter main window: feed of matching posts + settings editor.

Threading model:
    - Tkinter mainloop runs on the main thread (required by Tk).
    - DigestWorker (gui/worker.py) runs the Telethon/asyncio pipeline on
      its own thread and only ever *pushes* WorkerEvent objects onto a
      thread-safe queue.
    - This window polls that queue every POLL_INTERVAL_MS via `after()`
      and updates widgets from the main thread only -- the one safe way
      to mix a worker thread with Tk.
"""
from __future__ import annotations

import logging
import os
import queue
import re
import sys
import webbrowser
import zlib
from logging.handlers import RotatingFileHandler
from typing import Optional

import customtkinter as ctk

from config import (
    AppConfig,
    ChannelConfig,
    load_app_config,
    TopicConfig,
    save_app_config,
)
from gui.notifications import notify_new_posts
from gui.tray import TrayIcon
from gui.worker import DigestWorker, WorkerEvent
from pipeline import ScoredPost
from storage import Storage

logger = logging.getLogger(__name__)

from config import PROJECT_ROOT  # noqa: E402

LOG_PATH = PROJECT_ROOT / "digest.log"

POLL_INTERVAL_MS = 300
MAX_FEED_ITEMS = 400
ALL_TOPICS = "Все темы"
OTHER_TOPIC = "Прочее (не подошло)"


_ACTIVE_CONFIG: Optional[AppConfig] = None


def set_active_config(config: AppConfig) -> None:
    global _ACTIVE_CONFIG
    _ACTIVE_CONFIG = config


def topic_label(scored: ScoredPost) -> str:
    """The 'topic' a post falls under: the interest example it matched
    best. Falls back to a generic label if the filter didn't record one
    (e.g. the keyword fallback filter with no match above the threshold
    shouldn't reach here, but be defensive)."""
    if not scored.result.is_relevant:
        return OTHER_TOPIC
    example = scored.result.matched_example
    if _ACTIVE_CONFIG is not None:
        example = _ACTIVE_CONFIG.topic_name_for(example)
    return example or "Без темы"


def _short(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


TOPIC_COLORS = [
    ("#dbeafe", "#1e3a5f"),
    ("#fce7f3", "#5b2142"),
    ("#dcfce7", "#1d4a30"),
    ("#fef3c7", "#5a4310"),
    ("#ede9fe", "#3b2a6b"),
    ("#ffedd5", "#5a3213"),
    ("#cffafe", "#134e5a"),
    ("#fee2e2", "#5a1f1f"),
]
TEXT_ON_CHIP = ("#1f2937", "#e8ecf4")


def topic_color(topic: str) -> tuple[str, str]:
    """Stable colour per topic name (same topic -> same colour every run)."""
    if topic == OTHER_TOPIC:
        return ("#e5e7eb", "#374151")
    return TOPIC_COLORS[zlib.crc32(topic.encode("utf-8")) % len(TOPIC_COLORS)]


def _chip(master, text: str, fg, text_color=TEXT_ON_CHIP, size: int = 12):
    return ctk.CTkLabel(
        master,
        text=text,
        fg_color=fg,
        text_color=text_color,
        corner_radius=10,
        font=ctk.CTkFont(size=size),
        padx=10,
        pady=2,
        height=24,
    )


class PostCard(ctk.CTkFrame):
    """Compact digest card.

    Collapsed (default): small square thumbnail + channel/time + the first
    lines of the post. Click anywhere on the card to expand: full text with
    clickable links and a larger picture; click the picture to see it
    full-size. Short text-only posts have nothing to expand."""

    SNIPPET = 230
    THUMB = 104

    def __init__(
        self, master, scored: ScoredPost, on_move=None, topic_names=None, on_channel=None,
        on_confirm=None, on_reject=None,
    ) -> None:
        super().__init__(
            master,
            corner_radius=12,
            fg_color=("white", "#1c2230"),
            border_width=1,
            border_color=("#e3e7ee", "#2c3446"),
        )
        self.scored = scored
        self.topic = topic_label(scored)
        self._post = scored.post
        self._on_move = on_move
        self._on_channel = on_channel
        self._on_confirm = on_confirm
        self._on_reject = on_reject
        self._topic_names = list(topic_names or [])
        self._text_full = self._post.text.strip() or "(без текста, только картинка)"
        self._has_img = bool(
            getattr(self._post, "image_path", None) and os.path.exists(self._post.image_path)
        )
        self._expandable = self._has_img or len(self._text_full) > self.SNIPPET or self._text_full.count("\n") > 3
        self._expanded = False
        self._link_click = False
        self._refs = []  # keep CTkImage objects alive
        self._wrap = 560
        self._body_label = None
        self._built = False
        self._title, self._rest = self._split_title(self._text_full)
        self.grid_columnconfigure(1, weight=1)
        self.bind("<Configure>", self._on_resize)

    # Cards are built lazily (first time they are shown): hundreds of cards
    # with ~10 widgets each would make the window sluggish and glitchy.
    def ensure_built(self) -> None:
        if not self._built:
            self._built = True
            self._render()

    def pack(self, *a, **kw):
        self.ensure_built()
        super().pack(*a, **kw)

    @classmethod
    def _split_title(cls, text: str):
        """(title, rest): the first line is a title when it is short and the
        post has more text after it (typically a **bold** headline)."""
        lines = [ln for ln in text.strip().splitlines()]
        if not lines:
            return "", text
        first = cls._plain(lines[0])
        rest = "\n".join(lines[1:]).strip()
        if first and rest and len(first) <= 140:
            return first, rest
        return "", text

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _plain(text: str) -> str:
        t = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"\1", text)
        t = t.replace("**", "").replace("__", "").replace("`", "")
        return " ".join(t.split())

    def _snippet(self) -> str:
        t = self._plain(self._rest if self._title else self._text_full)
        return t if len(t) <= self.SNIPPET else t[: self.SNIPPET].rstrip() + "…"

    def _clear(self) -> None:
        for w in self.winfo_children():
            w.destroy()
        self._refs = []
        self._body_label = None

    def _toggle(self, _e=None) -> None:
        if not self._expandable:
            return
        if self._link_click:
            self._link_click = False
            return
        self._expanded = not self._expanded
        self._render()

    def _open_link(self, url: str) -> None:
        self._link_click = True
        webbrowser.open(url)

    def _bind_click(self, widget) -> None:
        widget.bind("<Button-1>", self._toggle)
        if self._expandable:
            try:
                widget.configure(cursor="hand2")
            except Exception:
                pass

    def _thumb(self, size: int):
        from PIL import Image, ImageOps

        img = Image.open(self._post.image_path).convert("RGB")
        img = ImageOps.fit(img, (size, size), Image.LANCZOS)
        ci = ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
        self._refs.append(ci)
        return ci

    def _big(self, max_w: int = 640, max_h: int = 420):
        from PIL import Image

        img = Image.open(self._post.image_path).convert("RGB")
        w, h = img.size
        scale = min(max_w / w, max_h / h, 1.0)
        size = (max(1, int(w * scale)), max(1, int(h * scale)))
        ci = ctk.CTkImage(light_image=img, dark_image=img, size=size)
        self._refs.append(ci)
        return ci

    def _show_full_image(self, _e=None) -> None:
        try:
            from PIL import Image

            top = ctk.CTkToplevel(self)
            top.title(self._post.channel_title or "Картинка")
            sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
            img = Image.open(self._post.image_path).convert("RGB")
            w, h = img.size
            scale = min((sw * 0.9) / w, (sh * 0.85) / h)
            size = (max(1, int(w * scale)), max(1, int(h * scale)))
            ci = ctk.CTkImage(light_image=img, dark_image=img, size=size)
            lab = ctk.CTkLabel(top, image=ci, text="", cursor="hand2")
            lab.image = ci
            lab.pack(padx=6, pady=6)
            top.geometry(f"{size[0] + 12}x{size[1] + 12}+{(sw - size[0]) // 2}+{max(0, (sh - size[1]) // 2 - 20)}")
            for w_ in (top, lab):
                w_.bind("<Button-1>", lambda _e: top.destroy())
            top.bind("<Escape>", lambda _e: top.destroy())
            top.after(150, top.focus_force)
        except Exception:
            logger.debug("Не удалось показать картинку", exc_info=True)

    # ------------------------------------------------------------ render
    def _render(self) -> None:
        self._clear()
        post, scored = self._post, self.scored
        col_body = 1 if (self._has_img and not self._expanded) else 0
        span = 1 if col_body else 2
        pad_l = 10

        if self._has_img and not self._expanded:
            try:
                tl = ctk.CTkLabel(self, image=self._thumb(self.THUMB), text="")
                tl.grid(row=0, column=0, rowspan=5, padx=(10, 0), pady=10, sticky="n")
                self._bind_click(tl)
            except Exception:
                self._has_img = False
                col_body, span = 0, 2

        # header: CHANNEL (always) . time . score
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=col_body, columnspan=span, sticky="ew", padx=(12, 12), pady=(10, 0))
        head.grid_columnconfigure(0, weight=1)
        chan = post.channel_title or (f"@{post.channel_username}" if post.channel_username else "Канал")
        title = ctk.CTkLabel(
            head,
            text=f"{chan}   ·   {post.date:%d.%m %H:%M}",
            anchor="w",
            text_color=("#1a56c4", "#8ab4ff"),
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        title.grid(row=0, column=0, sticky="w")
        self._bind_click(title)
        _chip(
            head, f"{scored.result.score:.2f}", fg=self._score_color(scored.result.score),
            text_color="white", size=11,
        ).grid(row=0, column=1, sticky="e")
        self._bind_click(head)
        if self._title:
            tl2 = ctk.CTkLabel(
                self, text=self._title, anchor="w", justify="left", wraplength=self._wrap,
                font=ctk.CTkFont(size=15, weight="bold"),
            )
            tl2.grid(row=1, column=col_body, columnspan=span, sticky="ew", padx=12, pady=(2, 0))
            self._title_label = tl2
            self._bind_click(tl2)
        else:
            self._title_label = None

        if not self._expanded:
            lab = ctk.CTkLabel(
                self, text=self._snippet(), anchor="nw", justify="left",
                wraplength=self._wrap, font=ctk.CTkFont(size=13),
            )
            lab.grid(row=2, column=col_body, columnspan=span, sticky="ew", padx=12, pady=(2, 4))
            self._body_label = lab
            self._bind_click(lab)
        else:
            if self._has_img:
                try:
                    il = ctk.CTkLabel(self, image=self._big(), text="", cursor="hand2")
                    il.grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(2, 6))
                    il.bind("<Button-1>", self._show_full_image)
                except Exception:
                    pass
            tx = self._make_text(self, self._text_full if not self._title else self._rest)
            tx.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))
            self._text = tx

        # footer: chips + buttons
        foot = ctk.CTkFrame(self, fg_color="transparent")
        foot.grid(row=4, column=col_body, columnspan=span, sticky="ew", padx=12, pady=(2, 10))
        foot.grid_columnconfigure(10, weight=1)
        c = 0
        _chip(foot, _short(self.topic, 28), fg=topic_color(self.topic), size=11).grid(row=0, column=c)
        c += 1
        src = {
            "ИИ": "ИИ", "ИИ-фото": "ИИ по картинке", "слова": "слова",
            "смысл": "смысл", "канал": "по теме канала", "вручную": "вручную",
            "картинка": "по картинке",
        }.get(scored.result.source, "")
        if src:
            _chip(foot, src, fg=("#eef1f6", "#2a3244"), text_color=("#4b5563", "#9aa5b8"), size=11).grid(
                row=0, column=c, padx=(6, 0)
            )
            c += 1
        if post.has_video:
            _chip(foot, "видео", fg=("#eef1f6", "#2a3244"), text_color=("#4b5563", "#9aa5b8"), size=11).grid(
                row=0, column=c, padx=(6, 0)
            )
            c += 1
        # one-click verdict on the classification: both feed the local filter's examples
        if (
            self._on_confirm and self._on_reject
            and scored.result.source != "вручную"
            and self.topic in self._topic_names
        ):
            ctk.CTkButton(
                foot, text="👍 Верно", width=74, height=24, corner_radius=8,
                fg_color=("#e6f4ea", "#1f3a2a"), hover_color=("#cfe9d7", "#2a4d38"),
                text_color=("#1e7a3c", "#7fd49a"), command=lambda: self._on_confirm(self),
            ).grid(row=0, column=c, padx=(8, 0))
            c += 1
            ctk.CTkButton(
                foot, text="👎 Не то", width=74, height=24, corner_radius=8,
                fg_color=("#fdeaea", "#3d2326"), hover_color=("#f8d3d3", "#552e32"),
                text_color=("#b3261e", "#f2a5a0"), command=lambda: self._on_reject(self),
            ).grid(row=0, column=c, padx=(6, 0))
            c += 1
        if self._expandable:
            ctk.CTkButton(
                foot, text="Свернуть ▴" if self._expanded else "Подробнее ▾", width=90, height=24,
                corner_radius=8, fg_color="transparent", hover_color=("#eef1f6", "#2a3244"),
                text_color=("#1a56c4", "#8ab4ff"), command=self._toggle,
            ).grid(row=0, column=10, sticky="e")
        if self._on_move and self._topic_names:
            choices = [n for n in self._topic_names] + [OTHER_TOPIC]
            menu = ctk.CTkOptionMenu(
                foot, values=choices, width=150, height=24, corner_radius=8,
                dynamic_resizing=False, font=ctk.CTkFont(size=11),
                fg_color=("#eef1f6", "#2a3244"), button_color=("#dbe1ea", "#3a4560"),
                text_color=("#374151", "#c9d3e6"),
                command=lambda choice: self._on_move(self, choice),
            )
            menu.set("Перенести в тему…")
            menu.grid(row=0, column=12, padx=(6, 0), sticky="e")
        if self._on_channel and self._topic_names:
            cmenu = ctk.CTkOptionMenu(
                foot, values=list(self._topic_names), width=150, height=24, corner_radius=8,
                dynamic_resizing=False, font=ctk.CTkFont(size=11),
                fg_color=("#eef1f6", "#2a3244"), button_color=("#dbe1ea", "#3a4560"),
                text_color=("#374151", "#c9d3e6"),
                command=lambda choice: self._on_channel(self, choice),
            )
            cmenu.set("Канал → тема…")
            cmenu.grid(row=0, column=13, padx=(6, 0), sticky="e")
        ctk.CTkButton(
            foot, text="Открыть в Telegram ↗", width=150, height=24, corner_radius=8,
            fg_color=("#e8f0fe", "#243049"), hover_color=("#d6e4fd", "#2f3f60"),
            text_color=("#1a56c4", "#8ab4ff"), command=lambda: webbrowser.open(post.link),
        ).grid(row=0, column=11, padx=(6, 0), sticky="e")
        self._bind_click(foot)

    def _on_resize(self, event) -> None:
        wrap = max(240, event.width - (self.THUMB + 60 if (self._has_img and not self._expanded) else 40))
        if abs(wrap - self._wrap) > 12:
            self._wrap = wrap
            if self._body_label is not None:
                self._body_label.configure(wraplength=wrap)
            if getattr(self, "_title_label", None) is not None:
                self._title_label.configure(wraplength=wrap)

    _MD = re.compile(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)"   # [text](url)
        r"|\*\*(.+?)\*\*"                       # **bold**
        r"|__(.+?)__"                             # __italic__
        r"|`([^`]+)`"                             # `code`
        r"|(https?://[^\s<>\"]+)",                # bare url
        re.S,
    )

    def _make_text(self, parent, text: str):
        import tkinter as tk

        mode = ctk.get_appearance_mode()
        dark = mode == "Dark"
        bg = "#1c2230" if dark else "#ffffff"
        fg = "#e6eaf2" if dark else "#1f2937"
        link = "#6cb6ff" if dark else "#1a56c4"
        w = tk.Text(
            parent, wrap="word", bd=0, highlightthickness=0, relief="flat",
            bg=bg, fg=fg, insertwidth=0, height=1, width=10, padx=0, pady=0,
            font=("Segoe UI", 11), cursor="arrow", spacing1=1, spacing3=3,
        )
        w.tag_config("b", font=("Segoe UI", 11, "bold"))
        w.tag_config("i", font=("Segoe UI", 11, "italic"))
        w.tag_config("code", font=("Consolas", 10), background="#2a3244" if dark else "#eef1f6")
        w.tag_config("link", foreground=link, underline=True)
        pos = 0
        n = 0
        for m in self._MD.finditer(text):
            w.insert("end", text[pos:m.start()])
            pos = m.end()
            if m.group(1):
                label, url, tag = m.group(1), m.group(2), "link"
            elif m.group(3):
                label, url, tag = m.group(3), None, "b"
            elif m.group(4):
                label, url, tag = m.group(4), None, "i"
            elif m.group(5):
                label, url, tag = m.group(5), None, "code"
            else:
                url = m.group(6).rstrip(".,;:!?)\u00bb")
                pos = m.start() + len(url)
                label, tag = url, "link"
            start = w.index("end-1c")
            w.insert("end", label)
            w.tag_add(tag, start, "end-1c")
            if url:
                n += 1
                name = f"u{n}"
                w.tag_add(name, start, "end-1c")
                w.tag_bind(name, "<Button-1>", lambda _e, u=url: self._open_link(u))
                w.tag_bind(name, "<Enter>", lambda _e: w.configure(cursor="hand2"))
                w.tag_bind(name, "<Leave>", lambda _e: w.configure(cursor="arrow"))
        w.insert("end", text[pos:])
        # read-only but selectable/copyable
        def _block(e):
            if e.state & 0x4 and e.keysym.lower() in ("c", "a", "insert"):
                return None
            return "break"
        w.bind("<Key>", _block)
        w.bind("<Configure>", lambda _e, _w=w: self._fit_text(_w))
        return w

    def _fit_text(self, w) -> None:
        try:
            lines = w.count("1.0", "end", "displaylines")
            n = int(lines[0]) if lines else 1
            if int(w.cget("height")) != max(1, n):
                w.configure(height=max(1, n))
        except Exception:
            pass

    @staticmethod
    def _score_color(score: float) -> str:
        if score >= 0.7:
            return "#1f8a3d"
        if score >= 0.55:
            return "#c98a02"
        return "#6b7280"


class SettingsTab(ctk.CTkFrame):
    """Interest examples, threshold, interval, per-channel watch toggles."""

    def __init__(self, master, app: "DigestApp") -> None:
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        top.grid_columnconfigure(0, weight=1)

        head = ctk.CTkFrame(top, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            head, text="Темы", font=ctk.CTkFont(size=15, weight="bold")
        ).pack(side="left")
        ctk.CTkLabel(
            head,
            text="  посты группируются по теме, к которой подходят больше всего",
            text_color="gray",
        ).pack(side="left")
        ctk.CTkButton(
            head, text="+ Добавить тему", width=140, command=lambda: self._add_topic_row()
        ).pack(side="right")
        ctk.CTkButton(
            head, text="Стандартные темы", width=140, fg_color=("#e8f0fe", "#243049"),
            hover_color=("#d6e4fd", "#2f3f60"), text_color=("#1a56c4", "#8ab4ff"),
            command=self._add_default_topics,
        ).pack(side="right", padx=(0, 8))

        cols = ctk.CTkFrame(top, fg_color="transparent")
        cols.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        cols.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(cols, text="Название темы", width=190, anchor="w",
                     text_color="gray").grid(row=0, column=0, padx=(6, 6), sticky="w")
        ctk.CTkLabel(cols, text="Слова и фразы через запятую", anchor="w",
                     text_color="gray").grid(row=0, column=1, sticky="w")

        self.topics_frame = ctk.CTkScrollableFrame(top, height=210)
        self.topics_frame.grid(row=2, column=0, sticky="ew", pady=(2, 8))
        self.topics_frame.grid_columnconfigure(1, weight=1)
        self._topic_rows: list[dict] = []

        ctk.CTkLabel(
            top,
            text="Дополнительные примеры интересов (необязательно, по строке):",
            anchor="w",
            text_color="gray",
        ).grid(row=3, column=0, sticky="ew")
        self.examples_box = ctk.CTkTextbox(top, height=50)
        self.examples_box.grid(row=4, column=0, sticky="ew", pady=(4, 8))

        params_frame = ctk.CTkFrame(self, fg_color="transparent")
        params_frame.grid(row=2, column=0, sticky="ew", padx=10)
        params_frame.grid_columnconfigure((1, 3), weight=1)

        ctk.CTkLabel(params_frame, text="Порог релевантности:").grid(
            row=0, column=0, sticky="w", pady=6
        )
        self.threshold_entry = ctk.CTkEntry(params_frame, width=80)
        self.threshold_entry.grid(row=0, column=1, sticky="w", padx=(6, 20))

        ctk.CTkLabel(params_frame, text="Интервал проверки (мин):").grid(
            row=0, column=2, sticky="w", pady=6
        )
        self.interval_entry = ctk.CTkEntry(params_frame, width=80)
        self.interval_entry.grid(row=0, column=3, sticky="w", padx=(6, 0))

        self.auto_topics_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            params_frame,
            text="Создавать новые темы автоматически (если набралась группа похожих постов, "
            "не подошедших ни под одну тему)",
            variable=self.auto_topics_var,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 6))

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=3, column=0, sticky="ew", padx=10, pady=(14, 0))
        ctk.CTkLabel(
            bar, text="Каналы", font=ctk.CTkFont(size=15, weight="bold")
        ).pack(side="left")
        self.channel_count_label = ctk.CTkLabel(bar, text="", text_color="gray")
        self.channel_count_label.pack(side="left", padx=(8, 0))
        ctk.CTkButton(
            bar, text="Снять все", width=100, fg_color="gray",
            command=lambda: self._set_visible_channels(False),
        ).pack(side="right")
        ctk.CTkButton(
            bar, text="Выбрать все", width=110,
            command=lambda: self._set_visible_channels(True),
        ).pack(side="right", padx=(0, 6))
        self.channel_filter_var = ctk.StringVar()
        self.channel_filter_var.trace_add("write", lambda *_: self._apply_channel_filter())
        ctk.CTkEntry(
            bar, textvariable=self.channel_filter_var, width=210,
            placeholder_text="Поиск по названию…",
        ).pack(side="right", padx=(0, 10))

        self.channels_frame = ctk.CTkScrollableFrame(self, height=180)
        self.channels_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=(4, 10))
        self.grid_rowconfigure(4, weight=1)
        # channel_id -> (var, username, checkbox widget, lowercase title)
        self._channel_vars: dict[int, tuple[ctk.BooleanVar, str, object, str]] = {}

        # Footer lives in the tab container and is packed BEFORE the tab
        # body, so the Save button is always visible however tall the
        # settings content gets.
        footer = ctk.CTkFrame(master, fg_color="transparent")
        footer.pack(side="bottom", fill="x", padx=10, pady=(0, 6))
        ctk.CTkButton(
            footer, text="Сохранить настройки", height=38, width=200,
            command=self._on_save,
        ).pack(side="right")
        self.status_label = ctk.CTkLabel(
            footer, text="", text_color="gray", anchor="w", justify="left", wraplength=560
        )
        self.status_label.pack(side="left", fill="x", expand=True)

        self.load(app.config)

    def _add_topic_row(self, name: str = "", keywords: str = "") -> None:
        row = len(self._topic_rows)
        name_entry = ctk.CTkEntry(
            self.topics_frame, width=190, placeholder_text="например: Авиация"
        )
        kw_entry = ctk.CTkEntry(
            self.topics_frame, placeholder_text="самолёты, аэропорты, Boeing, ВКС"
        )
        entry = {"name": name_entry, "kw": kw_entry}

        def remove() -> None:
            for w in (name_entry, kw_entry, del_btn):
                w.destroy()
            self._topic_rows.remove(entry)

        del_btn = ctk.CTkButton(
            self.topics_frame,
            text="✕",
            width=32,
            fg_color=("#e5b0b0", "#7a3a3a"),
            hover_color=("#d99", "#a04a4a"),
            command=remove,
        )
        name_entry.grid(row=row, column=0, padx=(2, 6), pady=3, sticky="w")
        kw_entry.grid(row=row, column=1, padx=(0, 6), pady=3, sticky="ew")
        del_btn.grid(row=row, column=2, padx=(0, 2), pady=3)
        if name:
            name_entry.insert(0, name)
        if keywords:
            kw_entry.insert(0, keywords)
        self._topic_rows.append(entry)

    def _add_default_topics(self) -> None:
        """Add the ready-made topics that are not in the list yet (by name)."""
        from config import default_topics

        have = {e["name"].get().strip().lower() for e in self._topic_rows}
        for t in default_topics():
            if t.name.lower() not in have:
                self._add_topic_row(t.name, ", ".join(t.keywords))

    def _collect_topics(self) -> list[TopicConfig]:
        topics = []
        for entry in self._topic_rows:
            name = entry["name"].get().strip()
            if not name:
                continue
            kws = [k.strip() for k in entry["kw"].get().split(",") if k.strip()]
            topics.append(TopicConfig(name=name, keywords=kws))
        return topics

    def load(self, config: AppConfig) -> None:
        for w in self.topics_frame.winfo_children():
            w.destroy()
        self._topic_rows.clear()
        for t in config.topics:
            self._add_topic_row(t.name, ", ".join(t.keywords))
        if not config.topics:
            self._add_topic_row()
        self.examples_box.delete("1.0", "end")
        self.examples_box.insert("1.0", "\n".join(config.interest_examples))
        self.threshold_entry.delete(0, "end")
        self.threshold_entry.insert(0, str(config.similarity_threshold))
        self.interval_entry.delete(0, "end")
        self.interval_entry.insert(0, str(config.check_interval_minutes))
        self.auto_topics_var.set(bool(config.auto_topics))
        self.refresh_channels()

    def refresh_channels(self) -> None:
        """Populate the channel checklist from storage (all discovered
        channels). Rebuilt only when the set of channels changes, so
        unsaved checkbox changes aren't lost on every feed update."""
        try:
            channels = self.app.storage.list_channels() if self.app.storage else []
        except Exception:
            channels = []
        if not channels:
            if not self._channel_vars and not getattr(self, "_empty_hint", None):
                self._empty_hint = ctk.CTkLabel(
                    self.channels_frame,
                    text="Список появится после первой проверки.",
                    text_color="gray",
                )
                self._empty_hint.pack(anchor="w", padx=6, pady=6)
            return
        if getattr(self, "_empty_hint", None):
            self._empty_hint.destroy()
            self._empty_hint = None
        if set(self._channel_vars) == {c.channel_id for c in channels}:
            return  # nothing new

        old_state = {cid: v[0].get() for cid, v in self._channel_vars.items()}
        for child in self.channels_frame.winfo_children():
            child.destroy()
        self._channel_vars.clear()

        off_ids = {c.channel_id for c in self.app.config.channels if not c.watch and c.channel_id is not None}
        off_names = {c.username for c in self.app.config.channels if not c.watch and c.username}

        for ch in sorted(channels, key=lambda c: (c.title or c.username or "").lower()):
            if ch.channel_id in old_state:
                checked = old_state[ch.channel_id]
            else:
                checked = not (
                    ch.channel_id in off_ids or (ch.username and ch.username in off_names)
                )
            var = ctk.BooleanVar(value=checked)
            var.trace_add("write", lambda *_: self._update_channel_count())
            label = ch.title or ch.username or str(ch.channel_id)
            box = ctk.CTkCheckBox(self.channels_frame, text=label, variable=var)
            self._channel_vars[ch.channel_id] = (var, ch.username or "", box, label.lower())
        self._apply_channel_filter()

    def _apply_channel_filter(self) -> None:
        needle = self.channel_filter_var.get().strip().lower()
        for _cid, (_var, _u, box, low) in self._channel_vars.items():
            box.pack_forget()
            if not needle or needle in low:
                box.pack(anchor="w", padx=6, pady=3)
        self._update_channel_count()

    def _set_visible_channels(self, value: bool) -> None:
        """Check/uncheck every channel matching the current search filter."""
        needle = self.channel_filter_var.get().strip().lower()
        for _cid, (var, _u, _box, low) in self._channel_vars.items():
            if not needle or needle in low:
                var.set(value)

    def _update_channel_count(self) -> None:
        total = len(self._channel_vars)
        on = sum(1 for v in self._channel_vars.values() if v[0].get())
        self.channel_count_label.configure(text=f"выбрано {on} из {total}" if total else "")

    def _on_save(self) -> None:
        try:
            self._do_save()
        except Exception as e:
            logger.exception("Не удалось сохранить настройки")
            self.status_label.configure(text=f"Ошибка сохранения: {e}", text_color="red")

    def _do_save(self) -> None:
        examples = [
            line.strip()
            for line in self.examples_box.get("1.0", "end").splitlines()
            if line.strip()
        ]
        topics = self._collect_topics()
        if not topics and not examples:
            self.status_label.configure(
                text="Добавьте хотя бы одну тему или пример интересов.", text_color="red"
            )
            return
        try:
            threshold = float(self.threshold_entry.get().strip())
            interval = int(self.interval_entry.get().strip())
        except ValueError:
            self.status_label.configure(
                text="Порог должен быть числом (0-1), интервал -- целым числом минут.",
                text_color="red",
            )
            return

        # Store only the channels switched OFF; all others are watched.
        channels = [
            ChannelConfig(username=username, watch=False, channel_id=cid)
            for cid, (var, username, _box, _low) in self._channel_vars.items()
            if not var.get()
        ]
        if self._channel_vars and not any(v[0].get() for v in self._channel_vars.values()):
            self.status_label.configure(
                text="Не выбран ни один канал.", text_color="red"
            )
            return

        import dataclasses

        # Keep every setting the editor doesn't expose (speed options, LLM...).
        new_config = dataclasses.replace(
            self.app.config,
            check_interval_minutes=interval,
            similarity_threshold=threshold,
            topics=topics,
            interest_examples=examples,
            channels=channels,
            auto_topics=bool(self.auto_topics_var.get()),
        )
        save_app_config(new_config)
        logger.info(
            "Настройки сохранены: тем %d (%s)",
            len(topics),
            ", ".join(t.name for t in topics),
        )
        self.app.config = new_config
        set_active_config(new_config)
        self.app.clear_feed()  # results depend on topics -> rescan from scratch
        self.app.worker.request_reload()
        self.status_label.configure(
            text="Сохранено. Фильтр перезагружается в фоне...", text_color="gray"
        )


class AskDialog(ctk.CTkToplevel):
    """Modal input dialog (login code / phone / 2FA password).

    Needed because the frozen GUI exe has no usable stdin, so Telethon's
    default input() prompt can't be used.
    """

    def __init__(self, master, prompt: str, secret: bool = False) -> None:
        super().__init__(master)
        self.title("Вход в Telegram")
        self.geometry("420x230")
        self.resizable(False, False)
        self.result: Optional[str] = None

        ctk.CTkLabel(self, text=prompt, justify="left", wraplength=380).pack(
            padx=20, pady=(18, 8), anchor="w"
        )
        self.entry = ctk.CTkEntry(
            self, width=380, font=ctk.CTkFont(size=18), show="*" if secret else ""
        )
        self.entry.pack(padx=20, pady=6)
        self.error_label = ctk.CTkLabel(self, text="", text_color="#d9534f")
        self.error_label.pack()

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(pady=10)
        ctk.CTkButton(row, text="OK", width=120, command=self._ok).pack(
            side="left", padx=6
        )
        ctk.CTkButton(
            row, text="Отмена", width=120, fg_color="gray", command=self._cancel
        ).pack(side="left", padx=6)

        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.after(150, self._grab)

    def _grab(self) -> None:
        try:
            self.lift()
            self.attributes("-topmost", True)
            self.grab_set()
            self.entry.focus_force()
        except Exception:
            pass

    def _ok(self) -> None:
        value = self.entry.get().strip()
        if not value:
            self.error_label.configure(text="Поле пустое")
            return
        self.result = value
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


class QueueLogHandler(logging.Handler):
    """Puts formatted log lines on a queue; the GUI drains it (thread-safe)."""

    def __init__(self, q: "queue.Queue[tuple[str, str]]") -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put_nowait((record.levelname, self.format(record)))
        except Exception:
            pass


class DigestApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Telegram Digest")
        self.geometry("920x820")
        self.minsize(760, 640)

        self.config: AppConfig = load_app_config()
        set_active_config(self.config)
        self.storage: Optional[Storage] = None  # worker owns the real one; GUI
        # opens its own read-only-ish handle just to list channels for Settings.
        try:
            from config import resolve_path

            self.storage = Storage(resolve_path(self.config.db_path))
        except Exception:
            logger.warning("Could not open storage for channel list", exc_info=True)

        self.worker = DigestWorker()

        self.log_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        handler = QueueLogHandler(self.log_queue)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        logging.getLogger().addHandler(handler)
        self._asking = False
        self._collapsed: set[str] = {OTHER_TOPIC}
        self._group_limit: dict[str, int] = {}
        self._summaries: dict[str, str] = {}
        self._summarizing: set[str] = set()
        self._llm_results: "queue.Queue[tuple[str, str, bool]]" = queue.Queue()
        self._group_widgets: list = []

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        self.feed_tab = self.tabview.add("Лента")
        self.settings_tab_container = self.tabview.add("Настройки")
        self.log_tab = self.tabview.add("Журнал")
        self._build_log_tab()

        self._build_feed_tab()
        self.settings_tab = SettingsTab(self.settings_tab_container, self)
        self.settings_tab.pack(fill="both", expand=True)

        self.status_bar = ctk.CTkLabel(
            self, text="Запуск...", anchor="w", justify="left", wraplength=720
        )
        self.status_bar.pack(fill="x", padx=10, pady=6)

        self.tray = TrayIcon(
            on_show=self._show_from_tray,
            on_check_now=self._check_now,
            on_quit=self._quit,
        )
        self.tray.start()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._bind_scroll_keys()

        self._refresh_topic_menu()
        self.worker.start()
        self.after(POLL_INTERVAL_MS, self._poll_worker)

    def _build_log_tab(self) -> None:
        bar = ctk.CTkFrame(self.log_tab, fg_color="transparent")
        bar.pack(fill="x", padx=4, pady=4)
        ctk.CTkButton(bar, text="Очистить", width=100, command=self._clear_log).pack(
            side="left"
        )
        ctk.CTkButton(
            bar, text="Открыть файл лога", width=150, command=self._open_log_file
        ).pack(side="left", padx=8)
        self.log_box = ctk.CTkTextbox(
            self.log_tab, wrap="word", font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.log_box.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.log_box.tag_config("err", foreground="#ff6b6b")
        self.log_box.tag_config("warn", foreground="#f0ad4e")
        self.log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _open_log_file(self) -> None:
        try:
            import os

            os.startfile(str(LOG_PATH))  # type: ignore[attr-defined]
        except Exception:
            logger.warning("Не удалось открыть %s", LOG_PATH)

    def _drain_logs(self) -> None:
        lines = []
        for _ in range(200):
            try:
                lines.append(self.log_queue.get_nowait())
            except Exception:
                break
        if not lines:
            return
        self.log_box.configure(state="normal")
        for level, text in lines:
            tag = "err" if level in ("ERROR", "CRITICAL") else "warn" if level == "WARNING" else None
            self.log_box.insert("end", text + "\n", tag) if tag else self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _ask_user(self, event: WorkerEvent) -> None:
        if self._asking:
            return
        self._asking = True
        self.deiconify()
        self.lift()
        dlg = AskDialog(self, event.prompt or "Введите значение:", secret=event.secret)
        self.wait_window(dlg)
        self._asking = False
        self.worker.answer(dlg.result)

    def _build_feed_tab(self) -> None:
        self.feed_tab.grid_columnconfigure(0, weight=1)
        self.feed_tab.grid_rowconfigure(1, weight=1)

        toolbar = ctk.CTkFrame(self.feed_tab, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ctk.CTkButton(
            toolbar, text="Проверить сейчас", command=self._check_now,
            height=34, corner_radius=10, font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left")

        ctk.CTkLabel(toolbar, text="Тема:").pack(side="left", padx=(16, 4))
        self.topic_var = ctk.StringVar(value=ALL_TOPICS)
        self.topic_menu = ctk.CTkOptionMenu(
            toolbar,
            variable=self.topic_var,
            values=[ALL_TOPICS],
            command=self._on_topic_selected,
            width=220,
        )
        self.topic_menu.pack(side="left")

        self.feed_count_label = ctk.CTkLabel(toolbar, text="Постов: 0", text_color="gray")
        self.feed_count_label.pack(side="left", padx=12)

        self.feed_frame = ctk.CTkScrollableFrame(self.feed_tab)
        self.feed_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
        self._feed_cards: list[PostCard] = []
        self._topics: list[str] = []  # insertion-ordered, de-duplicated

    def clear_feed(self) -> None:
        for card in self._feed_cards:
            card.destroy()
        self._feed_cards = []
        self._topics = []
        self._refresh_topic_menu()
        self._render_feed()

    def _add_posts(self, posts: list[ScoredPost]) -> list[ScoredPost]:
        """Add posts not already in the feed; returns the genuinely new ones."""
        existing = {
            (c.scored.post.channel_id, c.scored.post.message_id) for c in self._feed_cards
        }
        new_posts = []
        # irrelevant first: each card is inserted at the top, so relevant ones end up above them
        posts = sorted(posts, key=lambda s: s.result.is_relevant)
        for scored in posts:
            key = (scored.post.channel_id, scored.post.message_id)
            if key in existing:
                continue
            existing.add(key)
            new_posts.append(scored)
            card = PostCard(
                self.feed_frame, scored, on_move=self._move_card,
                topic_names=[t.name for t in self.config.topics],
                on_channel=self._set_channel_topic,
                on_confirm=self._confirm_card, on_reject=self._reject_card,
            )
            self._feed_cards.insert(0, card)
            if card.topic not in self._topics:
                self._topics.append(card.topic)

        # Cap total cards kept; drop the oldest ones beyond the limit.
        # (relevant cards and the 'Прочее' ones are capped separately, otherwise
        # a big pile of relevant posts pushes all 'Прочее' cards out)
        rel_cards = [c for c in self._feed_cards if c.topic != OTHER_TOPIC]
        oth_cards = [c for c in self._feed_cards if c.topic == OTHER_TOPIC]
        keep = set(map(id, rel_cards[:MAX_FEED_ITEMS])) | set(map(id, oth_cards[:100]))
        for card in self._feed_cards:
            if id(card) not in keep:
                card.destroy()
        self._feed_cards = [c for c in self._feed_cards if id(c) in keep]

        self._refresh_topic_menu()
        self._render_feed()
        return new_posts

    def _bind_scroll_keys(self) -> None:
        """Keyboard scrolling of the feed: arrows, PgUp/PgDn, Space, Home/End,
        and j/k. Active on the 'Лента' tab, never while typing in a field."""
        def scroll(kind, amount):
            def handler(event=None):
                try:
                    if self.tabview.get() != "Лента":
                        return None
                    w = self.focus_get()
                    cls = w.winfo_class() if w is not None else ""
                    if cls in ("Entry", "TEntry", "Text", "TCombobox") and kind != "page":
                        return None
                    canvas = self.feed_frame._parent_canvas
                    if kind == "units":
                        canvas.yview_scroll(amount, "units")
                    elif kind == "page":
                        canvas.yview_scroll(amount, "pages")
                    elif kind == "home":
                        canvas.yview_moveto(0.0)
                    elif kind == "end":
                        canvas.yview_moveto(1.0)
                except Exception:
                    return None
                return "break"
            return handler

        binds = {
            "<Up>": ("units", -3), "<Down>": ("units", 3),
            "<k>": ("units", -3), "<j>": ("units", 3),
            "<Prior>": ("page", -1), "<Next>": ("page", 1),
            "<space>": ("page", 1), "<Shift-space>": ("page", -1),
            "<Home>": ("home", 0), "<End>": ("end", 0),
        }
        for key, (kind, amount) in binds.items():
            self.bind_all(key, scroll(kind, amount))

    def _refresh_topic_menu(self) -> None:
        configured = [t.name for t in self.config.topics]
        extra = [t for t in self._topics if t not in configured]
        values = [ALL_TOPICS] + configured + sorted(extra, key=str.lower)
        current = self.topic_var.get()
        self.topic_menu.configure(values=values)
        if current not in values:
            self.topic_var.set(ALL_TOPICS)

    def _on_topic_selected(self, _choice: str) -> None:
        self._render_feed()

    def _move_card(self, card, topic: str) -> None:
        """User says: this post belongs to <topic> (or to none). Moves the
        card and remembers the correction -- it becomes an example for the
        local LLM on the next checks."""
        from embed_filter import FilterResult

        post = card.scored.post
        if topic == OTHER_TOPIC:
            card.scored.result = FilterResult(False, 0.0, None, "вручную")
            saved = ""
        else:
            example = next((t.as_example() for t in self.config.topics if t.name == topic), topic)
            card.scored.result = FilterResult(True, 0.95, example, "вручную")
            saved = topic
        card.topic = topic_label(card.scored)
        card._built = False  # re-render with the new chips
        try:
            if self.storage and post.text.strip():
                self.storage.add_feedback(post.text, saved)
        except Exception:
            logger.debug("не удалось сохранить правку", exc_info=True)
        self._refresh_topic_menu()
        self._render_feed()
        self.status_bar.configure(
            text=f"Запомнил: «{_short(post.text, 40)}» → {topic}. Учту при следующих проверках."
        )

    def _confirm_card(self, card) -> None:
        """👍 the topic shown on the card is right: remember it as an example."""
        self._move_card(card, card.topic)

    def _reject_card(self, card) -> None:
        """👎 the post does not belong to the shown topic: remember it as 'not interesting'."""
        self._move_card(card, OTHER_TOPIC)

    def _set_channel_topic(self, card, topic: str) -> None:
        """User says: this whole channel belongs to <topic>. Remembered for
        good; all of the channel's posts in the feed that were not placed by
        hand move there too, and future posts without a clear topic land there."""
        from embed_filter import FilterResult

        post = card.scored.post
        try:
            if self.storage:
                self.storage.set_channel_category(post.channel_id, topic, "manual")
        except Exception:
            logger.debug("не удалось сохранить категорию канала", exc_info=True)
        example = next((t.as_example() for t in self.config.topics if t.name == topic), topic)
        moved = 0
        for c in self._feed_cards:
            sp = c.scored
            if sp.post.channel_id != post.channel_id or sp.result.source == "вручную":
                continue
            if sp.result.source in ("ИИ", "ИИ-фото") and c.topic == topic:
                continue
            if c.topic == topic:
                continue
            # strong engine verdicts (keywords/LLM) for another topic are kept
            if sp.result.is_relevant and sp.result.source in ("ИИ", "ИИ-фото", "слова"):
                continue
            sp.result = FilterResult(True, max(sp.result.score, 0.5), example, "канал")
            c.topic = topic_label(sp)
            c._built = False
            moved += 1
        self._refresh_topic_menu()
        self._render_feed()
        self.status_bar.configure(
            text=f"Канал «{_short(post.channel_title or '', 30)}» → «{topic}». Перенесено постов: {moved}."
        )

    GROUP_PAGE = 20

    def _pack_limited(self, key: str, cards: list) -> None:
        """Show the first N cards of a group, with a 'show more' button."""
        limit = self._group_limit.get(key, self.GROUP_PAGE)
        for card in cards[:limit]:
            card.pack(fill="x", padx=6, pady=4)
        if len(cards) > limit:
            def more(k=key):
                self._group_limit[k] = self._group_limit.get(k, self.GROUP_PAGE) + self.GROUP_PAGE
                self._render_feed()

            btn = ctk.CTkButton(
                self.feed_frame, text=f"Показать ещё ({len(cards) - limit})", height=30,
                fg_color=("#e8f0fe", "#243049"), hover_color=("#d6e4fd", "#2f3f60"),
                text_color=("#1a56c4", "#8ab4ff"), command=more,
            )
            btn.pack(pady=6)
            self._group_widgets.append(btn)

    def _render_feed(self) -> None:
        """Feed layout. "Все темы": posts grouped into collapsible topic
        sections (best-scoring topic first, best posts first inside).
        A specific topic selected: flat list of just that topic."""
        selected = self.topic_var.get()
        for w in self._group_widgets:
            w.destroy()
        self._group_widgets = []
        for card in self._feed_cards:
            card.pack_forget()

        total = len(self._feed_cards)
        if selected != ALL_TOPICS:
            cards = [c for c in self._feed_cards if c.topic == selected]
            cards.sort(key=lambda c: c.scored.result.score, reverse=True)
            self._pack_limited(selected, cards)
            self.feed_count_label.configure(text=f"Постов: {len(cards)} из {total}")
            return

        groups: dict[str, list[PostCard]] = {}
        for card in self._feed_cards:
            groups.setdefault(card.topic, []).append(card)
        ordered = sorted(
            groups.items(),
            key=lambda kv: (kv[0] != OTHER_TOPIC, max(c.scored.result.score for c in kv[1])),
            reverse=True,
        )
        for topic, cards in ordered:
            cards.sort(key=lambda c: c.scored.result.score, reverse=True)
            collapsed = topic in self._collapsed
            head = ctk.CTkFrame(self.feed_frame, fg_color=topic_color(topic), corner_radius=12)
            head.pack(fill="x", padx=6, pady=(10, 2))
            self._group_widgets.append(head)
            ctk.CTkButton(
                head,
                text=f"{'▶' if collapsed else '▼'}  {_short(topic, 60)}  ({len(cards)})",
                anchor="w",
                fg_color="transparent",
                hover_color=("#d8dde6", "#2c3446"),
                text_color=TEXT_ON_CHIP,
                font=ctk.CTkFont(size=15, weight="bold"),
                command=lambda t=topic: self._toggle_group(t),
            ).pack(side="left", fill="x", expand=True, padx=4, pady=4)
            if self.config.llm_model:
                busy = topic in self._summarizing
                ctk.CTkButton(
                    head,
                    text="Сводка…" if busy else "Сводка (ИИ)",
                    width=110,
                    state="disabled" if busy else "normal",
                    command=lambda t=topic, c=cards: self._summarize_group(t, c),
                ).pack(side="right", padx=6, pady=4)
            summary = self._summaries.get(topic)
            if summary and not collapsed:
                lbl = ctk.CTkLabel(
                    self.feed_frame,
                    text=summary,
                    justify="left",
                    anchor="w",
                    wraplength=640,
                    fg_color=("#f4f6fa", "#1a2030"),
                    corner_radius=10,
                )
                lbl.pack(fill="x", padx=6, pady=(0, 4), ipadx=8, ipady=6)
                self._group_widgets.append(lbl)
            if not collapsed:
                self._pack_limited(topic, cards)
        self.feed_count_label.configure(text=f"Постов: {total}, тем: {len(ordered)}")

    def regroup_feed(self) -> None:
        """Re-label existing cards after the topic list was edited."""
        self._topics = []
        for card in self._feed_cards:
            new_topic = topic_label(card.scored)
            card.topic = new_topic
            if new_topic not in self._topics:
                self._topics.append(new_topic)
        self._refresh_topic_menu()
        self._render_feed()

    def _toggle_group(self, topic: str) -> None:
        if topic in self._collapsed:
            self._collapsed.discard(topic)
        else:
            self._collapsed.add(topic)
        self._render_feed()

    def _summarize_group(self, topic: str, cards: list) -> None:
        """Ask the local LLM (Ollama) for a summary of one topic group.
        Runs in a thread; result comes back through a queue."""
        import threading

        from llm import summarize_topic

        self._summarizing.add(topic)
        self._render_feed()
        texts = [c.scored.post.text for c in cards]
        model, url = self.config.llm_model, self.config.llm_url

        def job() -> None:
            try:
                text = summarize_topic(topic, texts, model, url)
                self._llm_results.put((topic, text or "(пустой ответ)", True))
            except Exception as e:
                logger.error("Сводка не удалась (ИИ-модель не загружена?): %s", e)
                self._llm_results.put((topic, f"Не удалось получить сводку: {e}", False))

        threading.Thread(target=job, daemon=True).start()

    # -- worker event polling (runs on the main/Tk thread) -----------------
    def _poll_worker(self) -> None:
        drained = 0
        while drained < 50:
            try:
                event: WorkerEvent = self.worker.events.get_nowait()
            except Exception:
                break
            drained += 1
            self._handle_worker_event(event)
        self._drain_logs()
        changed = False
        while True:
            try:
                topic, text, ok = self._llm_results.get_nowait()
            except Exception:
                break
            self._summarizing.discard(topic)
            self._summaries[topic] = text
            changed = True
        if changed:
            self._render_feed()
        self.after(POLL_INTERVAL_MS, self._poll_worker)

    def _handle_worker_event(self, event: WorkerEvent) -> None:
        if event.type == "ask":
            # Schedule so we don't open a modal dialog inside the poll loop.
            self.after(50, lambda e=event: self._ask_user(e))
        elif event.type == "status" and event.status:
            self.status_bar.configure(text=event.status, text_color=("black", "white"))
            if event.status == "Ожидание" or event.status.startswith("Найдено каналов"):
                self.settings_tab.refresh_channels()  # channel list is now in storage
        elif event.type == "topics_added":
            try:
                self.config = load_app_config()
                set_active_config(self.config)
                self.settings_tab.load(self.config)
            except Exception:
                logger.exception("Не удалось обновить список тем")
            self.clear_feed()  # regroup everything under the new topics
            self.status_bar.configure(
                text="Создана новая тема: " + ", ".join(f"«{n}»" for n in event.new_topics)
                + ". Её можно переименовать или удалить в Настройках.",
                text_color=("black", "white"),
            )
        elif event.type == "error" and event.error:
            self.status_bar.configure(
                text=f"Ошибка: {event.error}\n(подробности во вкладке «Журнал»)",
                text_color="#d9534f",
            )
            logger.error("Ошибка: %s", event.error)
        elif event.type == "posts" and event.posts:
            new_posts = self._add_posts(event.posts)
            self.settings_tab.refresh_channels()
            if new_posts:
                rel = [s for s in new_posts if s.result.is_relevant]
                if rel:
                    notify_new_posts(len(rel), first_link=rel[0].post.link)
                if rel:
                    self.tray.notify(
                        "Telegram Digest", f"{len(rel)} новых релевантных постов"
                    )

    # -- tray / window lifecycle -------------------------------------------
    def _show_from_tray(self) -> None:
        self.after(0, self._deiconify)

    def _deiconify(self) -> None:
        self.deiconify()
        self.lift()
        self.focus_force()

    def _check_now(self) -> None:
        self.worker.check_now()

    def _on_close(self) -> None:
        # Minimize to tray instead of quitting, if a tray icon is running.
        if self.tray.available:
            self.withdraw()
        else:
            self._quit()

    def _quit(self) -> None:
        """Quit for real. Called from the tray thread too, so every step is
        guarded and the process is force-exited at the end -- otherwise a
        failure here left an invisible zombie holding the Telegram session
        ("database is locked" on the next start)."""
        import os
        import threading

        # Hard exit even if something below hangs or raises.
        killer = threading.Timer(3.0, lambda: os._exit(0))
        killer.daemon = True
        killer.start()
        for step in (self.worker.stop, self.tray.stop):
            try:
                step()
            except Exception:
                logger.warning("quit step failed", exc_info=True)
        try:
            if self.storage:
                self.storage.close()
        except Exception:
            logger.warning("storage close failed", exc_info=True)
        try:
            self.after(0, self.destroy)
        except Exception:
            pass
        os._exit(0)


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        fh = RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass
    if sys.stderr is not None:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)


def main() -> None:
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    _setup_logging()
    logger.info("Запуск Telegram Digest, лог: %s", LOG_PATH)
    app = DigestApp()
    app.mainloop()


if __name__ == "__main__":
    main()
