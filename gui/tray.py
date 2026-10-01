"""Windows system tray icon (pystray), so the app can live in the tray
instead of the taskbar/window list.

Runs in its own thread (pystray's `icon.run()` blocks), independent from
both the Tkinter mainloop and the DigestWorker's asyncio loop. All
callbacks it fires (show/check now/quit) are handed off to the GUI thread
via the callbacks passed in -- those callbacks must themselves be
thread-safe (see gui/app.py, which uses `Tk.after(0, ...)` from them).
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _build_icon_image():
    """A simple generated icon (paper-plane-ish dot), so we don't need to
    ship a binary .ico asset."""
    from PIL import Image, ImageDraw

    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((4, 4, size - 4, size - 4), fill=(0, 122, 255, 255))
    draw.polygon(
        [(18, 32), (46, 20), (34, 46), (30, 36)],
        fill=(255, 255, 255, 255),
    )
    return img


class TrayIcon:
    """Optional: if pystray/Pillow aren't available, `start()` no-ops and
    the app just runs without a tray icon (still usable as a normal
    window)."""

    def __init__(
        self,
        on_show: Callable[[], None],
        on_check_now: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        self._on_show = on_show
        self._on_check_now = on_check_now
        self._on_quit = on_quit
        self._icon = None
        self._thread: Optional[threading.Thread] = None

    @property
    def available(self) -> bool:
        try:
            import pystray  # noqa: F401
            from PIL import Image  # noqa: F401

            return True
        except ImportError:
            return False

    def start(self) -> None:
        if not self.available:
            logger.info("pystray/Pillow not installed; running without a tray icon.")
            return

        import pystray

        menu = pystray.Menu(
            pystray.MenuItem("Показать", self._handle_show, default=True),
            pystray.MenuItem("Проверить сейчас", self._handle_check_now),
            pystray.MenuItem("Выход", self._handle_quit),
        )
        self._icon = pystray.Icon(
            "telegram_digest", _build_icon_image(), "Telegram Digest", menu
        )
        self._thread = threading.Thread(
            target=self._icon.run, name="TrayIcon", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._icon is not None:
            self._icon.stop()

    def notify(self, title: str, message: str) -> None:
        """Balloon/tray notification, if the icon is running."""
        if self._icon is not None:
            try:
                self._icon.notify(message, title)
            except Exception:
                logger.debug("tray notify failed", exc_info=True)

    # -- pystray callbacks run on the tray thread; just forward them -------
    def _handle_show(self, icon, item) -> None:
        self._on_show()

    def _handle_check_now(self, icon, item) -> None:
        self._on_check_now()

    def _handle_quit(self, icon, item) -> None:
        self._on_quit()
