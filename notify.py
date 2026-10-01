"""Cross-platform local notification dispatcher.

This is core (not gui/) because it's shared by the Windows GUI, the plain
CLI, and the Termux entry point -- picking the right backend for whatever
OS it's actually running on:

    - Termux/Android: shells out to `termux-notification` (from the
      Termux:API app + `termux-api` package). Supports a tap action that
      opens the post link via `termux-open-url`.
    - Windows (or anywhere plyer works): plyer, falling back to
      win10toast on win32 if plyer fails.
    - Anything else / nothing available: logged only, never raises.

Detection is by capability (does the binary/library exist?), not by
`platform.system()` string-matching, so this keeps working unmodified if
Termux ever changes how it reports itself.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

APP_NAME = "Telegram Digest"


def is_termux() -> bool:
    return shutil.which("termux-notification") is not None


def notify(title: str, message: str, url: Optional[str] = None, timeout: int = 8) -> None:
    """Best-effort local notification. Never raises.

    `url`, if given, is used as a tap action on platforms that support it
    (Termux: opens the link via termux-open-url; ignored elsewhere for now).
    """
    if is_termux():
        if _notify_termux(title, message, url):
            return
    if getattr(sys, "frozen", False):
        # In the packaged .exe plyer/win10toast backends aren't bundled and
        # spew tracebacks; the GUI shows notifications via its tray icon.
        logger.debug("Frozen build: skipping plyer/win10toast toast")
        return
    if _notify_plyer(title, message, timeout):
        return
    if sys.platform == "win32" and _notify_win10toast(title, message, timeout):
        return
    logger.debug("No notification backend available; skipping toast: %s", title)


def _notify_termux(title: str, message: str, url: Optional[str]) -> bool:
    try:
        cmd = ["termux-notification", "--title", title, "--content", message]
        if url:
            cmd += ["--action", f"termux-open-url {url}"]
        subprocess.run(cmd, check=True, timeout=10, capture_output=True)
        return True
    except Exception:
        logger.debug("termux-notification failed", exc_info=True)
        return False


def _notify_plyer(title: str, message: str, timeout: int) -> bool:
    try:
        from plyer import notification

        notification.notify(
            title=title, message=message, app_name=APP_NAME, timeout=timeout
        )
        return True
    except Exception:
        logger.debug("plyer notification failed", exc_info=True)
        return False


def _notify_win10toast(title: str, message: str, timeout: int) -> bool:
    try:
        from win10toast import ToastNotifier

        ToastNotifier().show_toast(title, message, duration=timeout, threaded=True)
        return True
    except Exception:
        logger.debug("win10toast notification failed", exc_info=True)
        return False


def notify_new_posts(count: int, first_link: Optional[str] = None) -> None:
    if count <= 0:
        return
    if count == 1:
        notify(APP_NAME, "1 новый релевантный пост", url=first_link)
    else:
        notify(APP_NAME, f"{count} новых релевантных постов")
