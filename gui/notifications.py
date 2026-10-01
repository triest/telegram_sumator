"""Windows GUI notification helper.

Thin wrapper kept for gui/app.py's import path; the actual cross-platform
dispatch logic (plyer / win10toast / termux-notification) lives in the
core notify.py, shared with the CLI and the Termux entry point so all
three degrade the same way when a backend isn't available.
"""
from __future__ import annotations

from notify import notify, notify_new_posts  # noqa: F401  (re-exported)
