"""Entry point for the Windows GUI app (Phase 2).

Usage:
    python gui_main.py

Requires config.yaml / .env to already be set up (see README.md) -- same
prerequisites as the CLI (main.py).
"""
from __future__ import annotations

from gui.app import main

def _single_instance_guard():
    """Refuse to start a second copy: two processes sharing one Telegram
    session file cause 'database is locked'."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 47613))
    except OSError:
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                0,
                "Telegram Digest уже запущен (проверьте значок в трее и "
                "диспетчер задач). Закройте лишние копии.",
                "Telegram Digest",
                0x30,
            )
        except Exception:
            print("Telegram Digest уже запущен.")
        raise SystemExit(1)
    return sock  # keep referenced for the process lifetime


if __name__ == "__main__":
    _guard = _single_instance_guard()
    main()
