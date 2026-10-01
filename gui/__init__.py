"""Windows GUI package (Phase 2). CustomTkinter app + tray + notifications.

Kept separate from the core (config.py, telegram_client.py, embed_filter.py,
storage.py, pipeline.py) so that core stays Termux-clean (Phase 3): nothing
in this package is imported by the CLI or by core modules.
"""
