# Claude Instructions — Telegram Channel Digest (Local Model)

## Goal
Build a personal Telegram channel digest tool that:
1. Collects new messages from subscribed channels.
2. Filters them using a **local** embedding model (no cloud LLM).
3. Shows only relevant posts with short preview + link.
4. Must work on **Windows** first, then **Android via Termux** with the same core code.

## Critical Constraints
- No Claude / OpenAI / any cloud LLM for filtering or summarization.
- All AI must be local.
- Same Python core must run on Windows and on Android (Termux).
- Model must be small enough to run on a mid-range Android phone (CPU).

## Preferred Local Model Approach
**Primary: Embedding + Cosine Similarity**

- Library: `sentence-transformers`
- Recommended model: `paraphrase-multilingual-MiniLM-L12-v2` (or `intfloat/multilingual-e5-small`)
- User provides 5–20 example texts of “what is interesting to me”.
- For each new post:
  - Compute embedding
  - Calculate max cosine similarity to user examples
  - If similarity ≥ threshold (default 0.45–0.55) → show it
- Score = similarity score
- Preview = first 250 characters of the post

**Optional fallback**: pure keyword/regex filter if model is not downloaded.

**Future optional**: tiny GGUF generative model via llama-cpp-python (Phi-3-mini / Qwen2-1.5B) for yes/no + summary. Not required for MVP.

## Tech Stack
- Python 3.11+
- Telethon
- sentence-transformers + torch (CPU)
- CustomTkinter (Windows GUI)
- SQLite
- YAML config
- plyer / win10toast for notifications
- PyInstaller for Windows
- Termux + termux-api for Android notifications

## Required Features (MVP)
1. Telethon login → local session file.
2. Auto-discover channels/supergroups.
3. User can enable/disable channels.
4. User writes interest examples in config or GUI (plain text).
5. Configurable similarity threshold and check interval.
6. Local filtering only.
7. List of matching posts sorted by score:
   - Channel
   - Time
   - Preview
   - Score
   - Open in Telegram
8. System tray + notifications (Windows).
9. Same core works in Termux on Android with notifications.

## Non-goals
- No cloud LLM calls.
- No reading private chats.
- No automatic posting/replies.

## Coding Rules
- Type hints.
- Async Telethon.
- Clear modules: telegram_client.py, embed_filter.py, storage.py, config.py, gui/
- Model is downloaded once and cached locally.
- Everything must be runnable offline after first model download.
- Secrets only in .env.

## Delivery Order
1. CLI MVP (Windows): login → fetch → embed filter → print.
2. Add SQLite + last_seen.
3. Windows GUI + tray.
4. Make core clean enough for Termux.
5. Termux version with notifications.