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
Local embeddings (`Qwen/Qwen3-Embedding-0.6B`, config `embedding_model`, threshold `similarity_threshold` 0.62 —
thresholds are model-specific) + exact topic keywords + local CLIP for pictures. Everything runs in-process:
**no Ollama, no cloud**. The embedded GGUF LLM is only for "Сводка (ИИ)" and naming auto topics, not for classification.
Fallback: keyword filter if the model is not downloaded.

Details (architecture, measured results, how to benchmark, pitfalls) are in the skill
`digest-recognition` — load it when touching recognition. History/decisions: `digest-session-log`.

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
## Working agreements (from the owner)
- **No Ollama.** Everything runs inside the program (sentence-transformers, CLIP, embedded llama-cpp GGUF).
- **Desktop (Windows) first**; Termux / Android get the same ideas later.
- **Test on real posts** from the connected Telegram (read-only, broadcast channels only), not on made-up samples.
  Re-check the final result through the real app code path after implementing ("сверь результат").
- Prefer the measurably better option and implement it; report numbers honestly (what got better, what did not).
- The GPU is shared with ComfyUI generations: heavy runs can be slow; wait instead of killing other processes.
- Commit and push only when asked. Secrets stay in `.env`; `config.yaml`, `session/`, `data/`, `models/` are git-ignored.
- Run from source with `run_source.bat` (`TelegramDigest.exe` / `run_gui.bat` use the old `dist` build until rebuilt).

## Skills in this repo
Skills load on demand (only their name/description sit in context; the body opens when the task matches):
- `.claude/skills/digest-recognition` — architecture, measured results and how to benchmark recognition.
- `.claude/skills/digest-auto-topics` — auto topics: algorithm, constants, live-check recipe, Android port (2026-10-19).
- `.claude/skills/digest-session-log` — what was decided/done in the 2026-10-17 tuning session; open questions.

## Open work
- Port Qwen3-Embedding / feedback / baseline fixes to `termux_main.py` (only `embedding_model` is wired for Termux). Android `core.py` already has auto topics (untuned thresholds, Kotlin not compiled).
- Build the Android APK (no JDK/SDK found on this machine; old `TelegramDigest.apk` predates auto topics).
- Rebuild the PyInstaller exe (the old `dist` build lacks all recent changes).
- Validate in the running GUI: 👍/👎 buttons, auto-topics checkbox, CLIP model download on first picture.
- Auto topics: merge into similar topics is implemented but not yet seen on live data; keywords still sometimes generic (capped at 5).
- Re-tune `similarity_threshold` (0.62) after the user's own 👍/👎 data accumulates in the `feedback` table.
