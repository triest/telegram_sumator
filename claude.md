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
**Primary: Embedding + Cosine Similarity** (measured on real posts from the user's channels:
overall accuracy 0.85 vs 0.70 for the old MiniLM, see skill `.claude/skills/digest-recognition`)

- Library: `sentence-transformers`, model `Qwen/Qwen3-Embedding-0.6B` (config `embedding_model`).
  The old `paraphrase-multilingual-MiniLM-L12-v2` still works (threshold ~0.38) for weak devices.
- Posts are encoded with a task instruction prefix; topics are compared by their **name only**
  (keywords are matched exactly by `TopicKeywordBoost`). Default `similarity_threshold` 0.62.
- Thresholds are **model-specific**: changing the model means re-tuning the threshold.
- For each new post: max cosine similarity to the topics -> if >= threshold -> show it.
- Channel context: per-topic baseline from `NEUTRAL_PROBES` is subtracted, a channel gets a "home"
  topic only with a clear margin (generic topics like "кино" otherwise attract everything).
- Manual verdicts (👍 Верно / 👎 Не то / move to topic) are stored in SQLite `feedback` and applied
  to near-identical posts (cosine >= 0.85).
- Pictures: local CLIP (`image_topics.py`, multilingual text encoder) for picture-only / short-caption posts.
- Auto topics (`auto_topics: true`): posts that fit no topic are clustered; a big enough group becomes
  a new topic (name from the embedded LLM, keywords from the group).
- Preview = first 250 characters of the post. Everything runs in-process: **no Ollama**.

**Embedded LLM** (llama-cpp-python GGUF, only for "Сводка (ИИ)" and naming auto topics). It is NOT used
for classification (`llm_classify: false`): on real posts it was worse than embeddings (61% vs 85%) and slow on CPU.

**Fallback**: pure keyword/regex filter if the model is not downloaded.

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