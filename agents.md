# Agents & Plan — Telegram Digest (Local Embeddings)

## Agents

### 1. Architect
- Decide on sentence-transformers + multilingual MiniLM.
- Design interest-examples format.
- Ensure the same code path works on Windows and Termux.
- Keep generative llama.cpp as optional future path only.

### 2. Telegram Agent
- Telethon session management.
- Discover channels.
- Fetch only new messages (by last message_id).
- Return clean Post objects (text, date, channel, link, has_media).

### 3. Embed Filter Agent
- Load sentence-transformers model (CPU).
- Load user interest examples from config.
- Compute embeddings for examples once at startup (cache them).
- For each post: embed → max cosine similarity → decide + score.
- Threshold configurable (default 0.48).
- Must work without GPU.
- Graceful fallback to keyword filter if model fails to load.

### 4. Storage Agent
- SQLite: channels, last_seen, shown_posts, settings.
- Store interest examples and threshold.

### 5. GUI Agent (Windows)
- Main feed of filtered posts.
- Settings page: edit interest examples, threshold, interval, channel list.
- System tray + toast.

### 6. Android / Termux Agent
- Same Python core.
- Notifications via termux-notification or termux-api.
- Simple CLI or minimal Textual/TUI interface for phone.

## Phases

### Phase 0 — Skeleton
- Project structure
- requirements.txt (telethon, sentence-transformers, torch, customtkinter, pyyaml, etc.)
- config.example.yaml
- .env.example
- README (api_id/api_hash + how to download model)

### Phase 1 — CLI MVP (Windows)
1. Telethon login
2. List & select channels
3. Fetch new posts
4. Embed filter with interest examples
5. Print sorted by score
6. Save last_seen IDs

### Phase 2 — Windows App
1. SQLite
2. CustomTkinter GUI
3. Background worker
4. Tray + notifications
5. Interest examples editor

### Phase 3 — Make it Termux-ready
- Remove any Windows-only code from core
- Add termux notification helper
- Document how to run on Android

### Phase 4 — Optional later
- Tiny GGUF generative model support
- Better multilingual embeddings
- Export to Markdown

## config.example.yaml
```yaml
check_interval_minutes: 40
similarity_threshold: 0.48

interest_examples:
  - "Новые модели ИИ, релизы LLM, Claude, GPT, open-source модели"
  - "Python библиотеки, инструменты разработчика, релизы фреймворков"
  - "Стартапы, инвестиции, интересные продукты"
  - "Научные прорывы в AI и компьютерных науках"
  - "Мемы про программистов и нейросети"

global_exclude_keywords:
  - "реклама"
  - "казино"
  - "заработок без вложений"

channels:
  # leave empty = watch all, or list specific ones
  # - username: "somechannel"
  #   watch: true