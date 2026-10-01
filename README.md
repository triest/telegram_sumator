# Telegram Channel Digest (Local Model)

Personal Telegram channel digest: collects new posts from your channels,
filters them locally with an embedding model (no cloud LLM), and shows
only what matches your interests, with a short preview + link.

Runs on Windows first; the core (`config.py`, `telegram_client.py`,
`embed_filter.py`, `storage.py`, `main.py`) is written to also run
unchanged on Android via Termux.

## How it works

1. Telethon logs into your Telegram account (local session file, nothing
   sent to any third party).
2. For each watched channel, only messages newer than the last-seen id are
   fetched.
3. Each post's text is embedded with a local multilingual sentence-transformers
   model and compared (cosine similarity) against 5-20 example texts you
   write describing "what's interesting to me".
4. Posts scoring above your threshold are printed, sorted by score, with
   channel, time, a short preview, the score, and a link to open in Telegram.
5. If the embedding model isn't available, a keyword fallback filter is
   used instead so the tool still works.

No Claude/OpenAI/any cloud LLM is used anywhere in this pipeline.

## Quick install (Windows, one click)

1. Double-click **`install.bat`**. It will: create a virtualenv, detect
   whether you have an NVIDIA GPU and install the matching `torch` build,
   install all dependencies, create `.env`/`config.yaml` from the example
   files, build a standalone **`TelegramDigest.exe`**, and open `.env`
   and `config.yaml` in Notepad for you to fill in (Telegram API
   credentials + your interests).
2. After that, just double-click **`Telegram Digest.vbs`** (or
   `TelegramDigest.exe` directly) any time you want to run it -- no
   console window, no typing commands.
3. If the `.exe` build fails for some reason (antivirus, disk space,
   etc.), `install.bat` still leaves you with a working setup --
   `run_gui.bat` runs the app from source the same way, and
   `Telegram Digest.vbs` automatically falls back to it if no `.exe` is
   present.

Keep `TelegramDigest.exe` (or `run_gui.bat`) in this same folder --
`.env`, `config.yaml`, your Telegram session, and the local database all
live next to it and need to stay together.

The manual steps below are what `install.bat` automates, useful if you
want to understand what's happening or set it up by hand (e.g. on
Termux/Android, see further down).

## Setup

1. Get Telegram API credentials at https://my.telegram.org (API development
   tools -> create an app) to obtain `api_id` and `api_hash`.

2. Create a virtualenv:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # Termux/Linux/macOS
   ```

3. Install `torch` first, choosing the build for your platform, **then**
   the rest of the requirements:

   **Windows with an NVIDIA GPU (recommended, much faster embeddings):**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
   Use `cu121` for CUDA 12.1, or check
   https://pytorch.org/get-started/locally/ for the index URL matching
   your NVIDIA driver. You need a recent NVIDIA driver; the CUDA toolkit
   itself does NOT need to be installed separately, the wheel bundles
   what it needs.

   **Windows without a GPU, or Termux/Android:**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
   On Termux, installing `torch` can be slow/fragile depending on the
   device; see `agents.md`'s Phase 3 notes once that phase lands. Plain
   `pip install -r requirements.txt` without installing `torch` first
   also works and defaults to a CPU-only build.

   Either way, `embedding_device: "auto"` in `config.yaml` (the default)
   picks CUDA automatically when it's available and falls back to CPU
   otherwise -- no code changes needed when moving between your PC and
   your phone.

4. Copy the example files and fill them in:
   ```bash
   copy .env.example .env          # Windows
   # cp .env.example .env          # Termux/Linux/macOS
   copy config.example.yaml config.yaml
   ```
   Edit `.env` with your `TG_API_ID` / `TG_API_HASH` / `TG_PHONE`.
   Edit `config.yaml`: write your own `interest_examples` (5-20 short texts
   describing what you want to see), adjust `similarity_threshold`
   (0.45-0.55 is a reasonable range), optionally list specific `channels`
   to watch (leave empty to watch everything), and leave
   `embedding_device: "auto"` unless you want to force `"cuda"` or `"cpu"`.

5. First run downloads the embedding model
   (`paraphrase-multilingual-MiniLM-L12-v2`) once; after that everything
   runs fully offline.

## Run (CLI MVP)

```bash
python main.py
```

On first run you'll be prompted in the console for your Telegram login
code (and 2FA password if enabled) — this happens once; a local session
file is saved afterwards.

The console log line "CUDA GPU detected (...); using GPU for embeddings."
confirms the GPU path is active on Windows; if you don't see it, either
your torch build is CPU-only (reinstall per step 3 above) or no
CUDA-capable GPU/driver was found.

Use `--limit N` to control how many messages back are scanned per channel
per run (default 200) — useful the first time you run it against a channel
with lots of history.

## Прокси (Россия и другие места, где Telegram заблокирован)

Если Telegram у тебя открывается только через VPN/прокси, учти: этот
инструмент подключается к Telegram напрямую через Telethon, поэтому ему
тоже нужен доступ в обход блокировки.

- **Если у тебя системный VPN** (весь трафик компьютера идёт через
  VPN-туннель, не просто прокси в браузере) -- скорее всего, ничего
  делать не нужно: Windows и Telethon и так пойдут через VPN.
- **Если у тебя SOCKS5/HTTP-прокси**, настроенный в Windows (Параметры ->
  Сеть и Интернет -> Прокси, или его настроил VPN/прокси-клиент) --
  приложение подхватит его **автоматически**, ничего вписывать не нужно.
  В консоли при запуске появится строка `Auto-detected system proxy (...)`.
- **Если автоопределение не сработало** (прокси не зарегистрирован как
  системный прокси Windows, либо ты на Termux/Linux/macOS) -- впиши
  вручную в `.env`: `TG_PROXY_TYPE`, `TG_PROXY_HOST`, `TG_PROXY_PORT`
  (и `TG_PROXY_USERNAME`/`TG_PROXY_PASSWORD`, если нужны). Используй те
  же данные, что в настройках прокси Telegram Desktop (Настройки ->
  Данные и память -> Прокси). MTProto-прокси (ссылки вида
  `tg://proxy?...`) не поддерживаются -- только обычный SOCKS5/SOCKS4/HTTP.

## Run (Windows GUI)

```bash
python gui_main.py
```

Opens a window with two tabs:
- **Лента** (Feed) — matching posts as they're found, sorted newest-first,
  each with score, channel, time, preview, and an "Open in Telegram" button.
  "Проверить сейчас" (Check now) triggers an immediate check outside the
  regular schedule.
- **Настройки** (Settings) — edit interest examples, similarity threshold,
  check interval, and per-channel watch toggles (populated after the first
  check discovers your channels); "Сохранить настройки" (Save) writes
  `config.yaml` and hot-reloads the filter in the background, no restart
  needed.

The fetch+filter pipeline runs on a background thread (`gui/worker.py`) on
your configured `check_interval_minutes`, so the window stays responsive.
Closing the window minimizes to the system tray (if `pystray`+`Pillow` are
installed) instead of quitting; use the tray icon's "Выход" (Quit) to
actually exit, or "Проверить сейчас" from the tray menu. New matching posts
also raise a Windows toast notification.

## Run (Termux / Android)

The exact same core (`config.py`, `telegram_client.py`, `embed_filter.py`,
`storage.py`, `pipeline.py`) runs unchanged on Android via
[Termux](https://termux.dev/) -- only the entry point differs.

### 1. Install Termux + Termux:API

Install both **Termux** and **Termux:API** from F-Droid (not the outdated
Play Store builds). Then in Termux:

```bash
pkg update && pkg install python git termux-api
```

`termux-api` is the command-line side of the Termux:API app; together
they're what `notify.py` uses for notifications (`termux-notification`)
and for tap-to-open links (`termux-open-url`).

### 2. Get the project onto the phone and install dependencies

```bash
git clone <your repo, or transfer the folder some other way>
cd TelegrammSumator
python -m venv .venv
source .venv/bin/activate
```

**Torch on Termux -- read this first.** Termux uses Android's C library
(bionic), not glibc, so the normal PyPI `torch` wheels often don't install
directly. You have two reasonable paths:

- **Option A -- full local AI filtering (recommended):** run a proper
  Linux userspace on the phone via
  [`proot-distro`](https://github.com/termux/proot-distro) (e.g. Ubuntu),
  where normal aarch64 pip wheels for `torch`/`sentence-transformers`
  install the same way they do on a Linux server:
  ```bash
  pkg install proot-distro
  proot-distro install ubuntu
  proot-distro login ubuntu
  # inside the Ubuntu shell:
  apt update && apt install -y python3 python3-pip python3-venv
  cd /path/to/TelegrammSumator   # bind-mount or copy the project in
  python3 -m venv .venv && source .venv/bin/activate
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
  ```
  Note `termux-notification`/`termux-open-url` are Termux binaries, not
  visible inside the proot Ubuntu shell by default -- either call them via
  `termux-chroot`-style PATH access if your proot-distro setup exposes it,
  or run the AI-heavy check from inside Ubuntu and trigger notifications
  from a small wrapper script back in plain Termux.

- **Option B -- lightweight, no GPU/embedding model at all:** just run in
  plain Termux and skip installing `torch`/`sentence-transformers`
  entirely (or let their install fail). `embed_filter.py`'s
  `build_filter()` automatically falls back to the keyword filter
  (`KeywordFilter`) whenever the embedding model can't be loaded -- the
  app still runs and still filters, just with simpler keyword matching
  instead of semantic similarity. This is the quickest way to get
  notifications working on a phone.
  ```bash
  pip install telethon PyYAML python-dotenv
  ```

Either way, `embedding_device` in `config.yaml` should stay `"auto"` (or
be set to `"cpu"` explicitly) on a phone -- there's no CUDA on Android.

### 3. Configure and run

Same as the Windows setup: copy `.env.example` -> `.env` and
`config.example.yaml` -> `config.yaml`, fill in your Telegram credentials
and interest examples (see the Setup section above).

```bash
python termux_main.py            # loops forever, checking on your
                                  # configured interval, notifying via
                                  # termux-notification (tap a
                                  # notification to open the post)
python termux_main.py --once     # single check, then exit -- use this
                                  # with cron/termux-job-scheduler instead
                                  # of a long-running loop
```

### 4. Keep it running in the background

Android aggressively kills background processes. To keep `termux_main.py`
alive:

- Run `termux-wake-lock` once per session (prevents Doze from suspending
  Termux); `termux-wake-unlock` to release it.
- Install the **Termux:Boot** app (F-Droid) and drop a start script in
  `~/.termux/boot/` to auto-launch the digest loop on phone boot.
- Alternatively, skip the long-lived loop and schedule
  `python termux_main.py --once` periodically with `cron` (via
  `pkg install cronie`) or Android's `termux-job-scheduler`, which is
  gentler on battery than a permanent background loop.

## Project layout

- `config.py` — loads `.env` secrets and `config.yaml` settings.
- `telegram_client.py` — Telethon login, channel discovery, incremental
  message fetch, `Post` model.
- `embed_filter.py` — local embedding + cosine similarity scoring
  (CPU or CUDA, auto-detected), with a keyword-based fallback filter.
- `storage.py` — SQLite storage: channels, last-seen ids, shown posts,
  settings.
- `pipeline.py` — shared fetch+filter pass (`check_once`), used by both
  the CLI and the GUI so they never drift apart.
- `notify.py` — cross-platform notification dispatcher (termux-notification
  on Android, plyer/win10toast on Windows), shared by the GUI, CLI, and
  Termux entry points.
- `main.py` — CLI entry point (Windows/desktop).
- `termux_main.py` — Termux/Android entry point (loop or `--once`).
- `gui_main.py` — Windows GUI entry point.
- `install.bat` — one-click Windows setup: venv, deps, `.exe` build.
- `run_gui.bat` / `run_cli.bat` — launch from source (fallback if no `.exe`).
- `Telegram Digest.vbs` — console-free double-click launcher.
- `gui/app.py` — CustomTkinter window: feed tab + settings tab.
- `gui/worker.py` — background thread running the pipeline on a schedule.
- `gui/tray.py` — system tray icon (pystray), optional.
- `gui/notifications.py` — desktop toast notifications, optional.

## Roadmap

See `agents.md` for the full phased plan (SQLite -> Windows GUI/tray ->
Termux port -> optional tiny local generative model for summaries).

## Non-goals

No cloud LLM calls, no reading private chats, no automatic posting/replies.
