---
name: digest-topic-suggestions
description: Use when changing or debugging the "✨ Предложить темы (ИИ)" button in Settings (on-demand topic suggestions with a review/save dialog), or the per-post "Этот пост → тема…" / "Весь канал → тема…" menus. Holds the flow, the files involved and known limits (2026-10-19).
---

# On-demand topic suggestions (2026-10-19)

Same algorithm as auto topics (`auto_topics.discover_topics`, see skill `digest-auto-topics`), but **nothing is saved by itself**: the user reviews the result first.

## Flow
1. Settings tab: button `✨ Предложить темы (ИИ)` (`SettingsTab._suggest_topics`, `gui/app.py`) disables itself and calls `worker.suggest_topics()`.
2. `DigestWorker.suggest_topics()` (`gui/worker.py`) sets `_suggest_event` + `_check_now_event`. The check loop then runs `check_once` with `unplaced_out`/`context_out` filled even if the `auto_topics` checkbox is off.
3. `_auto_create_topics(..., suggest_only=True)` runs `discover_topics` and puts a `topic_suggestions` event (`suggestions` = list of `auto_topics.NewTopic`; empty list = nothing found) instead of writing `config.yaml`.
4. `App._handle_worker_event` → `SettingsTab.show_suggestions(found)`: dialog, one row per topic (checkbox, editable name, editable comma-separated tags, "постов: N · новая тема / дополнит «X»"). `merge_into` topics have a locked name and only append tags.
5. "Сохранить выбранные" adds rows to the Settings topic list (or appends missing tags to an existing row), then calls `_on_save()` → config saved, filter reloaded, feed regrouped. "Закрыть" changes nothing.

## Per-post menus (same file, post card footer)
- `Этот пост → тема…` (`_on_move`): moves one post, counts as feedback example.
- `Весь канал → тема…` (`_on_channel`): all posts of the channel move, topic stored as the channel's `manual` home topic. Not the same as a topic (a topic = keywords in config; channel topic = which topic a channel belongs to). Suggestions never change channel topics.

## Known limits
- Only posts that are *new in this scan* are considered. With `only_unread: false` right after a scan there may be none → "Новых тем не нашлось". A fix would be scanning recent posts regardless of `last_seen`.
- Needs the filter to have `embed_texts` (not the keyword fallback) and ≥ `auto_topics_min_posts` unplaced posts.
- Not tried by clicking in the live GUI yet (no GUI-control tool in the session); only syntax-checked and launched without errors.
- Desktop only; not ported to Android/Termux.
