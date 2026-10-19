---
name: digest-auto-topics
description: Use when changing or debugging auto topics (new topics created from posts no topic fit, merging into existing topics, spam/repost filtering, keyword extraction), their Android port in core.py, or when checking auto topics on the live feed. Holds the algorithm, constants, live-check recipe and known weak spots (2026-10-19 session).
---

# Auto topics (2026-10-19 session, commit `32efffd`)

Desktop: `auto_topics.py` + `DigestWorker._auto_create_topics` (`gui/worker.py`). Android: port in `android/app/src/main/python/core.py` (`_discover`, `suggestions()`), applied by `applyTopicSuggestions` in `MainActivity.kt`. Architecture of the rest is in `digest-recognition`.

## Algorithm (`discover_topics`)
1. `clean()` each unplaced post: drop markdown links *with labels*, URLs, `t.me/…`, `@mentions`, `*`/`_` (channel signatures are not content).
2. Keep posts ≥ 40 chars, embed, `dedupe` (cosine ≥ `DUPLICATE_SIM` 0.95 = same repost, counts once) — a template can't pass for a topic.
3. Greedy `cluster` (`CLUSTER_SIM` 0.58), groups ≥ `auto_topics_min_posts`, mean cohesion ≥ `MIN_COHESION` 0.66.
4. Keywords: shared inside the group, rare outside; words found in the first/last 40 chars of ≥70% of members (`template_words`) are dropped; keep **5**. Fewer than 2 left → it's an ad series, no topic.
5. Group centre vs existing topic vectors ≥ `EXISTING_TOPIC_SIM` 0.62 → **merge**: `NewTopic.merge_into` set, up to 4 keywords appended to that topic (not counted in `max_new`). Merging only when topic vectors map 1:1 to topics (`len(existing_vectors) == len(existing_names)`); otherwise old behaviour (skip).
6. Otherwise a new topic: name from the embedded GGUF (`llm.suggest_topic_name`), fallback = top two keywords.

## Live check recipe
Read-only, temp DB so `last_seen` and `config.yaml` stay untouched: build the filter, `TelegramDigestClient`, `check_once(..., limit_per_channel=8, unplaced_out=un)`, then `discover_topics(un, f.embed_texts, names, None, min_posts, 2, f.topic_name_vectors())`. Count topics by `result.matched_example.split(":")[0]`. Use `.venv\Scripts\python.exe` (system Python lacks dotenv), `PYTHONIOENCODING=utf-8`, run in background (~8 min with ComfyUI on the GPU), `faulthandler.dump_traceback_later` if it hangs. The feed changes between runs, so topic counts are NOT comparable run to run.

## Findings on the real feed (~370 posts, 85–90 unplaced)
- Raw version turned reposted ads (cigarettes, one template in many channels) and a channel-signature series ("эксклюзив") into topics. Dedupe removed the first, link/signature stripping + template words the second. Remaining legit group: AliExpress deals (name from LLM; fallback "Ценой / Продавца" is weak).
- Merge never triggered on live data — only verified on synthetic vectors. If it should be tested live, pick posts of a topic that missed them.

## Android port
Same logic in numpy, constants **looser and untuned** (MiniLM similarities are lower): cluster 0.50, existing 0.55, cohesion 0.58. Scan returns the same feed JSON; the app then calls `core.suggestions()` and adds/extends topics (effective on the next scan). No toggle in the UI (`auto_topics` setting defaults to true, Kotlin doesn't send it). Kotlin was **not compiled**: the machine has no JDK/Android SDK (searched all drives); old `TelegramDigest.apk` (Oct 6) predates all of this. Build needs JDK 17, SDK platform 34, `android/local.properties` with `sdk.dir`; how the earlier APK was built is unknown (not documented in repo).

## Pitfalls
- Words at the start/end of every post of a group look like a template even when synthetic tests put them there — write test texts with varied openings/endings.
- Synthetic vectors: noise σ≈0.07 per dim at 64 dims gives pairwise cos ≈0.8; tighter noise gets deduped.
- Only the 5-keyword cap and merge keywords come from the group; keywords are still weak/generic sometimes (`подсветка`, `взять`).

## Open
- Verify merge on live data; tune Android thresholds on real phone posts.
- Toggle for auto topics in the Android settings; port to `termux_main.py`.
- Build the APK once JDK/SDK exist.
