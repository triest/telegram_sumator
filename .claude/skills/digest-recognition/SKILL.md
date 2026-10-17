---
name: digest-recognition
description: Use when changing or tuning how Telegram Digest recognises/classifies posts and pictures (embedding model, thresholds, topics, auto-topics, channel "home" topic, feedback buttons, CLIP images), or when posts land in the wrong topic. Gives the architecture, measured results, and how to benchmark on real posts.
---

# Telegram Digest: post & picture recognition

All AI is local and in-process (no Ollama, no cloud). Windows desktop first.

## Pipeline (files)
- `embed_filter.py` — `EmbeddingFilter` (Qwen3-Embedding-0.6B, instruction prefix on posts; a topic = average of two views: bare name and "name: keywords"; exact keyword hits ADD `kw_bonus` 0.12 per hit (max 2) to the semantic score), `TopicKeywordBoost` (exact keyword stems, passes hit counts to the embedding filter), `LLMTopicVerifier` (disabled: `llm_classify: false`).
- `pipeline.py` — `check_once`: fetch → score → "home topic" fallback. A channel's home topic = the topic most of its recent posts fall into (share ≥ `HOME_MIN_SHARE` 0.30 and ≥ 1.5× the runner-up), cached in `channel_profile` (signature includes model + `#v2`); `NEUTRAL_PROBES` baseline is subtracted from the similarity prior. Posts of channels without a clear home stay in "Прочее" (no "closest topic" forcing).
- `image_topics.py` — CLIP (`clip-ViT-B-32` + multilingual text encoder). Used for picture-only / caption < 60 chars posts; distractor classes (food, cats, scenery, screenshots…) prevent forcing.
- `auto_topics.py` + `DigestWorker._auto_create_topics` — clusters posts no topic fit; group ≥ `auto_topics_min_posts` → new topic. Name via `llm.suggest_topic_name` (embedded GGUF), keywords from the group.
- Feedback: GUI buttons 👍/👎 and "move to topic" → `storage.add_feedback`; `EmbeddingFilter.set_feedback` applies verdicts to near-identical posts (cosine ≥ `FEEDBACK_SIM` 0.85). Source label `вручную` beats keywords.
- Config: `embedding_model`, `similarity_threshold` (0.62 for Qwen3; ~0.38 for MiniLM; keyword bonus scale depends on it), `auto_topics`, `auto_topics_min_posts`, `llm_classify`.

## Measured results (114 real posts from the user's channels, labeled by hand)
| setup | correct |
|---|---|
| MiniLM, thr 0.75 (old config) | 62 |
| MiniLM, thr 0.38 | 81 |
| **Qwen3-Embedding-0.6B + instruction + name-only topics, thr 0.62** | **96** (5 FP, 13 miss, 0 wrong topic) |
| Qwen3-4B GGUF as classifier (CPU, ~13 s/post) | 70 |
Qwen3-Embedding-4B ≈ 0.6B (0.86 vs 0.85) but 7× heavier → not used. bge-m3 needs torch ≥ 2.6; EmbeddingGemma is gated.
Pictures: CLIP name+6 keywords+distractors, prob ≥ 0.35 → 38/49, 0 false positives.

## Regression found and fixed (2nd round, user report: "misses Ukraine posts, mistakes with кино")
- Raising the threshold to 0.62 left the old keyword score (0.5 per hit) below it → topic СВО got **0 posts**. Keyword hits now add to the semantic score (bonus), they no longer have their own scale.
- Name-only topic text failed for acronyms ("СВО"): the topic is now avg(name, "name: keywords").
- Tried and rejected: LLM-written topic descriptions (3B model just echoes the keywords), prototypes from keyword-matched posts (floods false positives), topic-as-query orientation (set114 max 88 vs 98), per-topic baselines (no gain).
- Real fix for vague topics is the user's keyword list: added Украина/Зеленский/Киев/ВС РФ/обстрел/дрон/… to СВО and сериал/трейлер/режиссёр/… to кино in `config.yaml`.
- Live check on the real unread feed (133 posts, `live.py` pattern: `check_once` + dump): СВО 36→55 posts after switching the home topic to the posts-share rule (picture-only posts of war channels now follow the channel topic); 'Прочее' 57→43. Known weak spot: CLIP sends some cosplay/humor pictures to 'история'.
- Second benchmark (290 posts, weak labels by channel + regex, noisy): СВО war posts 25/32 (was 0), кино 73/79, set114 91/114 (96 before the bonus; the extra false positives come from keyword hits).

## How to benchmark (do this before changing models/thresholds)
1. Fetch real posts read-only with `TelegramDigestClient` (broadcast channels only, ~8 posts × 40 channels) into a JSON in the scratchpad.
2. Label by index: `{idx: {topics}}`, `None` = should match no topic. Print previews to a UTF-8 file and read it with the Read tool.
3. Score with `build_filter(cfg.all_examples(), threshold, device, model_name, topics)` and count correct / false-pos / missed / wrong-topic. Sweep the threshold; thresholds are model-specific.
4. Pictures: save thumbnails, build numbered contact sheets with PIL and label by looking at them.

## Pitfalls seen
- Short generic topic names ("кино") are similarity attractors → subtract the neutral baseline; never fall back to "closest topic".
- Channel profile cache (`channel_profile`) must include the model name in its signature.
- Windows console is cp1251: run Python with `PYTHONIOENCODING=utf-8`, write results to files and Read them; emoji in titles crash `print`.
- Writing Python files via shell heredocs eats backslashes (`\n` became a real newline) → use the Edit tool for strings with escapes.
- `llama-cpp-python` installs from a prebuilt wheel (`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`); GGUF download can fail on rename if antivirus holds the `.part` file.
- Run from source with `run_source.bat`; `TelegramDigest.exe` / `run_gui.bat` use the old `dist` build until rebuilt with PyInstaller.
- Not ported yet: Termux (`termux_main.py` only gets `embedding_model`) and the Android app (`android/.../core.py`).
