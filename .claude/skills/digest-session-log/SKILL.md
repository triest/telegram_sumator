---
name: digest-session-log
description: Use only when asked what was changed or decided in the 2026-10-17 recognition-tuning session and why, or to resume its open to-do list. Log of requests, decisions, dead ends. Not needed for ordinary coding tasks.
---

# Session log: recognition tuning (2026-10-17)

Commit `f55de1c` on `master` (pushed). Details of the architecture are in the `digest-recognition` skill; this file is the history and the owner's preferences.

## Requests, in order
1. "Improve post recognition, maybe new local models appeared. Desktop first."
2. "Take real posts from the connected Telegram" → fetched read-only, labeled by hand (114 posts, 49 pictures).
3. "No Ollama, everything embedded in the program."
4. "Come up with something for the interface" → 👍 Верно / 👎 Не то buttons on cards.
5. "Implement the best option; check the result afterwards."
6. "Create topics yourself if none fits" → `auto_topics` mode (Settings checkbox, off by default).
7. "Too much in Кино (sex posts), pictures must also go through a neural net" → baseline + clear-margin home topic, CLIP for pictures.
8. "Check, commit and push", "make a skill and CLAUDE.md", then this log.

## Decisions and why
- Embeddings: Qwen3-Embedding-0.6B + instruction + name-only topics: 96/114 vs 62 (old config). 4B model gives no real gain.
- LLM as classifier rejected: Qwen3-4B GGUF 70/114, ~13 s/post on CPU; Ollama models were worse and are out of scope anyway. LLM kept only for summaries and naming auto topics.
- Feedback had to be taught to the embedding filter, otherwise the new buttons would have been inert (only the disabled LLM verifier read it).
- "Кино" problem: generic one-word topics are attractors; fixed with a neutral-probe baseline and a margin gate; no "closest topic" fallback.
- Pictures: CLIP (not a VLM) because it is small, in-process and understands Russian topic names via the multilingual text encoder.

## Dead ends (do not repeat)
- Lowering the cluster threshold (≤0.5) merges unrelated posts; a cohesion check did not help (post embeddings are all fairly close).
- bge-m3 (needs torch ≥ 2.6), EmbeddingGemma (gated on HuggingFace), multilingual-e5-large (worse than MiniLM on real posts).
- Qwen3-VL via Ollama returned nothing in JSON mode; Qwen2.5-VL mislabeled most posts.

## Environment notes
- RTX 4060 Ti 16 GB shared with ComfyUI; torch 2.5.1+cu121; `.venv` has llama-cpp-python 0.3.36 (CPU wheel) and the GGUFs in `models/` (git-ignored).
- A stale empty `.git/index.lock` (from Oct 6) was removed to allow committing.
- Console is cp1251: use `PYTHONIOENCODING=utf-8`, write outputs to files.

## Not done
See "Open work" in `claude.md`.
