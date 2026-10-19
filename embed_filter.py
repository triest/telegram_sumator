"""Local embedding-based relevance filter.

Primary path: sentence-transformers (multilingual MiniLM) + cosine similarity
against user-provided "interest examples". Fully local/offline after the
model has been downloaded once (cached by sentence-transformers/huggingface).

Device selection:
    - "auto" (default): use CUDA if a GPU + CUDA-enabled torch are
      available (typical on a Windows PC), otherwise fall back to CPU
      (typical on Android/Termux, where there's no CUDA at all).
    - "cuda": force GPU, error out if unavailable.
    - "cpu": force CPU (what Termux always effectively gets, since it
      has no CUDA).

This lets the exact same code run fast on a Windows PC with an NVIDIA GPU
and still run correctly, just slower, on a phone.

If the model can't be loaded (not downloaded yet, no disk space, etc.) this
falls back to a simple keyword/regex filter so the app still works.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
LEGACY_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Qwen3-Embedding is instruction-aware: posts are encoded with a task prefix and
# topics are compared by their *name* (the keyword list dilutes the vector; the
# keywords are matched exactly by TopicKeywordBoost instead). On real posts from
# the user's channels this raised overall accuracy 0.70 -> 0.85 (AUC 0.68 -> 0.91)
# compared with paraphrase-multilingual-MiniLM.
QWEN_QUERY_PREFIX = (
    "Instruct: Определи, к какой теме относится пост из Telegram-канала\nQuery: "
)


def _is_instruct_model(model_name: str) -> bool:
    return "qwen3-embedding" in model_name.lower()


@dataclass
class FilterResult:
    is_relevant: bool
    score: float
    matched_example: Optional[str] = None
    # Which engine produced this verdict: "смысл" (embedding model),
    # "слова" (topic keywords), "ИИ" (local LLM confirmed), "ИИ-нет" (LLM rejected).
    source: str = ""


class PostLike(Protocol):
    text: str


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between every vector in a and every in b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def resolve_device(requested: str = "auto") -> str:
    """Resolve "auto"/"cuda"/"cpu" to a concrete torch device string.

    "auto" picks CUDA when a GPU and CUDA-enabled torch are present (PC),
    and silently falls back to CPU otherwise (e.g. Termux/Android, or a
    PC without an NVIDIA GPU / CUDA torch build installed).
    """
    requested = (requested or "auto").lower()

    if requested == "cpu":
        return "cpu"

    try:
        logger.info("Загрузка PyTorch…")
        import torch

        logger.info("PyTorch загружен, проверяю GPU…")
    except ImportError:
        if requested == "cuda":
            raise RuntimeError("torch is not installed; cannot use device='cuda'.")
        return "cpu"

    cuda_available = torch.cuda.is_available()

    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "device='cuda' was requested but no CUDA GPU is available. "
                "Make sure you installed a CUDA-enabled build of torch "
                "(see requirements.txt / README) and that NVIDIA drivers "
                "are installed."
            )
        return "cuda"

    # requested == "auto"
    if cuda_available:
        logger.info("CUDA GPU detected (%s); using GPU for embeddings.", torch.cuda.get_device_name(0))
        return "cuda"
    return "cpu"


class EmbeddingFilter:
    """Embeds interest examples once, then scores posts by max similarity."""

    def __init__(
        self,
        interest_examples: list[str],
        threshold: float = 0.48,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
    ) -> None:
        if not interest_examples:
            raise ValueError("interest_examples must not be empty")
        self.interest_examples = interest_examples
        self.threshold = threshold
        self.model_name = model_name
        self.feedback: list[tuple[str, str]] = []
        self._fb_vecs = None
        self._instruct = _is_instruct_model(model_name)
        self._query_prefix = QWEN_QUERY_PREFIX if self._instruct else ""

        from sentence_transformers import SentenceTransformer  # local import: heavy

        self.device = resolve_device(device)
        logger.info("Loading embedding model '%s' on device '%s'.", model_name, self.device)
        # Model is cached after the first run: load it OFFLINE first. Online
        # loading does dozens of HTTP checks against huggingface.co (slow,
        # especially through a VPN). Fall back to online only if not cached.
        try:
            self._model = SentenceTransformer(
                model_name, device=self.device, local_files_only=True
            )
        except Exception:
            logger.info("Model not cached yet; downloading '%s'.", model_name)
            self._model = SentenceTransformer(model_name, device=self.device)

        # Larger batch size pays off on GPU; harmless (unused headroom) on CPU.
        encode_batch_size = (32 if self._instruct else 64) if self.device == "cuda" else 8
        self._encode_batch_size = encode_batch_size
        if self._instruct:
            self._model.max_seq_length = 512  # bounds memory; posts are cut to ~1000 chars anyway

        # Instruct models: a topic is described by TWO views averaged: its bare name and
        # "name: keywords". The name alone fails for acronyms ("СВО") and keywords alone
        # drag in everything; the average keeps the user's keywords meaningful.
        name_vecs = self._model.encode(
            [self._doc_text(e) for e in interest_examples],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=encode_batch_size,
        )
        if self._instruct:
            full_vecs = self._model.encode(
                list(interest_examples),
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=encode_batch_size,
            )
            both = name_vecs + full_vecs
            self._example_embeddings = both / (np.linalg.norm(both, axis=1, keepdims=True) + 1e-12)
        else:
            self._example_embeddings = self._model.encode(
                list(interest_examples),
                convert_to_numpy=True,
                normalize_embeddings=False,
                batch_size=encode_batch_size,
            )

    # A post this similar to one the user already judged gets the same verdict.
    FEEDBACK_SIM = 0.85
    supports_hits = True
    kw_bonus = 0.12  # added per exact keyword hit (max 2 hits count)

    def set_feedback(self, feedback) -> None:
        """Manual 👍/👎/moves: [(post text, topic or "" = not interesting)]."""
        self.feedback = list(feedback or [])
        self._fb_vecs = None
        if not self.feedback:
            return
        docs = [" ".join(t.split())[:1000] for t, _ in self.feedback]
        vecs = self._model.encode(
            docs, convert_to_numpy=True, normalize_embeddings=True, batch_size=self._encode_batch_size
        )
        self._fb_vecs = vecs
        self._fb_example = []
        for _, topic in self.feedback:
            self._fb_example.append(
                next((e for e in self.interest_examples if self._doc_text(e) == topic), None) if topic else None
            )

    def _apply_feedback(self, texts: list[str], results: list[FilterResult]) -> None:
        if getattr(self, "_fb_vecs", None) is None or not texts:
            return
        vecs = self._model.encode(
            [" ".join(t.split())[:1000] if t.strip() else " " for t in texts],
            convert_to_numpy=True, normalize_embeddings=True, batch_size=self._encode_batch_size,
        )
        sims = vecs @ self._fb_vecs.T
        for i, row in enumerate(sims):
            j = int(np.argmax(row))
            if row[j] < self.FEEDBACK_SIM or not texts[i].strip():
                continue
            topic = self.feedback[j][1]
            if topic and self._fb_example[j]:
                results[i] = FilterResult(True, max(results[i].score, 0.95), self._fb_example[j], "вручную")
            elif not topic:
                results[i] = FilterResult(False, 0.0, results[i].matched_example, "вручную")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """L2-normalised post vectors for post-to-post similarity (no task prefix)."""
        return self._model.encode(
            [" ".join(t.split())[:1000] or " " for t in texts],
            convert_to_numpy=True, normalize_embeddings=True, batch_size=self._encode_batch_size,
        )

    def topic_name_vectors(self) -> np.ndarray:
        """Normalised embeddings of the topic texts (same space as embed_texts)."""
        e = self._example_embeddings
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)

    def _doc_text(self, example: str) -> str:
        """Topic text to embed: for instruct models "Name: kw, kw" -> "Name"."""
        if self._instruct and ": " in example:
            return example.split(": ", 1)[0].strip()
        return example

    def _encode_posts(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            [self._query_prefix + " ".join(t.split())[:1000] for t in texts],
            convert_to_numpy=True,
            normalize_embeddings=False,
            batch_size=self._encode_batch_size,
        )

    def score(self, text: str) -> FilterResult:
        if not text.strip():
            return FilterResult(is_relevant=False, score=0.0)

        embedding = self._encode_posts([text])
        sims = _cosine_similarity_matrix(embedding, self._example_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        return FilterResult(
            is_relevant=best_score >= self.threshold,
            score=best_score,
            matched_example=self.interest_examples[best_idx],
        )

    def describe(self) -> str:
        return f"смысловая модель {self.model_name} ({self.device})"

    def topic_matrix(self, texts: list[str]) -> np.ndarray:
        """Raw similarity of each text to every topic/example (n x T)."""
        if not texts:
            return np.zeros((0, len(self.interest_examples)))
        emb = self._encode_posts([t if t.strip() else " " for t in texts])
        return _cosine_similarity_matrix(emb, self._example_embeddings)

    # A picture this close (CLIP cosine) to one the user already judged gets the same verdict.
    IMAGE_FEEDBACK_SIM = 0.92

    def set_image_feedback(self, feedback) -> None:
        """Manual verdicts for pictures: [(thumbnail path, topic or "" = not interesting)]."""
        feedback = list(feedback or [])
        if feedback == getattr(self, "_img_fb_src", None):
            return
        self._img_fb_src = feedback
        self._img_fb_vecs, self._img_fb_topics = None, []
        if not feedback:
            return
        if getattr(self, "_image_clf", None) is None:
            from image_topics import ImageTopicClassifier

            self._image_clf = ImageTopicClassifier(self.interest_examples, self.device)
        vecs = self._image_clf.encode([p for p, _ in feedback])
        keep = [(v, t) for v, (_, t) in zip(vecs, feedback) if v is not None]
        if keep:
            self._img_fb_vecs = np.stack([v for v, _ in keep])
            self._img_fb_topics = [t for _, t in keep]

    def _apply_image_feedback(self, texts: list[str], images, results: list[FilterResult]) -> None:
        if getattr(self, "_img_fb_vecs", None) is None:
            return
        todo = [i for i, im in enumerate(images) if im and len(texts[i].strip()) < 60]
        if not todo:
            return
        for i, v in zip(todo, self._image_clf.encode([images[i] for i in todo])):
            if v is None:
                continue
            sims = self._img_fb_vecs @ v
            j = int(np.argmax(sims))
            if sims[j] < self.IMAGE_FEEDBACK_SIM:
                continue
            topic = self._img_fb_topics[j]
            example = next((e for e in self.interest_examples if self._doc_text(e) == topic), None) if topic else None
            if topic and example:
                results[i] = FilterResult(True, 0.95, example, "вручную")
            elif not topic:
                results[i] = FilterResult(False, 0.0, results[i].matched_example, "вручную")

    def _apply_images(self, texts: list[str], images, results: list[FilterResult]) -> None:
        """Picture-only / short-caption posts: the local picture model picks the topic."""
        todo = [
            i for i, (t, im) in enumerate(zip(texts, images))
            if im and len(t.strip()) < 60 and not results[i].is_relevant and results[i].source != "вручную"
        ]
        if not todo:
            return
        if getattr(self, "_image_clf", None) is None:
            from image_topics import ImageTopicClassifier

            self._image_clf = ImageTopicClassifier(self.interest_examples, self.device)
        for i, (j, prob) in zip(todo, self._image_clf.classify([images[i] for i in todo])):
            if j is not None:
                results[i] = FilterResult(
                    True, min(0.95, 0.6 + 0.35 * prob), self.interest_examples[j], "картинка"
                )

    def score_batch(self, texts: list[str], prior=None, **_kw) -> list[FilterResult]:
        """Batch-score multiple posts at once -- much faster on GPU.

        `prior`: optional per-topic bias (length = number of examples),
        derived from what the channel usually posts about; added to the
        similarities so ambiguous posts lean toward the channel's topics.
        """
        if not texts:
            return []

        non_empty_idx = [i for i, t in enumerate(texts) if t.strip()]
        results: list[Optional[FilterResult]] = [
            FilterResult(is_relevant=False, score=0.0) for _ in texts
        ]
        images = _kw.get("images")
        use_images = bool(images) and getattr(self, "image_recognition", True)
        if not non_empty_idx:
            if use_images:  # picture-only posts: nothing to read, the picture decides
                self._apply_image_feedback(texts, images, results)  # type: ignore[arg-type]
                self._apply_images(texts, images, results)  # type: ignore[arg-type]
            return results  # type: ignore[return-value]

        embeddings = self._encode_posts([texts[i] for i in non_empty_idx])
        sims = _cosine_similarity_matrix(embeddings, self._example_embeddings)
        if prior is not None:
            sims = sims + np.asarray(prior, dtype=float)[None, :]
        hits = _kw.get("hits")
        if hits is not None:
            # exact keyword matches add to the semantic score (they do not replace it)
            h = np.asarray(hits, dtype=float)[non_empty_idx]
            sims = sims + self.kw_bonus * np.minimum(h, 2.0)
        for row, i in enumerate(non_empty_idx):
            best_idx = int(np.argmax(sims[row]))
            best_score = float(sims[row][best_idx])
            results[i] = FilterResult(
                is_relevant=best_score >= self.threshold,
                score=best_score,
                matched_example=self.interest_examples[best_idx],
            )
            results[i].source = "смысл"
        if use_images:
            self._apply_image_feedback(texts, images, results)  # type: ignore[arg-type]
            self._apply_images(texts, images, results)  # type: ignore[arg-type]
        self._apply_feedback(texts, results)  # type: ignore[arg-type]
        return results  # type: ignore[return-value]


class KeywordFilter:
    """Fallback filter used when the embedding model can't be loaded.

    Matches if any interest example's significant words appear in the post
    text (case-insensitive substring match). Much cruder than embeddings but
    keeps the tool usable offline / on constrained devices.
    """

    def __init__(self, interest_examples: list[str], min_hits: int = 1) -> None:
        self.interest_examples = interest_examples
        self.min_hits = min_hits
        self._keyword_sets = [
            {w for w in re.findall(r"\w+", ex.lower()) if len(w) > 3}
            for ex in interest_examples
        ]

    def score(self, text: str) -> FilterResult:
        text_lower = text.lower()
        text_words = set(re.findall(r"\w+", text_lower))

        best_hits = 0
        best_idx = -1
        for idx, keywords in enumerate(self._keyword_sets):
            hits = len(keywords & text_words)
            if hits > best_hits:
                best_hits = hits
                best_idx = idx

        # Pseudo-score in [0, 1] so downstream sorting/printing works the
        # same way as with the embedding filter.
        pseudo_score = min(1.0, best_hits / 5.0)
        return FilterResult(
            is_relevant=best_hits >= self.min_hits,
            score=pseudo_score,
            matched_example=(
                self.interest_examples[best_idx] if best_idx >= 0 else None
            ),
        )

    def score_batch(self, texts: list[str], **_kw) -> list[FilterResult]:
        out = [self.score(t) for t in texts]
        for r in out:
            r.source = "слова"
        return out

    def describe(self) -> str:
        return "только слова (модель не загружена)"


def contains_excluded_keyword(text: str, excluded: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in excluded if kw)


class TopicKeywordBoost:
    """Wraps a base filter and adds exact keyword matching per topic.

    The embedding model alone compares *meaning* of a post with a topic
    description, which is fuzzy for short keyword lists (a post that
    obviously mentions "самолёт" can score below the threshold). Here, if a
    post contains a topic's own keywords (Russian word endings tolerated),
    the topic gets a strong score, more hits -> higher. The final result is
    whichever is better for the post: embedding similarity or keyword hits.
    """

    def __init__(self, base, topics, threshold: float) -> None:
        self.base = base
        self.threshold = threshold
        self.interest_examples = getattr(base, "interest_examples", [])
        self._topics = []  # (example_text, [matchers])
        for t in topics:
            matchers = [m for m in (self._make_matcher(k) for k in t.keywords) if m]
            self._topics.append((t.as_example(), matchers))

    @staticmethod
    def _make_matcher(keyword: str):
        import re

        kw = keyword.strip().lower()
        if len(kw) < 2:
            return None
        words = re.findall(r"[\w\-]+", kw)
        pats = []
        for w in words:
            # long words: match the stem (tolerates endings: самолёт/самолёты/самолётов)
            stem = w[:-2] if len(w) >= 6 else (w[:-1] if len(w) == 5 else w)
            pats.append(re.compile(r"(?<![\w])" + re.escape(stem)))
        return pats

    def _hits(self, text_lower: str, matchers) -> int:
        return sum(1 for pats in matchers if all(p.search(text_lower) for p in pats))

    def score(self, text: str) -> FilterResult:
        return self.score_batch([text])[0]

    @property
    def feedback(self):
        return getattr(self.base, "feedback", [])

    def set_feedback(self, feedback) -> None:
        if hasattr(self.base, "set_feedback"):
            self.base.set_feedback(feedback)

    def set_image_feedback(self, feedback) -> None:
        if hasattr(self.base, "set_image_feedback"):
            self.base.set_image_feedback(feedback)

    def describe(self) -> str:
        return getattr(self.base, "describe", lambda: "?")() + " + слова тем"

    def topic_matrix(self, texts: list[str]):
        fn = getattr(self.base, "topic_matrix", None)
        return fn(texts) if fn else None

    def embed_texts(self, texts):
        return self.base.embed_texts(texts) if hasattr(self.base, "embed_texts") else None

    def topic_name_vectors(self):
        return self.base.topic_name_vectors() if hasattr(self.base, "topic_name_vectors") else None

    def _hit_matrix(self, texts: list[str], n_cols: int) -> np.ndarray:
        h = np.zeros((len(texts), n_cols))
        for i, t in enumerate(texts):
            low = t.lower()
            for k, (_, matchers) in enumerate(self._topics[:n_cols]):
                h[i, k] = self._hits(low, matchers)
        return h

    def score_batch(self, texts: list[str], **kw) -> list[FilterResult]:
        if getattr(self.base, "supports_hits", False):
            # the embedding filter adds keyword hits to its own semantic score
            n_cols = len(getattr(self.base, "interest_examples", self._topics))
            return self.base.score_batch(texts, hits=self._hit_matrix(texts, n_cols), **kw)
        results = self.base.score_batch(texts, **kw)
        out = []
        for text, res in zip(texts, results):
            if res.source == "вручную":  # the user's own verdict beats keywords
                out.append(res)
                continue
            low = text.lower()
            best_ex, best_hits = None, 0
            for example, matchers in self._topics:
                hits = self._hits(low, matchers)
                if hits > best_hits:
                    best_ex, best_hits = example, hits
            if best_hits:
                # exact keyword hits are strong evidence: one hit passes the threshold
                # (scale follows the threshold, it differs between embedding models)
                kw_score = min(self.threshold + 0.02 + 0.08 * (best_hits - 1), 0.95)
                if kw_score > res.score:
                    res = FilterResult(
                        is_relevant=kw_score >= self.threshold,
                        score=kw_score,
                        matched_example=best_ex,
                        source="слова",
                    )
            out.append(res)
        return out


class LLMTopicVerifier:
    """Second stage: a LOCAL LLM (Ollama) confirms/assigns the topic.

    Embeddings + keywords are fast but crude. Posts that look at least
    borderline-relevant are checked by the LLM, which reads the post and
    picks the right topic (or says none). That removes false hits (a word
    mentioned in passing, ads) and fixes wrong topic assignment. Results are
    cached per post text, so repeated checks don't re-ask the model. If the
    LLM errors, the fast filter's verdict is used unchanged.
    """

    def __init__(self, base, topics, threshold: float, model: str, url: str) -> None:
        self.base = base
        self.threshold = threshold
        self.model = model
        self.url = url
        self.interest_examples = getattr(base, "interest_examples", [])
        self._topics = [(t.name, list(t.keywords)) for t in topics]
        self._example_by_name = {t.name: t.as_example() for t in topics}
        self._borderline = max(0.2, threshold - 0.15)
        # True: the local LLM classifies EVERY post (embeddings/keywords only
        # serve as fallback if the LLM is unavailable).
        self.llm_primary = True
        self.force = True  # LLM must pick a topic (no "none" answer)
        self.feedback: list[tuple[str, str]] = []  # (text, topic) manual corrections
        self.vision_model = ""
        try:
            from llm import EMBEDDED, find_vision_model

            if model != EMBEDDED:  # the embedded model cannot look at pictures
                self.vision_model = find_vision_model(url)
        except Exception:
            pass
        if self.vision_model:
            logger.info("Анализ картинок включён: модель %s", self.vision_model)
            if self.vision_model != self.model:
                # Two different 7B models don't fit in VRAM together: Ollama would
                # unload/reload them on every alternating request (-> timeouts).
                # The vision model reads text just as well, so use ONE model.
                logger.info(
                    "Использую одну модель %s и для текста, и для картинок (без перезагрузок "
                    "между моделями)", self.vision_model,
                )
                self.model = self.vision_model
        else:
            logger.info(
                "Модель для картинок не найдена (ollama pull qwen2.5vl:7b) — посты без текста "
                "относятся по тематике канала"
            )
        self._cache: dict[int, Optional[str]] = {}
        self._failures = 0
        self.stats = {"calls": 0, "cached": 0, "accepted": 0, "rejected": 0}

    def describe(self) -> str:
        return (
            getattr(self.base, "describe", lambda: "?")()
            + f" + ИИ-проверка Ollama ({self.model})"
        )

    def pop_stats(self) -> dict:
        s, self.stats = self.stats, {"calls": 0, "cached": 0, "accepted": 0, "rejected": 0}
        return s

    def score(self, text: str) -> FilterResult:
        return self.score_batch([text])[0]

    def topic_matrix(self, texts: list[str]):
        fn = getattr(self.base, "topic_matrix", None)
        return fn(texts) if fn else None

    def embed_texts(self, texts):
        return self.base.embed_texts(texts) if hasattr(self.base, "embed_texts") else None

    def topic_name_vectors(self):
        return self.base.topic_name_vectors() if hasattr(self.base, "topic_name_vectors") else None

    def _lean(self, prior):
        """(topic name, strength) the channel is mostly about, or (None, 0)."""
        if prior is None or len(prior) == 0:
            return None, 0.0
        n = min(len(self._topics), len(prior))
        if n == 0:
            return None, 0.0
        i = max(range(n), key=lambda k: float(prior[k]))
        return self._topics[i][0], float(prior[i])

    def _hint(self, prior, channel_title: str, channel_topic: str = "") -> str:
        parts = []
        if channel_title:
            parts.append(f"канал «{channel_title}»")
        if channel_topic:
            parts.append(f"канал относится к теме «{channel_topic}» — по умолчанию относи посты к ней")
        if prior is not None:
            n = min(len(self._topics), len(prior))
            leaning = [
                self._topics[i][0]
                for i in sorted(range(n), key=lambda i: -float(prior[i]))
                if float(prior[i]) > 0.03
            ][:2]
            if leaning:
                parts.append("обычно публикует про: " + ", ".join(leaning))
            name, strength = self._lean(prior)
            if name and strength >= 0.08:
                parts.append(f"это тематический канал, в основном про «{name}»")
        return "; ".join(parts)

    def _classify_image(self, path: str, caption: str, hint: str):
        from llm import classify_image

        key = hash(("img", path, caption, hint, self.force))
        if key in self._cache:
            self.stats["cached"] += 1
            return True, self._cache[key]
        if self._failures >= 8:
            return False, None
        try:
            topic = classify_image(
                path, self._topics, self.vision_model, self.url, hint=hint, caption=caption,
                force=self.force,
            )
        except Exception as e:
            self._failures += 1
            logger.warning("Анализ картинки не удался (%s)", e)
            return False, None
        self.stats["calls"] += 1
        self._failures = 0  # only CONSECUTIVE failures disable the LLM
        self._cache[key] = topic
        return True, topic

    def set_feedback(self, feedback) -> None:
        self.feedback = list(feedback or [])
        self._cache.clear()  # answers may change with new examples

    def _pick_examples(self, text: str, n: int = 6):
        """Most similar manual corrections (word overlap) + the latest ones."""
        if not self.feedback:
            return []
        words = {w for w in re.findall(r"\w+", text.lower()) if len(w) > 3}
        scored = []
        for i, (t, topic) in enumerate(self.feedback):
            tw = {w for w in re.findall(r"\w+", t.lower()) if len(w) > 3}
            ov = len(words & tw) / (len(words | tw) or 1)
            scored.append((ov, -i, t, topic))
        scored.sort(reverse=True)
        picked = [(t, topic) for ov, _, t, topic in scored[:n] if ov > 0]
        for t, topic in self.feedback[:2]:  # most recent corrections
            if (t, topic) not in picked:
                picked.append((t, topic))
        return picked[: n + 2]

    def _classify(self, text: str, hint: str = ""):
        from llm import classify_topic

        examples = self._pick_examples(text)
        key = hash((text, hint, tuple(examples), self.force))
        if key in self._cache:
            self.stats["cached"] += 1
            return True, self._cache[key]
        if self._failures >= 8:  # LLM is down: stop hammering it this session
            return False, None
        try:
            topic = classify_topic(
                text, self._topics, self.model, self.url, hint=hint, examples=examples, force=self.force
            )
        except Exception as e:
            self._failures += 1
            logger.warning("ИИ-проверка не удалась (%s), использую быстрый фильтр", e)
            return False, None
        self.stats["calls"] += 1
        self._failures = 0  # only CONSECUTIVE failures disable the LLM
        self._cache[key] = topic
        return True, topic

    def score_batch(
        self, texts: list[str], prior=None, channel_title: str = "", channel_topic: str = "", **kw
    ) -> list[FilterResult]:
        images = kw.pop("images", None)
        results = self.base.score_batch(texts, prior=prior, **kw)
        hint = self._hint(prior, channel_title, channel_topic)
        lean_name, lean_strength = self._lean(prior)
        if channel_topic:
            lean_name, lean_strength = channel_topic, max(lean_strength, 0.2)
        # In a clearly thematic channel even weakly-matching posts are worth a
        # look by the LLM (the channel's subject is the context).
        _, strength = self._lean(prior)
        border = min(self._borderline, 0.10) if strength >= 0.08 else self._borderline
        if self.llm_primary:
            border = 0.0  # the LLM decides the category for every post
        out = []
        for idx, (text, res) in enumerate(zip(texts, results)):
            img = images[idx] if images and idx < len(images) else None
            if img and len(text.strip()) < 40 and self.vision_model:
                # picture (+ short caption) judged by the local vision model
                okv, topic = self._classify_image(img, text, hint)
                if okv:
                    if topic:
                        self.stats["accepted"] += 1
                        out.append(FilterResult(True, 0.7, self._example_by_name.get(topic, res.matched_example), "ИИ-фото"))
                    else:
                        self.stats["rejected"] += 1
                        out.append(FilterResult(False, min(res.score, 0.3), res.matched_example, "ИИ-нет"))
                    continue
            if not text.strip():
                # nothing to read: fall back to the channel's usual subject
                if lean_name and lean_strength >= 0.08:
                    out.append(FilterResult(True, 0.5, self._example_by_name.get(lean_name), "канал"))
                else:
                    out.append(res)
                continue
            if not self.llm_primary and res.score < border:
                out.append(res)  # too short for the LLM: embeddings/keywords decide
                continue
            ok, topic = self._classify(text, hint)
            if not ok:
                out.append(res)  # LLM unavailable: keep the fast verdict
            elif topic is None and res.is_relevant and (res.source == "слова" or res.score >= self.threshold + 0.08):
                # The LLM said "none", but embeddings/keywords are clearly
                # confident: both engines look at every post, either can add it.
                self.stats["rescued"] = self.stats.get("rescued", 0) + 1
                out.append(FilterResult(True, res.score, res.matched_example, "смысл"))
            elif topic is None:
                self.stats["rejected"] += 1
                out.append(FilterResult(is_relevant=False, score=min(res.score, 0.3), matched_example=res.matched_example, source="ИИ-нет"))
            else:
                self.stats["accepted"] += 1
                out.append(
                    FilterResult(
                        is_relevant=True,
                        score=max(res.score, 0.75),
                        matched_example=self._example_by_name.get(topic, res.matched_example),
                        source="ИИ",
                    )
                )
        return out


def build_filter(
    interest_examples: list[str],
    threshold: float = 0.48,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    topics=None,
    llm_model: str = "",
    llm_url: str = "http://localhost:11434",
):
    """Try to build the embedding filter; fall back to keyword filter.

    `device`: "auto" (default, prefers CUDA when available), "cuda", or
    "cpu". Termux/Android has no CUDA, so "auto" naturally becomes "cpu"
    there without any code changes.
    """
    base = _build_base_filter(interest_examples, threshold, model_name, device)
    flt = base
    if topics and any(t.keywords for t in topics):
        flt = TopicKeywordBoost(base, topics, threshold)
    if topics and llm_model:
        try:
            from llm import diagnose

            ready, why = diagnose(llm_model, llm_url)
            if ready:
                logger.info("ИИ-проверка тем включена (Ollama, модель %s).", llm_model)
                return LLMTopicVerifier(flt, topics, threshold, llm_model, llm_url)
            logger.warning(
                "ИИ-проверка выключена: %s. Работает быстрый фильтр. "
                "(Скачать модель: ollama pull %s)",
                why,
                llm_model,
            )
        except Exception:
            logger.warning("ИИ-проверка недоступна", exc_info=True)
    return flt


def _build_base_filter(interest_examples, threshold, model_name, device):
    try:
        return EmbeddingFilter(
            interest_examples, threshold=threshold, model_name=model_name, device=device
        )
    except Exception:
        logger.warning(
            "Failed to load embedding model '%s'; falling back to keyword filter.",
            model_name,
            exc_info=True,
        )
        return KeywordFilter(interest_examples)
