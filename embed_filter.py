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

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


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
        import torch
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
        encode_batch_size = 64 if self.device == "cuda" else 16
        self._encode_batch_size = encode_batch_size

        self._example_embeddings = self._model.encode(
            interest_examples,
            convert_to_numpy=True,
            normalize_embeddings=False,
            batch_size=encode_batch_size,
        )

    def score(self, text: str) -> FilterResult:
        if not text.strip():
            return FilterResult(is_relevant=False, score=0.0)

        embedding = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False,
            batch_size=self._encode_batch_size,
        )
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
        emb = self._model.encode(
            [t if t.strip() else " " for t in texts],
            convert_to_numpy=True,
            normalize_embeddings=False,
            batch_size=self._encode_batch_size,
        )
        return _cosine_similarity_matrix(emb, self._example_embeddings)

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
        if not non_empty_idx:
            return results  # type: ignore[return-value]

        embeddings = self._model.encode(
            [texts[i] for i in non_empty_idx],
            convert_to_numpy=True,
            normalize_embeddings=False,
            batch_size=self._encode_batch_size,
        )
        sims = _cosine_similarity_matrix(embeddings, self._example_embeddings)
        if prior is not None:
            sims = sims + np.asarray(prior, dtype=float)[None, :]
        for row, i in enumerate(non_empty_idx):
            best_idx = int(np.argmax(sims[row]))
            best_score = float(sims[row][best_idx])
            results[i] = FilterResult(
                is_relevant=best_score >= self.threshold,
                score=best_score,
                matched_example=self.interest_examples[best_idx],
            )
            results[i].source = "смысл"
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

    def describe(self) -> str:
        return getattr(self.base, "describe", lambda: "?")() + " + слова тем"

    def topic_matrix(self, texts: list[str]):
        fn = getattr(self.base, "topic_matrix", None)
        return fn(texts) if fn else None

    def score_batch(self, texts: list[str], **kw) -> list[FilterResult]:
        results = self.base.score_batch(texts, **kw)
        out = []
        for text, res in zip(texts, results):
            low = text.lower()
            best_ex, best_hits = None, 0
            for example, matchers in self._topics:
                hits = self._hits(low, matchers)
                if hits > best_hits:
                    best_ex, best_hits = example, hits
            if best_hits:
                kw_score = min(0.5 + 0.1 * (best_hits - 1), 0.9)
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

    def _hint(self, prior, channel_title: str) -> str:
        parts = []
        if channel_title:
            parts.append(f"канал «{channel_title}»")
        if prior is not None:
            n = len(self._topics)
            leaning = [
                self._topics[i][0]
                for i in sorted(range(n), key=lambda i: -float(prior[i]))
                if i < len(prior) and float(prior[i]) > 0.03
            ][:2]
            if leaning:
                parts.append("обычно публикует про: " + ", ".join(leaning))
        return "; ".join(parts)

    def _classify(self, text: str, hint: str = ""):
        from llm import classify_topic

        key = hash((text, hint))
        if key in self._cache:
            self.stats["cached"] += 1
            return True, self._cache[key]
        if self._failures >= 5:  # LLM is down: stop hammering it this session
            return False, None
        try:
            topic = classify_topic(text, self._topics, self.model, self.url, hint=hint)
        except Exception as e:
            self._failures += 1
            logger.warning("ИИ-проверка не удалась (%s), использую быстрый фильтр", e)
            return False, None
        self.stats["calls"] += 1
        self._cache[key] = topic
        return True, topic

    def score_batch(
        self, texts: list[str], prior=None, channel_title: str = "", **kw
    ) -> list[FilterResult]:
        results = self.base.score_batch(texts, prior=prior, **kw)
        hint = self._hint(prior, channel_title)
        out = []
        for text, res in zip(texts, results):
            if res.score < self._borderline or not text.strip():
                out.append(FilterResult(is_relevant=False, score=res.score, matched_example=res.matched_example))
                continue
            ok, topic = self._classify(text, hint)
            if not ok:
                out.append(res)  # LLM unavailable: keep the fast verdict
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
