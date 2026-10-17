"""Local picture recognition: which topic does a post's picture belong to?

Fully inside the program (no Ollama): a multilingual CLIP model maps the picture
and the Russian topic texts into one space; the picture goes to the closest topic.
"Distractor" classes (food, animals, scenery, screenshots...) soak up pictures that
fit none of the topics, so they are not forced into one.

On ~50 real pictures from the user's channels (erotica, film posters, war photos,
old drawings vs. food, cats, landscapes, screenshots...) this picked the right
topic for ~78% and had no false positives.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_MODEL = "clip-ViT-B-32"
TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
MIN_PROB = 0.35  # probability of the best topic needed to accept a picture
LOGIT_SCALE = 100.0  # CLIP's own temperature

DISTRACTORS = [
    "еда на тарелке",
    "кошка или собака",
    "пейзаж, природа, лес",
    "скриншот экрана телефона",
    "портрет человека, селфи",
    "спорт, тренировка",
    "улица города, здание",
    "рекламный баннер товара",
    "интерьер, комната",
    "текст на фоне, плакат",
]


def _load(name: str, device: str):
    from sentence_transformers import SentenceTransformer

    try:  # cached after the first download -> offline start, no HTTP checks
        return SentenceTransformer(name, device=device, local_files_only=True)
    except Exception:
        logger.info("Скачиваю модель для картинок '%s' (один раз)…", name)
        return SentenceTransformer(name, device=device)


class ImageTopicClassifier:
    def __init__(self, examples: Sequence[str], device: str = "cpu", min_prob: float = MIN_PROB) -> None:
        """`examples`: topic texts like "Кино: кино, фильмы" (name, then keywords)."""
        self.min_prob = min_prob
        self._device = device
        self._labels = [self._label(e) for e in examples]
        self._n = len(self._labels)
        self._img = None
        self._txt = None
        self._text_vecs: Optional[np.ndarray] = None
        self.failed = False

    @staticmethod
    def _label(example: str) -> str:
        if ": " not in example:
            return example
        name, kws = example.split(": ", 1)
        return f"{name}: {', '.join(k.strip() for k in kws.split(',')[:6])}"

    def _ensure(self) -> bool:
        if self.failed:
            return False
        if self._img is not None:
            return True
        try:
            self._img = _load(IMAGE_MODEL, self._device)
            self._txt = _load(TEXT_MODEL, self._device)
            self._text_vecs = self._txt.encode(
                self._labels + DISTRACTORS, normalize_embeddings=True, convert_to_numpy=True
            )
            logger.info("Распознавание картинок включено (CLIP, %s).", self._device)
            return True
        except Exception:
            self.failed = True
            logger.warning("Распознавание картинок недоступно (модель не загрузилась)", exc_info=True)
            return False

    def classify(self, paths: Sequence[str]) -> list[tuple[Optional[int], float]]:
        """For each picture: (index of the topic or None, probability). Never raises."""
        out: list[tuple[Optional[int], float]] = [(None, 0.0)] * len(paths)
        if not paths or not self._ensure():
            return out
        from PIL import Image

        ok_idx, images = [], []
        for i, p in enumerate(paths):
            try:
                with Image.open(p) as im:
                    images.append(im.convert("RGB").copy())
                ok_idx.append(i)
            except Exception:
                continue
        if not images:
            return out
        try:
            vecs = self._img.encode(images, normalize_embeddings=True, convert_to_numpy=True, batch_size=16)
        except Exception:
            logger.warning("Картинки не распознаны", exc_info=True)
            return out
        logits = vecs @ self._text_vecs.T * LOGIT_SCALE
        prob = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob /= prob.sum(axis=1, keepdims=True)
        for row, i in zip(prob, ok_idx):
            j = int(row.argmax())
            if j < self._n and row[j] >= self.min_prob:
                out[i] = (j, float(row[j]))
            else:
                out[i] = (None, float(row[j]) if j < self._n else 0.0)
        return out
