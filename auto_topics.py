"""Auto-create topics for posts that fit none of the existing ones.

Posts no topic could place are embedded (locally) and grouped by similarity.
A group of at least `min_posts` similar posts becomes a new topic: keywords come
from the words that are shared inside the group but rare outside it, the name is
suggested by the embedded LLM (falls back to the top keywords if it's unavailable).
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Two posts belong to one group when their (normalised) embeddings are at least this
# similar. Tuned on real Telegram posts with Qwen3-Embedding: reddit stories, deal
# channels, gadget news, relationship psychology... separate cleanly at ~0.55-0.6.
CLUSTER_SIM = 0.58
# A group whose centre is this close to an existing topic's name is not a new topic.
EXISTING_TOPIC_SIM = 0.62
# Mean similarity of a group's posts to its centre; mixed bags fall below this.
MIN_COHESION = 0.66

_STOP = set(
    """это как что так для при про или его она они оно все всё еще уже только очень можно нужно будет
    если когда чтобы потому также этот этой этом эти этих тоже которые который которая которое есть были
    было была быть себя свой своих мне меня тебя вас нас нам вам них ним них там тут здесь даже чем
    более менее после перед между через без под над сейчас сегодня вчера завтра года году лет раз
    http https www com the and for you with that this have from потом точно происходит увидели хорошо хотел
    хочу просто почему какой какая какие каждый несколько много многие после поэтому теперь тогда однако
    вообще конечно именно можете могут должны нужны давай давайте смотри слово слова пост посты друзья
    друг привет спасибо пожалуйста ставь подписаться подписывайтесь канал канале""".split()
)
_WORD = re.compile(r"[а-яёa-z][а-яёa-z]{4,}")


@dataclass
class NewTopic:
    name: str
    keywords: list[str]
    members: list[int] = field(default_factory=list)  # indices into the input texts


def _words(text: str) -> set[str]:
    text = re.sub(r"https?://\S+|@\w+", " ", text.lower())
    return {w for w in _WORD.findall(text) if w not in _STOP}


def cluster(vectors: np.ndarray, sim: float = CLUSTER_SIM) -> list[list[int]]:
    """Greedy single-pass grouping against running centroids (vectors are L2-normalised)."""
    centroids: list[np.ndarray] = []
    groups: list[list[int]] = []
    for i, v in enumerate(vectors):
        best, best_s = -1, sim
        for k, c in enumerate(centroids):
            s = float(v @ (c / (np.linalg.norm(c) + 1e-12)))
            if s >= best_s:
                best, best_s = k, s
        if best < 0:
            centroids.append(v.copy())
            groups.append([i])
        else:
            centroids[best] += v
            groups[best].append(i)
    return groups


def extract_keywords(member_texts: Sequence[str], all_texts: Sequence[str], limit: int = 7) -> list[str]:
    """Words shared by several posts of the group but rare in the other unplaced posts."""
    in_group = Counter()
    for t in member_texts:
        in_group.update(_words(t))
    overall = Counter()
    for t in all_texts:
        overall.update(_words(t))
    n = len(all_texts)
    scored = []
    for w, df in in_group.items():
        if df < 2:
            continue
        scored.append((df * math.log((n + 1) / (overall[w] - df + 1) + 1.0), w))
    scored.sort(reverse=True)
    return [w for _, w in scored[:limit]]


def _fallback_name(keywords: list[str]) -> str:
    return " / ".join(w.capitalize() for w in keywords[:2]) or "Новая тема"


def discover_topics(
    texts: Sequence[str],
    embed: Callable[[list[str]], np.ndarray],
    existing_names: Sequence[str],
    namer: Optional[Callable[[list[str], list[str]], str]] = None,
    min_posts: int = 4,
    max_new: int = 2,
    existing_vectors: Optional[np.ndarray] = None,
) -> list[NewTopic]:
    """Find up to `max_new` new topics among posts no existing topic matched.

    `embed` returns L2-normalised vectors. `existing_vectors` (same space) are the
    embeddings of the current topic names, used to avoid duplicating a topic.
    `namer(sample_texts, existing_names)` returns a short name or "".
    """
    texts = [" ".join(t.split())[:1000] for t in texts]
    idx = [i for i, t in enumerate(texts) if len(t) >= 40]  # tiny posts carry no topic
    if len(idx) < min_posts:
        return []
    vecs = embed([texts[i] for i in idx])
    groups = [g for g in cluster(vecs) if len(g) >= min_posts]
    groups.sort(key=len, reverse=True)
    out: list[NewTopic] = []
    taken = {n.lower() for n in existing_names}
    for g in groups:
        if len(out) >= max_new:
            break
        centre = vecs[g].mean(axis=0)
        centre /= np.linalg.norm(centre) + 1e-12
        if float((vecs[g] @ centre).mean()) < MIN_COHESION:
            continue  # a loose mix of unrelated posts, not a topic
        if existing_vectors is not None and len(existing_vectors):
            if float((existing_vectors @ centre).max()) >= EXISTING_TOPIC_SIM:
                continue
        members = [idx[k] for k in g]
        sample = [texts[i] for i in members]
        keywords = extract_keywords(sample, [texts[i] for i in idx])
        name = ""
        if namer is not None:
            try:
                name = namer(sample[:6], list(existing_names))
            except Exception as e:  # LLM missing/slow: keywords will do
                logger.info("Название темы от ИИ не получено (%s), беру ключевые слова", e)
        name = (name or (_fallback_name(keywords) if keywords else "")).strip()
        if not name or name.lower() in taken:
            continue
        taken.add(name.lower())
        out.append(NewTopic(name=name, keywords=keywords, members=members))
    return out
