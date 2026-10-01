"""Optional local LLM (Ollama) for topic summaries.

Fully local: talks only to an Ollama server on this machine
(http://localhost:11434 by default). Disabled unless `llm_model` is set in
config.yaml. No cloud, no extra Python dependencies (plain urllib), and it
is only ever called for a whole topic group on demand -- never per post --
so it doesn't slow down the digest itself.

Setup (one time): install Ollama from ollama.com, then e.g.
    ollama pull qwen2.5:3b
and put `llm_model: "qwen2.5:3b"` into config.yaml.
"""
from __future__ import annotations

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

# Talk to the local Ollama directly: never through a system/VPN HTTP proxy
# (a proxy in the environment would send "localhost" requests off-machine
# and fail), and use 127.0.0.1 (Windows may resolve "localhost" to IPv6 ::1
# while Ollama listens on IPv4 only).
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _norm(url: str) -> str:
    return url.rstrip("/").replace("//localhost", "//127.0.0.1")


def _open(req_or_url, timeout):
    return _OPENER.open(req_or_url, timeout=timeout)


def is_available(url: str = "http://localhost:11434") -> bool:
    try:
        with _open(_norm(url) + "/api/tags", 3):
            return True
    except Exception:
        return False


def summarize_topic(
    topic: str,
    texts: list[str],
    model: str,
    url: str = "http://localhost:11434",
    timeout: int = 180,
) -> str:
    """Short Russian summary of several posts on one topic. Raises on error."""
    snippets = "\n\n".join(
        f"[{i + 1}] {' '.join(t.split())[:500]}" for i, t in enumerate(texts[:12])
    )
    prompt = (
        f"Тема: {topic}\n\nНиже несколько постов из Telegram-каналов.\n"
        "Сделай очень краткую сводку на русском: 3-5 коротких пунктов "
        "с главным. Без вступлений и без выдумок, только по тексту постов.\n\n"
        f"{snippets}\n\nСводка:"
    )
    body = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    ).encode("utf-8")
    req = urllib.request.Request(
        _norm(url) + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with _open(req, timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("response") or "").strip()


def available_models(url: str = "http://localhost:11434") -> list[str]:
    """Names of models pulled into the local Ollama, [] if it isn't running."""
    with _open(_norm(url) + "/api/tags", 5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m.get("name", "") for m in data.get("models", [])]


def diagnose(model: str, url: str = "http://localhost:11434") -> tuple[bool, str]:
    """(ready, human-readable explanation) -- says exactly what is wrong."""
    try:
        names = available_models(url)
    except Exception as e:
        return False, f"Ollama по адресу {_norm(url)} не отвечает ({type(e).__name__}: {e})"
    base = model.split(":")[0]
    if model in names or (":" not in model and any(n.split(":")[0] == base for n in names)):
        return True, "ok"
    shown = ", ".join(names) if names else "(пусто)"
    return False, f"Ollama работает, но модели '{model}' нет. Скачанные модели: {shown}"


def model_ready(model: str, url: str = "http://localhost:11434") -> bool:
    return diagnose(model, url)[0]


def classify_topic(
    text: str,
    topics: list[tuple[str, list[str]]],
    model: str,
    url: str = "http://localhost:11434",
    timeout: int = 240,  # a cold model load can take a minute or two
    hint: str = "",
) -> str | None:
    """Ask the local LLM which topic a post belongs to.

    `topics` = [(name, keywords)]. Returns the topic name, or None if the
    model says the post fits none of them. Raises on connection errors.
    """
    listing = "\n".join(
        f"- {name}" + (f" (слова: {', '.join(kws)})" if kws else "")
        for name, kws in topics
    )
    prompt = (
        "Ты классификатор постов из Telegram-каналов. Определи, относится ли "
        "пост ПО СМЫСЛУ (а не просто упоминанием слова) к одной из тем.\n\n"
        f"Темы:\n{listing}\n\n"
        "Пост относится к теме, только если эта тема — ГЛАВНЫЙ предмет поста. "
        "Если пост про что-то другое (реклама, розыгрыш, шутка, новости других сфер, "
        "случайное упоминание слова, побочная деталь) — ответь НЕТ. "
        "Не натягивай тему: когда сомневаешься — НЕТ.\n"
        + (
            (
                f"Контекст о канале: {hint}\n"
                "Тематика канала — важный контекст для КОРОТКОГО поста: если канал "
                "тематический, а пост не реклама и не противоречит теме канала, можно "
                "отнести его к теме канала, даже если нужных слов в тексте нет "
                "(подпись к фото, короткая новость). Но шутки, реклама и розыгрыши — НЕТ.\n"
                if len(text.strip()) < 200
                else f"Контекст о канале: {hint}\n"
                "Это лишь подсказка: решай по содержанию самого поста.\n"
            )
            if hint
            else ""
        )
        + 'Ответь строго JSON: {"topic": "<точное название одной темы или НЕТ>"}\n\n'
        f"Пост:\n<<<\n{' '.join(text.split())[:900]}\n>>>"
    )
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 40},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        _norm(url) + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with _open(req, timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    try:
        answer = str(json.loads(data.get("response", "{}")).get("topic", "")).strip()
    except Exception:
        return None
    low = answer.lower()
    for name, _ in topics:
        if name.lower() == low:
            return name
    for name, _ in topics:  # tolerate small deviations from the model
        if name.lower() in low and low != "нет":
            return name
    return None


_VISION_PREFS = ("qwen2.5vl", "qwen3-vl", "minicpm-v", "llama3.2-vision", "llava", "moondream")


def find_vision_model(url: str = "http://localhost:11434") -> str:
    """Name of a pulled Ollama model that can look at pictures, or ''."""
    try:
        names = available_models(url)
    except Exception:
        return ""
    for pref in _VISION_PREFS:
        for n in names:
            if pref in n.lower():
                return n
    return ""


def classify_image(
    image_path: str,
    topics: list[tuple[str, list[str]]],
    model: str,
    url: str = "http://localhost:11434",
    timeout: int = 240,
    hint: str = "",
    caption: str = "",
) -> str | None:
    """Look at a post's picture (+ its short caption) with a LOCAL vision
    model and pick the topic, or None. Raises on connection errors."""
    import base64

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    listing = "\n".join(
        f"- {name}" + (f" (слова: {', '.join(kws)})" if kws else "") for name, kws in topics
    )
    prompt = (
        "Это картинка из поста Telegram-канала. Определи, к какой теме относится пост.\n\n"
        f"Темы:\n{listing}\n\n"
        + (f"Подпись к картинке: {' '.join(caption.split())[:300]}\n" if caption.strip() else "")
        + (f"Контекст о канале: {hint}\n" if hint else "")
        + "Если канал тематический, а картинка не реклама и не противоречит теме канала — "
        "относи к теме канала. Если картинка не про эти темы — НЕТ.\n"
        'Ответь строго JSON: {"topic": "<точное название одной темы или НЕТ>"}'
    )
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "images": [b64],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 40},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        _norm(url) + "/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with _open(req, timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    try:
        answer = str(json.loads(data.get("response", "{}")).get("topic", "")).strip()
    except Exception:
        return None
    low = answer.lower()
    for name, _ in topics:
        if name.lower() == low:
            return name
    for name, _ in topics:
        if name.lower() in low and low != "нет":
            return name
    return None
