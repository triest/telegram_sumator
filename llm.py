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
    if model == EMBEDDED:
        return _embedded_text(prompt, max_tokens=500)
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


def suggest_topic_name(
    samples: list[str],
    existing: list[str],
    model: str,
    url: str = "http://localhost:11434",
    timeout: int = 180,
) -> str:
    """Short Russian name (1-3 words) for a group of similar posts. Raises on error."""
    snippets = "\n".join(f"- {' '.join(t.split())[:300]}" for t in samples[:6])
    prompt = (
        "Ниже несколько похожих постов из Telegram-каналов. Придумай короткое название "
        "общей темы на русском: 1-3 слова, существительное, как название рубрики "
        "(например: «Кино», «Кулинария», «Скидки и акции», «Психология отношений»).\n"
        + (f"Эти темы уже есть, не повторяй их: {', '.join(existing)}.\n" if existing else "")
        + f"\nПосты:\n{snippets}\n\nОтветь только названием, без кавычек и пояснений."
    )
    if model == EMBEDDED:
        raw = _embedded_text(prompt, max_tokens=16)
    else:
        body = json.dumps(
            {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2, "num_predict": 16}}
        ).encode("utf-8")
        req = urllib.request.Request(
            _norm(url) + "/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        with _open(req, timeout) as resp:
            raw = (json.loads(resp.read().decode("utf-8")).get("response") or "").strip()
    name = raw.strip().splitlines()[0] if raw.strip() else ""
    name = name.strip(" 	\"'«»“”.:;-—*#")
    return name[:40] if 2 <= len(name) <= 60 else ""


def available_models(url: str = "http://localhost:11434") -> list[str]:
    """Names of models pulled into the local Ollama, [] if it isn't running."""
    with _open(_norm(url) + "/api/tags", 5) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m.get("name", "") for m in data.get("models", [])]


def diagnose(model: str, url: str = "http://localhost:11434") -> tuple[bool, str]:
    """(ready, human-readable explanation) -- says exactly what is wrong."""
    if model == EMBEDDED:
        return load_embedded()
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


# --------------------------------------------------------------------------
# Embedded model: a small GGUF model that runs INSIDE the app (llama-cpp-python),
# no Ollama needed. Used when config `llm_model` is "embedded".
# Output is constrained by a grammar, so the answer is always one of the topics.
# --------------------------------------------------------------------------
import threading
from pathlib import Path

EMBEDDED = "embedded"
EMBEDDED_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
EMBEDDED_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"  # ~2 GB; 1.5B variant: qwen2.5-1.5b-instruct-q4_k_m.gguf
_engine = None
_engine_lock = threading.Lock()
_engine_error = ""
_settings = {"repo": EMBEDDED_REPO, "file": EMBEDDED_FILE, "dir": "models"}


def configure_embedded(repo: str = "", filename: str = "", models_dir: str = "") -> None:
    """Choose another GGUF model (call before the first use)."""
    if repo:
        _settings["repo"] = repo
    if filename:
        _settings["file"] = filename
    if models_dir:
        _settings["dir"] = models_dir


def load_embedded() -> tuple[bool, str]:
    """Download (once) and load the embedded model. Idempotent."""
    global _engine, _engine_error
    with _engine_lock:
        if _engine is not None:
            return True, "ok"
        if _engine_error:
            return False, _engine_error
        try:
            from llama_cpp import Llama
        except Exception as e:
            _engine_error = (
                "не установлен llama-cpp-python (pip install llama-cpp-python "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu): %s" % e
            )
            return False, _engine_error
        models_dir = Path(_settings["dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / _settings["file"]
        try:
            if not path.exists():
                logger.info("Скачиваю встроенную ИИ-модель %s (один раз, ~2 ГБ)…", _settings["file"])
                _download_model(_settings["repo"], _settings["file"], path)
            try:
                import llama_cpp

                gpu = -1 if llama_cpp.llama_supports_gpu_offload() else 0
            except Exception:
                gpu = 0
            logger.info("Загружаю встроенную ИИ-модель (GPU-слои: %s)…", gpu)
            _engine = Llama(
                model_path=str(path), n_ctx=2048, n_gpu_layers=gpu, n_threads=None, verbose=False
            )
            logger.info("Встроенная ИИ-модель готова: %s", path.name)
            return True, "ok"
        except Exception as e:
            _engine_error = "встроенная модель не загрузилась: %s" % e
            logger.warning(_engine_error, exc_info=True)
            return False, _engine_error


def _download_model(repo: str, filename: str, dest: Path) -> None:
    """Stream the GGUF file from HuggingFace (uses the system/VPN proxy settings)."""
    import time

    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "telegram-digest"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        got, last = 0, time.time()
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if time.time() - last > 10:
                last = time.time()
                logger.info("Скачано %d из %d МБ…", got >> 20, total >> 20)
    tmp.replace(dest)


def _gbnf_literal(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _embedded_topic(prompt: str, names: list[str]) -> str:
    """Run the model with a grammar that only allows {"topic": "<one of names>"}."""
    from llama_cpp import LlamaGrammar

    alternatives = " | ".join(_gbnf_literal(n) for n in names)
    grammar = LlamaGrammar.from_string('root ::= "{\\"topic\\": \\"" (' + alternatives + ') "\\"}"')
    with _engine_lock:
        out = _engine.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            grammar=grammar,
            temperature=0,
            max_tokens=48,
        )
    return out["choices"][0]["message"]["content"]


def _embedded_text(prompt: str, max_tokens: int = 400) -> str:
    ok, why = load_embedded()
    if not ok:
        raise RuntimeError(why)
    with _engine_lock:
        out = _engine.create_chat_completion(
            messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=max_tokens
        )
    return (out["choices"][0]["message"]["content"] or "").strip()


def _ask_topic(prompt, topics, model, url, timeout, force) -> str:
    """Send the prompt, return the raw topic string the model answered."""
    if model == EMBEDDED:
        ok, why = load_embedded()
        if not ok:
            raise RuntimeError(why)
        names = [n for n, _ in topics] + ([] if force else ["НЕТ"])
        raw = _embedded_topic(prompt, names)
        try:
            return str(json.loads(raw).get("topic", "")).strip()
        except Exception:
            return ""
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
        return str(json.loads(data.get("response", "{}")).get("topic", "")).strip()
    except Exception:
        return ""


def _match_topic(answer: str, topics) -> str | None:
    low = answer.lower()
    for name, _ in topics:
        if name.lower() == low:
            return name
    for name, _ in topics:  # tolerate small deviations from the model
        if name.lower() in low and low != "нет":
            return name
    return None




def classify_topic(
    text: str,
    topics: list[tuple[str, list[str]]],
    model: str,
    url: str = "http://localhost:11434",
    timeout: int = 240,  # a cold model load can take a minute or two
    hint: str = "",
    examples: list[tuple[str, str]] | None = None,
    force: bool = False,
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
        + (
            "ВАЖНО: каждый пост обязательно принадлежит ровно одной из тем — ответ НЕТ запрещён. "
            "Выбери ближайшую по смыслу тему. Если пост общий (шутка, реклама, анонс, подпись к фото, "
            "поздравление), выбери тему, к которой относится сам канал.\n"
            if force
            else "Пост относится к теме, если она ЗАМЕТНО присутствует в посте (не мимоходом одним словом). "
            "Если подходят несколько тем — выбери ближайшую. Если пост про другое "
            "(реклама, розыгрыш, шутка, другая сфера) — ответь НЕТ.\n"
        )
        + (
            "Примеры того, как пользователь сам разметил похожие посты (учитывай их!):\n"
            + "\n".join(
                f"- «{' '.join(t.split())[:220]}» → {name or 'НЕТ'}" for t, name in examples
            )
            + "\n"
            if examples
            else ""
        )
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
        + (
            'Ответь строго JSON: {"topic": "<точное название одной темы из списка>"}\n\n'
            if force
            else 'Ответь строго JSON: {"topic": "<точное название одной темы или НЕТ>"}\n\n'
        )
        + f"Пост:\n<<<\n{' '.join(text.split())[:900]}\n>>>"
    )
    answer = _ask_topic(prompt, topics, model, url, timeout, force)
    return _match_topic(answer, topics)


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
    force: bool = False,
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
        + (
            "Ответ НЕТ запрещён: выбери ближайшую тему из списка, а если непонятно — тему канала.\n"
            'Ответь строго JSON: {"topic": "<точное название одной темы из списка>"}'
            if force
            else "Если канал тематический, а картинка не реклама и не противоречит теме канала — "
            "относи к теме канала. Если картинка не про эти темы — НЕТ.\n"
            'Ответь строго JSON: {"topic": "<точное название одной темы или НЕТ>"}'
        )
    )
    if model == EMBEDDED:
        raise RuntimeError("встроенная модель не умеет смотреть картинки")
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
    return _match_topic(answer, topics)
