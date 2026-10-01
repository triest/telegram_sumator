"""Android core: Telethon + local relevance scoring.

Runs inside Chaquopy. Kotlin passes a `helper` object that exposes the
on-device models (ONNX embeddings, MediaPipe LLM) and receives progress.
"""
import asyncio
import json
import os
import re
import threading
import time

import numpy as np
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.types import Channel, Chat

# Public API credentials of the official Telegram Desktop client.
API_ID = 2040
API_HASH = "b18441a1ff607e10a989891a5462e627"

_loop = None
_client = None
_data_dir = None
_phone = None
_hash = None
_sp = None  # sentencepiece processor


def _start_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    threading.Thread(target=_loop.run_forever, daemon=True).start()


def _run(coro, timeout=None):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result(timeout)


def init(data_dir):
    global _data_dir, _client
    _data_dir = data_dir
    os.makedirs(data_dir, exist_ok=True)
    if _loop is None:
        _start_loop()

    async def make():
        global _client
        _client = TelegramClient(
            os.path.join(data_dir, "tg_session"), API_ID, API_HASH,
            device_model="Android", system_version="13", app_version="10.14.0",
            lang_code="ru", system_lang_code="ru",
        )
        await _client.connect()
        return await _client.is_user_authorized()

    return bool(_run(make(), 60))


def send_code(phone):
    global _phone, _hash
    _phone = phone.strip()

    async def go():
        global _hash
        sent = await _client.send_code_request(_phone)
        _hash = sent.phone_code_hash

    _run(go(), 60)
    return "sent"


def sign_in(code):
    async def go():
        try:
            await _client.sign_in(_phone, code.strip(), phone_code_hash=_hash)
            return "ok"
        except SessionPasswordNeededError:
            return "password"

    return _run(go(), 60)


def sign_in_password(password):
    async def go():
        await _client.sign_in(password=password)
        return "ok"

    return _run(go(), 60)


def logout():
    async def go():
        await _client.log_out()

    _run(go(), 30)


# ---------------------------------------------------------------- channels

async def _dialogs():
    res, seen = [], set()
    for folder, archived in ((0, False), (1, True)):
        try:
            async for d in _client.iter_dialogs(folder=folder):
                if d.id in seen:
                    continue
                e = d.entity
                if isinstance(e, Channel) or (isinstance(e, Chat) and getattr(e, "megagroup", False)):
                    seen.add(d.id)
                    res.append({
                        "id": d.id,
                        "title": d.name or "",
                        "username": getattr(e, "username", None) or "",
                        "broadcast": bool(getattr(e, "broadcast", False)),
                        "unread": int(getattr(d, "unread_count", 0) or 0),
                        "read_max": int(getattr(d.dialog, "read_inbox_max_id", 0) or 0),
                        "archived": archived,
                        "_entity": e,
                    })
        except Exception:
            if folder == 0:
                raise
    return res


def list_channels():
    d = _run(_dialogs(), 180)
    return json.dumps([{k: v for k, v in c.items() if k != "_entity"} for c in d], ensure_ascii=False)


# ---------------------------------------------------------------- scoring

def load_tokenizer(spm_path):
    global _sp
    import sentencepiece as spm

    _sp = spm.SentencePieceProcessor()
    _sp.Load(spm_path)
    return True


def _tok(text, max_len=128):
    ids = _sp.EncodeAsIds(text[:1500])
    out = [0]
    for i in ids[: max_len - 2]:
        out.append(3 if i == 0 else i + 1)
    out.append(2)
    return out


def _embed(helper, texts):
    """-> (n, dim) float32 L2-normalised matrix, or None."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    vecs = []
    B = 16
    for i in range(0, len(texts), B):
        chunk = [_tok(t) for t in texts[i:i + B]]
        raw = helper.embed(json.dumps(chunk))
        arr = np.array(json.loads(raw), dtype=np.float32).reshape(len(chunk), -1)
        vecs.append(arr)
    return np.vstack(vecs)


def _kw_hits(text_low, keywords):
    hits = 0
    for kw in keywords:
        k = kw.strip().lower()
        if len(k) < 2:
            continue
        stem = k[: max(4, len(k) - 2)] if len(k) > 5 else k
        if re.search(r"(?<!\w)" + re.escape(stem), text_low):
            hits += 1
    return hits


_PROMPT = (
    "Ты классификатор постов Telegram. Темы:\n{topics}\n\n"
    "Канал: {channel}\nПост:\n{text}\n\n"
    "Выбери ОДНУ тему, только если она — ГЛАВНЫЙ предмет поста. Реклама, розыгрыши, шутки, "
    "побочные упоминания и другие сферы — ответь: нет. При сомнении — нет. "
    "Ответь только названием темы."
)


def _llm_pick(helper, topics, channel, text):
    prompt = _PROMPT.format(
        topics="\n".join(f"- {t['name']}: {', '.join(t['keywords'][:8])}" for t in topics),
        channel=channel, text=text[:700],
    )
    ans = (helper.classify(prompt) or "").strip().lower()
    for t in topics:
        if t["name"].lower() in ans:
            return t["name"]
    return None


# ---------------------------------------------------------------- scan

async def _scan(settings, helper):
    topics = [t for t in settings.get("topics", []) if t.get("name")]
    blacklist = set(int(x) for x in settings.get("blacklist", []))
    limit = int(settings.get("per_channel_limit", 40))
    threshold = float(settings.get("threshold", 0.42))
    use_llm = bool(settings.get("use_llm", True)) and helper.hasLlm()
    budget = int(settings.get("llm_budget", 30))
    only_broadcast = bool(settings.get("only_broadcast", True))
    images = bool(settings.get("show_images", True))
    use_emb = helper.hasEmbed() and _sp is not None and bool(topics)

    helper.progress("Список каналов…")
    dialogs = await _dialogs()
    todo = [
        c for c in dialogs
        if c["unread"] > 0 and c["id"] not in blacklist and (c["broadcast"] or not only_broadcast)
    ]
    helper.progress(f"Каналов с непрочитанным: {len(todo)}")

    sem = asyncio.Semaphore(6)
    posts = []
    done = 0

    async def fetch(c):
        nonlocal done
        async with sem:
            try:
                n = min(c["unread"], limit)
                async for m in _client.iter_messages(c["_entity"], limit=n, min_id=c["read_max"]):
                    txt = (m.message or "").strip()
                    if not txt:
                        continue
                    posts.append({
                        "channel_id": c["id"], "channel": c["title"], "username": c["username"],
                        "msg_id": m.id, "date": int(m.date.timestamp()), "text": txt,
                        "has_photo": bool(getattr(m, "photo", None)),
                        "has_video": bool(getattr(m, "video", None) or getattr(m, "gif", None)),
                        "_msg": m,
                    })
            except Exception:
                pass
            done += 1
            if done % 5 == 0 or done == len(todo):
                helper.progress(f"Загрузка постов: {done}/{len(todo)}")

    await asyncio.gather(*(fetch(c) for c in todo))
    helper.progress(f"Постов получено: {len(posts)}. Анализ…")

    names = [t["name"] for t in topics]
    result = []

    if not topics:
        for p in posts:
            p.update(topic="Все", score=0.0, engine="-")
            result.append(p)
    else:
        sims = None
        if use_emb:
            try:
                tvec = _embed(helper, [f"{t['name']}: {', '.join(t['keywords'])}" for t in topics])
                pvec = []
                B = 64
                for i in range(0, len(posts), B):
                    helper.progress(f"Эмбеддинги: {min(i + B, len(posts))}/{len(posts)}")
                    pvec.append(_embed(helper, [p["text"] for p in posts[i:i + B]]))
                pvec = np.vstack(pvec) if pvec else np.zeros((0, tvec.shape[1]), np.float32)
                sims = pvec @ tvec.T if len(posts) else np.zeros((0, len(topics)))
            except Exception as e:  # model failure -> keyword-only
                helper.progress(f"Эмбеддинги недоступны ({e}), только ключевые слова")
                sims = None

        cands = []  # (distance_to_accept, idx, topic) for LLM
        for i, p in enumerate(posts):
            low = p["text"].lower()
            best_name, best, hits_best = None, -1.0, 0
            for j, t in enumerate(topics):
                h = _kw_hits(low, t["keywords"])
                s = float(sims[i, j]) if sims is not None else 0.0
                s += min(0.3, 0.12 * h)
                if s > best:
                    best, best_name, hits_best = s, t["name"], h
            p["score"] = round(best, 3)
            if sims is None:
                if hits_best > 0:
                    p.update(topic=best_name, engine="kw")
                    result.append(p)
                continue
            if best >= threshold + 0.12 or (hits_best >= 2 and best >= threshold):
                p.update(topic=best_name, engine="emb")
                result.append(p)
            elif best >= threshold - 0.12:
                cands.append((abs(best - threshold), i, best_name))

        if use_llm and cands:
            cands.sort()
            cands = cands[:budget]
            helper.progress(f"ИИ-проверка спорных постов: 0/{len(cands)}")
            for k, (_, i, guess) in enumerate(cands, 1):
                p = posts[i]
                try:
                    pick = await asyncio.to_thread(_llm_pick, helper, topics, p["channel"], p["text"])
                except Exception:
                    pick = guess if p["score"] >= threshold else None
                if pick:
                    p.update(topic=pick, engine="llm")
                    result.append(p)
                helper.progress(f"ИИ-проверка спорных постов: {k}/{len(cands)}")
        else:
            for _, i, guess in cands:
                p = posts[i]
                if p["score"] >= threshold:
                    p.update(topic=guess, engine="emb")
                    result.append(p)

    # posts that matched nothing go to a separate group (newest 100, no images)
    if topics:
        accepted = {id(p) for p in result}
        others = sorted((p for p in posts if id(p) not in accepted), key=lambda p: -p["date"])[:100]
        for p in others:
            p.update(topic="Прочее (не подошло)", engine="")
            p["has_photo"] = p["has_video"] = False
            result.append(p)

    # thumbnails for accepted posts
    if images:
        isem = asyncio.Semaphore(4)
        tdir = os.path.join(_data_dir, "thumbs")
        os.makedirs(tdir, exist_ok=True)

        async def thumb(p):
            if not (p["has_photo"] or p["has_video"]):
                return
            out = os.path.join(tdir, f"{p['channel_id']}_{p['msg_id']}.jpg")
            if os.path.exists(out):
                p["image"] = out
                return
            async with isem:
                try:
                    r = await _client.download_media(p["_msg"], file=out, thumb=-1)
                    if r:
                        p["image"] = out
                except Exception:
                    pass

        helper.progress(f"Картинки: {sum(1 for p in result if p['has_photo'] or p['has_video'])}")
        await asyncio.gather(*(thumb(p) for p in result))

    order = {n: i for i, n in enumerate(names)}
    result.sort(key=lambda p: (order.get(p.get("topic"), 99), -p["date"]))
    out = []
    for p in result:
        link = (f"https://t.me/{p['username']}/{p['msg_id']}" if p["username"]
                else f"https://t.me/c/{str(p['channel_id']).replace('-100', '', 1)}/{p['msg_id']}")
        out.append({
            "channel": p["channel"], "date": p["date"], "text": p["text"],
            "topic": p.get("topic", ""), "score": p.get("score", 0), "engine": p.get("engine", ""),
            "image": p.get("image", ""), "link": link,
            "has_video": p["has_video"], "channel_id": p["channel_id"], "msg_id": p["msg_id"],
        })
    helper.progress(f"Готово: подошло {len(out)} из {len(posts)}")
    return out


def scan(settings_json, helper):
    settings = json.loads(settings_json)
    res = _run(_scan(settings, helper), 3600)
    return json.dumps(res, ensure_ascii=False)
