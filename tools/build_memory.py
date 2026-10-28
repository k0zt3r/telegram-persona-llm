"""Строит память бота из экспорта Telegram: пары «реплика собеседника → твой ответ» с эмбеддингами.

Пример:
    python -m tools.build_memory path/to/result.json --exclude-chat 123456

Нужна модель эмбеддингов в Ollama (ollama pull bge-m3). Если прервать, повторный запуск продолжит:
уже посчитанные фрагменты пропускаются.
"""

import argparse
import asyncio
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from bot.config import EMBED_MODEL, MEMORY_FILE
from bot.memory import normalize
from bot.ollama_client import embed
from tools.prepare_dataset import chat_to_sessions, guess_my_id, load_chats


MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль",
          "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", type=Path, help="result.json из экспорта Telegram Desktop")
    parser.add_argument("--me-id", type=int, help="твой Telegram ID, если не определится сам")
    parser.add_argument("--out", type=Path, default=MEMORY_FILE)
    parser.add_argument("--exclude-chat", action="append", default=[], help="имя или ID чата (можно несколько раз)")
    parser.add_argument("--gap-minutes", type=int, default=60)
    parser.add_argument("--min-chars", type=int, default=40, help="пары короче (вместе) бесполезны для памяти")
    parser.add_argument("--max-chars", type=int, default=300, help="обрезать каждую реплику до стольких символов")
    parser.add_argument("--batch", type=int, default=32)
    return parser.parse_args()


def fragment(prompt: dict, reply: dict, max_chars: int) -> str:
    date = datetime.fromtimestamp(prompt["time"])
    # Хвост реплики собеседника (там обычно сам вопрос) и начало ответа.
    return (f"[{MONTHS[date.month - 1]} {date.year}]\n"
            f"собеседник: {prompt['content'][-max_chars:]}\n"
            f"ты: {reply['content'][:max_chars]}")


def collect(args: argparse.Namespace) -> list[tuple[int, str]]:
    export = json.loads(args.export.read_text(encoding="utf-8"))
    chats = load_chats(export)
    me_id = args.me_id or (export.get("personal_information") or {}).get("user_id") or guess_my_id(chats)
    if not me_id:
        raise SystemExit("Не нашел твой ID в экспорте, укажи --me-id.")
    me = f"user{me_id}"
    excluded = {str(value) for value in args.exclude_chat}

    fragments = []
    for chat in chats:
        # Только личные чаты: chat id в экспорте совпадает с ID собеседника, как chat_id в Business.
        if chat.get("type") != "personal_chat":
            continue
        if str(chat.get("id")) in excluded or str(chat.get("name")) in excluded:
            continue
        for session in chat_to_sessions(chat, me, args.gap_minutes * 60, 1000, with_time=True):
            for prompt, reply in zip(session, session[1:]):
                if prompt["role"] != "user" or reply["role"] != "assistant":
                    continue
                if len(prompt["content"]) + len(reply["content"]) < args.min_chars:
                    continue
                fragments.append((int(chat["id"]), fragment(prompt, reply, args.max_chars)))
    return list(dict.fromkeys(fragments))


async def main() -> None:
    args = parse_args()
    fragments = collect(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.out)
    db.execute("CREATE TABLE IF NOT EXISTS fragments (chat_id INTEGER, text TEXT, vector BLOB, UNIQUE(chat_id, text))")
    done = set(db.execute("SELECT chat_id, text FROM fragments").fetchall())
    todo = [f for f in fragments if f not in done]
    print(f"Фрагментов: {len(fragments)}, уже в памяти {len(fragments) - len(todo)}, считать {len(todo)} ({EMBED_MODEL})")

    started = time.monotonic()
    for start in range(0, len(todo), args.batch):
        batch = todo[start:start + args.batch]
        vectors = normalize(np.asarray(await embed([text for _, text in batch]), dtype=np.float32))
        db.executemany(
            "INSERT OR IGNORE INTO fragments VALUES (?, ?, ?)",
            [(chat_id, text, vector.tobytes()) for (chat_id, text), vector in zip(batch, vectors)],
        )
        db.commit()
        done_now = start + len(batch)
        rate = done_now / (time.monotonic() - started)
        print(f"\r{done_now}/{len(todo)}, ~{(len(todo) - done_now) / rate / 60:.0f} мин осталось", end="", flush=True)

    total = db.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
    db.close()
    print(f"\nГотово: {total} фрагментов в {args.out.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
