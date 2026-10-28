"""Готовит датасет из экспорта Telegram Desktop (JSON).

Пример:
    python -m tools.prepare_dataset path/to/result.json

На выходе в data/:
    train.jsonl, val.jsonl — диалоги в формате {"messages": [...]} для fine-tune (ChatML/ShareGPT);
    examples.jsonl         — короткие пары {"prompt", "reply"} для few-shot в промпте бота.
"""

import argparse
import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from bot.config import DATA_DIR
from bot.persona import load_persona


MEDIA_PLACEHOLDERS = {
    "sticker": "[стикер]",
    "voice_message": "[голосовое]",
    "video_message": "[кружок]",
    "video_file": "[видео]",
    "animation": "[гифка]",
    "audio_file": "[аудио]",
}

# Типы сущностей, которые Telegram сам размечает в text_entities.
ENTITY_REPLACEMENTS = {
    "phone": "<телефон>",
    "email": "<email>",
    "bank_card": "<карта>",
    "link": "<ссылка>",
    "text_link": None,  # оставляем видимый текст ссылки
    "mention": "<ник>",
}

REGEX_REPLACEMENTS = [
    (re.compile(r"https?://\S+|www\.\S+|t\.me/\S+"), "<ссылка>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
    (re.compile(r"(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)"), "<карта>"),
    (re.compile(r"(?<![\w])(?:\+7|8|\+\d{1,3})[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)"), "<телефон>"),
]


PLACEHOLDER_ONLY = re.compile(r"(\s*<(ссылка|телефон|email|карта|ник)>\s*)*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", type=Path, help="result.json из экспорта Telegram Desktop")
    parser.add_argument("--me-id", type=int, help="твой Telegram ID (нужен, если экспортирован один чат)")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--gap-minutes", type=int, default=60, help="пауза, после которой начинается новый диалог")
    parser.add_argument("--max-turns", type=int, default=16, help="максимум реплик в одном примере")
    parser.add_argument("--max-reply-chars", type=int, default=1000, help="длиннее — скорее всего копипаста")
    parser.add_argument("--min-my-messages", type=int, default=20, help="пропускать чаты, где ты почти не писал")
    parser.add_argument("--exclude-chat", action="append", default=[], help="имя или ID чата (можно несколько раз)")
    parser.add_argument("--include-groups", action="store_true", help="брать также групповые чаты")
    parser.add_argument("--max-dialogs-per-chat", type=int, default=0,
                        help="не больше N диалогов из одного чата, чтобы самые активные не задавили остальных (0 — без лимита)")
    parser.add_argument("--system-file", type=Path, help="файл с системным промптом для датасета (по умолчанию — персона бота)")
    parser.add_argument("--min-reply-chars", type=int, default=0,
                        help="ответы короче — контекст без обучения (\"weight\": 0), чтобы модель не училась отвечать «да/ок»")
    parser.add_argument("--min-long-share", type=float, default=0,
                        help="выкидывать диалоги, где доля ответов не короче --min-reply-chars меньше этой")
    parser.add_argument("--val-share", type=float, default=0.05)
    parser.add_argument("--examples", type=int, default=300, help="сколько пар сохранить для few-shot")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def anonymize(text: str) -> str:
    for pattern, replacement in REGEX_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def extract_text(message: dict) -> str:
    """Собирает текст сообщения, заменяя телефоны/почты/ссылки плейсхолдерами."""
    entities = message.get("text_entities")
    if entities is None:
        raw = message.get("text", "")
        entities = raw if isinstance(raw, list) else [{"type": "plain", "text": raw}]

    parts = []
    for entity in entities:
        if isinstance(entity, str):
            parts.append(entity)
            continue
        replacement = ENTITY_REPLACEMENTS.get(entity.get("type"))
        parts.append(replacement if replacement else entity.get("text", ""))

    return anonymize("".join(parts)).strip()


def media_placeholder(message: dict) -> str:
    if "photo" in message:
        return "[фото]"
    media_type = message.get("media_type")
    if media_type:
        return MEDIA_PLACEHOLDERS.get(media_type, "[файл]")
    if "file" in message:
        return "[файл]"
    return ""


def message_time(message: dict) -> float:
    if "date_unixtime" in message:
        return float(message["date_unixtime"])
    return datetime.fromisoformat(message["date"]).timestamp()


def load_chats(export: dict) -> list[dict]:
    if "chats" in export:
        return export["chats"].get("list", [])
    if "messages" in export:
        return [export]
    raise SystemExit("Не похоже на экспорт Telegram: нет ни 'chats', ни 'messages'.")


def guess_my_id(chats: list[dict]) -> int | None:
    """Владелец экспорта — отправитель, который есть почти во всех личных чатах."""
    seen: Counter = Counter()
    personal = [c for c in chats if c.get("type") == "personal_chat"]
    for chat in personal:
        for sender in {m.get("from_id") for m in chat.get("messages", []) if m.get("from_id")}:
            seen[sender] += 1

    if not seen:
        return None
    sender, count = seen.most_common(1)[0]
    if count < max(2, len(personal) // 2) or not sender.startswith("user"):
        return None
    return int(sender.removeprefix("user"))


def chat_to_sessions(chat: dict, me: str, gap_seconds: int, max_reply_chars: int,
                     with_time: bool = False) -> list[list[dict]]:
    """Режет чат на диалоги по паузам и склеивает подряд идущие реплики одного автора.

    with_time добавляет в реплику "time" (unixtime первого сообщения) — для памяти бота, не для датасета.
    """
    sessions: list[list[dict]] = []
    turns: list[dict] = []
    last_time = None

    for message in chat.get("messages", []):
        if message.get("type") != "message" or "forwarded_from" in message:
            continue

        timestamp = message_time(message)
        if last_time is not None and timestamp - last_time > gap_seconds:
            sessions.append(turns)
            turns = []
        last_time = timestamp

        role = "assistant" if message.get("from_id") == me else "user"
        text = extract_text(message)

        if role == "assistant":
            # Бот не умеет слать стикеры, голосовые и настоящие ссылки, поэтому учим только на тексте:
            # сообщение из одних плейсхолдеров модель повторяла бы буквально.
            text = "\n".join(line for line in text.split("\n") if not PLACEHOLDER_ONLY.fullmatch(line))
            if not text:
                continue
            if len(text) > max_reply_chars:
                continue
        else:
            placeholder = media_placeholder(message)
            text = f"{placeholder} {text}".strip() if placeholder else text
            if not text:
                continue

        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n" + text
        else:
            turns.append({"role": role, "content": text, **({"time": timestamp} if with_time else {})})

    sessions.append(turns)
    return [s for s in sessions if any(t["role"] == "assistant" for t in s)]


def session_to_windows(turns: list[dict], max_turns: int) -> list[list[dict]]:
    """Делит длинный диалог на куски, каждый начинается с реплики собеседника."""
    windows = []
    start = 0
    while start < len(turns):
        while start < len(turns) and turns[start]["role"] != "user":
            start += 1
        window = turns[start:start + max_turns]
        if window and window[-1]["role"] == "user":
            window = window[:-1]
        if any(t["role"] == "assistant" for t in window):
            windows.append(window)
        start += max(len(window), 1)
    return windows


def is_good_example(prompt: str, reply: str) -> bool:
    """Пара для few-shot: понятная без контекста, без медиа-заглушек и не из одного слова.

    Модель копирует примеры буквально, поэтому ответы вроде «3112» на вырванную реплику
    учат ее отвечать невпопад.
    """
    if not (15 <= len(prompt) <= 200 and 6 <= len(reply) <= 160):
        return False
    if "<" in prompt or "[" in prompt or "<" in reply:
        return False
    if len(re.findall(r"[а-яёa-z]+", reply.lower())) < 2:
        return False
    # Вопрос или явная реплика, на которую ответ осмыслен без предыдущей переписки.
    return "?" in prompt or len(prompt.split()) >= 4


def mark_short_replies(window: list[dict], min_chars: int) -> float:
    """Помечает короткие ответы "weight": 0 и возвращает долю длинных."""
    replies = [turn for turn in window if turn["role"] == "assistant"]
    for turn in replies:
        if len(turn["content"]) < min_chars:
            turn["weight"] = 0
    return sum("weight" not in turn for turn in replies) / len(replies)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    export = json.loads(args.export.read_text(encoding="utf-8"))
    chats = load_chats(export)
    me_id = args.me_id or (export.get("personal_information") or {}).get("user_id")
    if not me_id:
        me_id = guess_my_id(chats)
        if not me_id:
            raise SystemExit("Не нашел твой ID в экспорте, укажи его через --me-id (узнать: /whoami у бота).")
        print(f"Твой ID определен по чатам: {me_id} (если неверно — укажи --me-id)")
    me = f"user{me_id}"

    allowed_types = {"personal_chat"}
    if args.include_groups:
        allowed_types |= {"private_group", "private_supergroup", "public_supergroup"}
    excluded = {str(value) for value in args.exclude_chat}

    system_prompt = args.system_file.read_text(encoding="utf-8").strip() if args.system_file else load_persona()
    dialogs, examples = [], []
    used_chats = 0

    for chat in chats:
        if chat.get("type") not in allowed_types:
            continue
        if str(chat.get("id")) in excluded or str(chat.get("name")) in excluded:
            continue

        my_messages = sum(1 for m in chat.get("messages", []) if m.get("from_id") == me)
        if my_messages < args.min_my_messages:
            continue
        used_chats += 1

        sessions = chat_to_sessions(chat, me, args.gap_minutes * 60, args.max_reply_chars)
        chat_dialogs = []
        for session in sessions:
            for window in session_to_windows(session, args.max_turns):
                window = [dict(turn) for turn in window]  # few-shot ниже берет пары из sessions без пометок
                if args.min_reply_chars and mark_short_replies(window, args.min_reply_chars) < max(args.min_long_share, 1e-9):
                    continue
                chat_dialogs.append({"messages": [{"role": "system", "content": system_prompt}, *window]})
        if args.max_dialogs_per_chat and len(chat_dialogs) > args.max_dialogs_per_chat:
            chat_dialogs = random.sample(chat_dialogs, args.max_dialogs_per_chat)
        dialogs.extend(chat_dialogs)

        for session in sessions:

            for prev, turn in zip(session, session[1:]):
                if prev["role"] == "user" and turn["role"] == "assistant":
                    if is_good_example(prev["content"], turn["content"]):
                        examples.append({"prompt": prev["content"], "reply": turn["content"]})

    if not dialogs:
        raise SystemExit("Ничего не собралось — проверь --me-id и фильтры.")

    random.shuffle(dialogs)
    val_size = max(1, int(len(dialogs) * args.val_share))
    val, train = dialogs[:val_size], dialogs[val_size:]

    unique_examples = list({(e["prompt"], e["reply"]): e for e in examples}.values())
    random.shuffle(unique_examples)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "val.jsonl", val)
    write_jsonl(args.out_dir / "examples.jsonl", unique_examples[:args.examples])

    replies = [t for d in dialogs for t in d["messages"] if t["role"] == "assistant"]
    avg_len = sum(len(t["content"]) for t in replies) / len(replies)
    print(f"Чатов: {used_chats}")
    print(f"Диалогов: {len(dialogs)} (train {len(train)}, val {len(val)})")
    print(f"Твоих ответов: {len(replies)}, средняя длина {avg_len:.0f} символов")
    if args.min_reply_chars:
        trained = sum("weight" not in t for t in replies)
        print(f"Из них в обучении: {trained} ({trained / len(replies):.0%}), короче {args.min_reply_chars} — только контекст")
    print(f"Few-shot пар: {min(len(unique_examples), args.examples)} из {len(unique_examples)}")
    print(f"Сохранено в {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
