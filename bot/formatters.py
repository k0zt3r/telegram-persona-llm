import re


MAX_TELEGRAM_MESSAGE_LENGTH = 4000

# Модель иногда подписывает реплики как в few-shot примерах — срезаем это.
_SPEAKER_PREFIX = re.compile(r"^\s*(Ты|Я|Assistant|Ассистент)\s*:\s*", re.IGNORECASE)


def split_reply(answer: str, split_lines: bool, max_messages: int = 0) -> list[str]:
    """Режет ответ модели на сообщения для Telegram."""
    # Владелец не пишет «ё», а модель упорно ставит — надежнее заменить, чем уговаривать в промпте.
    answer = answer.replace("ё", "е").replace("Ё", "Е")
    answer = _SPEAKER_PREFIX.sub("", answer.strip())
    parts = answer.split("\n") if split_lines else [answer]

    result = []
    for part in parts:
        part = _SPEAKER_PREFIX.sub("", part).strip()
        while part:
            result.append(part[:MAX_TELEGRAM_MESSAGE_LENGTH])
            part = part[MAX_TELEGRAM_MESSAGE_LENGTH:]

    if max_messages > 0:
        result = result[:max_messages]
    return result
