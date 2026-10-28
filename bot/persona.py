import json
import random
import re
from collections import defaultdict, deque
from functools import lru_cache

from bot.config import EXAMPLES_FILE, FEW_SHOT_EXAMPLES, HISTORY_LIMIT, PERSONA_FILE


DEFAULT_PERSONA = (
    "Ты — владелец этого Telegram-аккаунта и переписываешься от своего имени. "
    "Пиши коротко, по-человечески, как в мессенджере. "
    "Никогда не говори, что ты бот, ИИ или языковая модель. "
    "Если не знаешь факта о себе или собеседнике, отвечай уклончиво, ничего не выдумывай."
)


def load_persona() -> str:
    """Читает описание персоны из файла или возвращает дефолт."""
    if PERSONA_FILE.exists():
        text = PERSONA_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
    return DEFAULT_PERSONA


def load_examples() -> list[dict]:
    """Читает пары {"prompt": ..., "reply": ...} из examples.jsonl."""
    if not EXAMPLES_FILE.exists():
        return []

    examples = []
    for line in EXAMPLES_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            examples.append(json.loads(line))
    return examples


@lru_cache(maxsize=1)
def build_system_prompt() -> str:
    """Собирает системный промпт: персона + случайные примеры реальных ответов.

    Промпт собирается один раз на запуск и одинаков для всех чатов: так Ollama переиспользует
    уже посчитанный префикс, а не обрабатывает пару тысяч токенов примеров на каждое сообщение.
    """
    persona = load_persona()
    examples = load_examples()

    if FEW_SHOT_EXAMPLES <= 0 or not examples:
        return persona

    # Примеры идут первыми, а инструкции — последними, ближе к диалогу: если поставить примеры
    # в конец, модель продолжает их, а не отвечает собеседнику, и тащит оттуда готовые фразы.
    sample = random.sample(examples, min(FEW_SHOT_EXAMPLES, len(examples)))
    lines = [f"— {ex['prompt']}\n— {ex['reply']}" for ex in sample]
    return (
        "Ниже обрывки твоих старых переписок с другими людьми. Они показывают только твою манеру: "
        "длину сообщений, пунктуацию, тон. Это не текущий разговор — не продолжай их и не копируй "
        "из них фразы и темы.\n\n" + "\n\n".join(lines) + "\n\n---\n\n" + persona
    )


def normalize_line(line: str) -> str:
    """Строка без регистра, пунктуации и эмодзи — чтобы ловить повторы с мелкими отличиями."""
    return " ".join(re.findall(r"[а-яёa-z0-9]+", line.lower().replace("ё", "е")))


def drop_repeats(parts: list[str], said_before: set[str]) -> list[str]:
    """Убирает строки, которые бот уже писал (модель любит зацикливаться на своих фразах)."""
    result, seen = [], set(said_before)
    for part in parts:
        key = normalize_line(part)
        # Короткие «да», «ахах», «ща гляну» повторять можно, это нормально для переписки.
        if len(key) > 12 and key in seen:
            continue
        seen.add(key)
        result.append(part)
    return result


class History:
    """Последние сообщения каждого чата в памяти процесса."""

    def __init__(self, limit: int = HISTORY_LIMIT) -> None:
        self._chats: dict[str, deque] = defaultdict(lambda: deque(maxlen=limit))

    def add(self, chat_key: str, role: str, content: str) -> None:
        chat = self._chats[chat_key]
        # Подряд идущие реплики одного автора склеиваем, как в датасете.
        if chat and chat[-1]["role"] == role:
            chat[-1]["content"] += "\n" + content
        else:
            chat.append({"role": role, "content": content})

    def said_before(self, chat_key: str) -> set[str]:
        """Нормализованные строки, которые бот уже писал в этом чате."""
        return {
            normalize_line(line)
            for turn in self._chats.get(chat_key, ())
            if turn["role"] == "assistant"
            for line in turn["content"].split("\n")
        }

    def reset(self, chat_key: str) -> None:
        self._chats.pop(chat_key, None)

    def last_user_message(self, chat_key: str) -> str:
        """Последняя (склеенная) реплика собеседника — по ней ищем в памяти."""
        for turn in reversed(self._chats.get(chat_key, ())):
            if turn["role"] == "user":
                return turn["content"]
        return ""

    def build_messages(self, chat_key: str, context: str = "") -> list[dict]:
        """context — добавка к системному промпту (факты и память); идет после персоны,
        чтобы общий префикс промпта оставался одинаковым и Ollama его переиспользовала."""
        return [{"role": "system", "content": build_system_prompt() + context}, *self._chats[chat_key]]
