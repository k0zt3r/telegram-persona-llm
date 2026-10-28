import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _get_ids(name: str) -> set[int]:
    return {
        int(value.strip())
        for value in os.getenv(name, "").split(",")
        if value.strip().lstrip("-").isdigit()
    }


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AUTHORIZED_USERS_FILE = DATA_DIR / "authorized_users.json"
PERSONA_FILE = Path(os.getenv("PERSONA_FILE", DATA_DIR / "persona.txt"))
EXAMPLES_FILE = Path(os.getenv("EXAMPLES_FILE", DATA_DIR / "examples.jsonl"))
# Факты о владельце, которые бот видит всегда, — чтобы не выдумывать их.
FACTS_FILE = Path(os.getenv("FACTS_FILE", DATA_DIR / "facts.txt"))
# Индекс прошлой переписки (tools.build_memory): перед ответом бот ищет в нем похожие места.
MEMORY_FILE = Path(os.getenv("MEMORY_FILE", DATA_DIR / "memory.sqlite"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_PROXY_URL = os.getenv("TELEGRAM_PROXY_URL", "")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.8"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
# Модель эмбеддингов для поиска по переписке; bge-m3 нормально понимает русский.
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
# Сколько фрагментов прошлой переписки подкладывать и насколько они должны быть похожи (косинус).
# У bge-m3 на коротких разговорных фразах почти все попадает в 0.55–0.7; ниже 0.65 в основном шум.
# По умолчанию выключено: дообученная модель копирует найденные ответы дословно (docs/experiments.md).
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "0"))
MEMORY_MIN_SCORE = float(os.getenv("MEMORY_MIN_SCORE", "0.65"))

# Сколько примеров из examples.jsonl подмешивать в системный промпт (0 — не подмешивать,
# например после fine-tune, когда стиль уже в весах модели).
FEW_SHOT_EXAMPLES = int(os.getenv("FEW_SHOT_EXAMPLES", "25"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "20"))

BOT_ACCESS_PASSWORD = os.getenv("BOT_ACCESS_PASSWORD", "")
ALLOWED_TELEGRAM_USER_IDS = _get_ids("ALLOWED_TELEGRAM_USER_IDS")

# Telegram Business
# ID владельца аккаунта: бот принимает business-подключение только от него.
OWNER_TELEGRAM_ID = int(os.getenv("OWNER_TELEGRAM_ID", "0") or 0)
# Чаты (ID собеседников), в которых бот молчит, даже если доступ к ним выдан в настройках Business.
BUSINESS_IGNORE_CHAT_IDS = _get_ids("BUSINESS_IGNORE_CHAT_IDS")
# Если владелец сам написал в чат, бот замолкает в нем на столько минут.
OWNER_TAKEOVER_MINUTES = int(os.getenv("OWNER_TAKEOVER_MINUTES", "30"))
# Сколько секунд ждать после последнего сообщения собеседника, прежде чем отвечать:
# люди часто пишут несколькими сообщениями подряд.
REPLY_DEBOUNCE_SECONDS = float(os.getenv("REPLY_DEBOUNCE_SECONDS", "8"))
# Отправлять каждую строку ответа отдельным сообщением.
SPLIT_REPLY_LINES = _get_bool("SPLIT_REPLY_LINES", True)
# Максимум сообщений в одном ответе (0 — без ограничения): модель любит растекаться.
MAX_REPLY_MESSAGES = int(os.getenv("MAX_REPLY_MESSAGES", "3"))
# Имитация набора текста: секунд на символ и максимум на одно сообщение.
TYPING_SECONDS_PER_CHAR = float(os.getenv("TYPING_SECONDS_PER_CHAR", "0.05"))
TYPING_MAX_SECONDS = float(os.getenv("TYPING_MAX_SECONDS", "6"))

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "180"))
