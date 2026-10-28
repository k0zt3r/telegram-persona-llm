import asyncio
import logging
import time

import httpx
from telegram import BusinessConnection, Message, Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    BusinessConnectionHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from bot.auth import authorize_by_password, is_allowed, logout_user
from bot.config import (
    ALLOWED_TELEGRAM_USER_IDS,
    BUSINESS_IGNORE_CHAT_IDS,
    MAX_REPLY_MESSAGES,
    OLLAMA_MODEL,
    OLLAMA_URL,
    OWNER_TAKEOVER_MINUTES,
    OWNER_TELEGRAM_ID,
    REPLY_DEBOUNCE_SECONDS,
    SPLIT_REPLY_LINES,
    TELEGRAM_PROXY_URL,
    TELEGRAM_TOKEN,
    TYPING_MAX_SECONDS,
    TYPING_SECONDS_PER_CHAR,
)
from bot.formatters import split_reply
from bot.memory import Memory
from bot.ollama_client import chat
from bot.persona import History, drop_repeats


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("neunizhayka")
logging.getLogger("httpx").setLevel(logging.WARNING)


history = History()
memory = Memory()
# Отложенные ответы по чатам: ждем, пока собеседник допишет серию сообщений.
pending_replies: dict[str, asyncio.Task] = {}
# chat key -> время (time.monotonic), до которого бот молчит, потому что владелец пишет сам.
owner_takeover_until: dict[str, float] = {}
# business_connection_id -> ID владельца подключенного аккаунта.
business_owners: dict[str, int] = {}
paused = False


def is_owner(user_id: int) -> bool:
    """Владелец — тот, кому разрешено подключать бота к своему аккаунту."""
    if OWNER_TELEGRAM_ID:
        return user_id == OWNER_TELEGRAM_ID
    return user_id in ALLOWED_TELEGRAM_USER_IDS


def chat_key(message: Message) -> str:
    """Ключ истории: business-чаты и личка с ботом не должны смешиваться."""
    return f"{message.business_connection_id or 'direct'}:{message.chat_id}"


def message_text(message: Message) -> str:
    return (message.text or message.caption or "").strip()


# ---------- Генерация и отправка ответа ----------


async def recall_context(key: str, chat_id: int | None) -> str:
    """Факты и похожие места из прошлой переписки для системного промпта."""
    try:
        fragments = await memory.recall(history.last_user_message(key), chat_id)
    except httpx.HTTPError:
        logger.exception("Memory search failed, answering without it")
        fragments = []
    if fragments:
        logger.info("Chat %s: %s memory fragments", key, len(fragments))
    return memory.prompt_block(fragments)


async def make_reply(key: str, chat_id: int | None) -> list[str]:
    """Генерирует ответ и режет его на сообщения; при сплошных повторах пробует еще раз.

    chat_id — собеседник, по чьей переписке искать в памяти (None — по всем чатам, для лички владельца).
    """
    context = await recall_context(key, chat_id)
    for _ in range(2):
        answer = await chat(history.build_messages(key, context))
        parts = split_reply(answer, SPLIT_REPLY_LINES)
        parts = drop_repeats(parts, history.said_before(key))
        if MAX_REPLY_MESSAGES > 0:
            parts = parts[:MAX_REPLY_MESSAGES]
        if parts:
            return parts
    return []


async def send_reply(context: ContextTypes.DEFAULT_TYPE, message: Message, key: str, parts: list[str]) -> None:
    """Отправляет ответ по частям с имитацией набора текста."""
    connection_id = message.business_connection_id

    for part in parts:
        try:
            await context.bot.send_chat_action(
                message.chat_id, ChatAction.TYPING, business_connection_id=connection_id
            )
        except TelegramError:
            pass
        await asyncio.sleep(min(len(part) * TYPING_SECONDS_PER_CHAR, TYPING_MAX_SECONDS))
        await context.bot.send_message(message.chat_id, part, business_connection_id=connection_id)

    history.add(key, "assistant", "\n".join(parts))


async def generate_and_reply(context: ContextTypes.DEFAULT_TYPE, message: Message, key: str) -> None:
    """Ждет паузу в сообщениях собеседника, генерирует ответ и отправляет его."""
    try:
        await asyncio.sleep(REPLY_DEBOUNCE_SECONDS)
        # По всем чатам ищем только для самого владельца в личке с ботом; остальным — их собственная история
        # (у тех, кто вошел по паролю, ее в индексе нет, и память им просто ничего не найдет).
        owner_testing = not message.business_connection_id and message.from_user and is_owner(message.from_user.id)
        parts = await make_reply(key, None if owner_testing else message.chat_id)
    except asyncio.CancelledError:
        return
    except httpx.HTTPError:
        logger.exception("Ollama request failed")
        if not message.business_connection_id:
            await message.reply_text("Ollama не отвечает, проверь, что она запущена.")
        return

    if not parts:
        logger.warning("Empty answer for %s", key)
        return

    # Отправку не прерываем, даже если собеседник успел написать еще: ответ уйдет целиком,
    # а новое сообщение запустит следующую генерацию.
    try:
        await asyncio.shield(send_reply(context, message, key, parts))
    except asyncio.CancelledError:
        pass
    except TelegramError:
        logger.exception("Failed to send reply to %s", key)


def schedule_reply(context: ContextTypes.DEFAULT_TYPE, message: Message, key: str) -> None:
    """Перезапускает отложенный ответ для чата."""
    previous = pending_replies.get(key)
    if previous and not previous.done():
        previous.cancel()
    pending_replies[key] = asyncio.create_task(generate_and_reply(context, message, key))


def cancel_reply(key: str) -> None:
    task = pending_replies.pop(key, None)
    if task and not task.done():
        task.cancel()


# ---------- Telegram Business ----------


async def get_business_owner(context: ContextTypes.DEFAULT_TYPE, connection_id: str) -> int | None:
    """Возвращает ID владельца business-подключения (с кэшем)."""
    if connection_id not in business_owners:
        try:
            connection = await context.bot.get_business_connection(connection_id)
        except TelegramError:
            logger.exception("Can't get business connection %s", connection_id)
            return None
        business_owners[connection_id] = connection.user.id
    return business_owners[connection_id]


async def handle_business_connection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Владелец подключил, изменил или отключил бота в настройках Telegram Business."""
    connection: BusinessConnection = update.business_connection
    user = connection.user

    if not connection.is_enabled:
        business_owners.pop(connection.id, None)
        logger.info("Business connection %s disabled by %s", connection.id, user.id)
        return

    if not is_owner(user.id):
        logger.warning("Rejected business connection from %s (%s)", user.id, user.username)
        return

    business_owners[connection.id] = user.id
    logger.info("Business connection %s enabled by owner %s", connection.id, user.id)

    rights = getattr(connection, "rights", None)
    can_reply = rights.can_reply if rights is not None else getattr(connection, "can_reply", True)
    note = "" if can_reply else "\n\nВнимание: у бота нет права отвечать в чатах — включи его в настройках Business."
    try:
        await context.bot.send_message(
            connection.user_chat_id,
            f"Подключен к аккаунту. Модель: {OLLAMA_MODEL}.\n/pause — остановить, /resume — продолжить.{note}",
        )
    except TelegramError:
        pass


async def handle_business_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сообщение в личном чате владельца: входящее от собеседника или исходящее от владельца."""
    message = update.business_message
    text = message_text(message)
    if not text:
        logger.info("Chat %s: no text (sticker/voice/media), skipped", message.chat_id)
        return

    owner_id = await get_business_owner(context, message.business_connection_id)
    if owner_id is None or not is_owner(owner_id):
        logger.info("Chat %s: connection %s is not the owner's, skipped", message.chat_id, message.business_connection_id)
        return

    key = chat_key(message)
    sender_id = message.from_user.id if message.from_user else None

    if sender_id == owner_id:
        # Пишет сам владелец (или это эхо ответа бота — его уже добавили в историю в send_reply).
        if message.sender_business_bot is None:
            history.add(key, "assistant", text)
            cancel_reply(key)
            owner_takeover_until[key] = time.monotonic() + OWNER_TAKEOVER_MINUTES * 60
            logger.info("Chat %s: owner wrote, bot silent for %s min", message.chat_id, OWNER_TAKEOVER_MINUTES)
        return

    history.add(key, "user", text)

    if paused or message.chat_id in BUSINESS_IGNORE_CHAT_IDS:
        logger.info("Chat %s: %s, skipped", message.chat_id, "paused" if paused else "in BUSINESS_IGNORE_CHAT_IDS")
        return
    takeover_left = owner_takeover_until.get(key, 0) - time.monotonic()
    if takeover_left > 0:
        logger.info("Chat %s: owner takeover, %.0f min left, skipped", message.chat_id, takeover_left / 60)
        return

    logger.info("Chat %s: message received, reply scheduled", message.chat_id)
    schedule_reply(context, message, key)


# ---------- Личка с ботом: тест и управление ----------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Неунижайка — отвечаю в личке за владельца через Telegram Business.\n\n"
        "Напиши мне что-нибудь — отвечу так, как ответил бы владелец.\n\n"
        "/reset — очистить историю этого чата\n"
        "/pause, /resume — выключить/включить ответы в business-чатах\n"
        "/auth пароль, /logout, /whoami"
    )


async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"Твой Telegram ID: {update.effective_user.id}")


async def auth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Использование: /auth пароль")
        return

    if authorize_by_password(update.effective_user.id, " ".join(context.args)):
        await update.message.reply_text("Доступ выдан.")
    else:
        await update.message.reply_text("Неверный пароль.")


async def logout(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logout_user(update.effective_user.id)
    await update.message.reply_text("Локальная авторизация удалена.")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    key = chat_key(update.message)
    cancel_reply(key)
    history.reset(key)
    await update.message.reply_text("История очищена.")


async def set_paused(update: Update, context: ContextTypes.DEFAULT_TYPE, value: bool) -> None:
    global paused
    if not is_owner(update.effective_user.id):
        await update.message.reply_text("Только для владельца.")
        return

    paused = value
    if value:
        for key in list(pending_replies):
            if not key.startswith("direct:"):
                cancel_reply(key)
    await update.message.reply_text("Ответы в чатах выключены." if value else "Ответы в чатах включены.")


async def pause(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await set_paused(update, context, True)


async def resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await set_paused(update, context, False)


async def handle_direct_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Личка с самим ботом — удобно тестировать стиль без Business."""
    user_id = update.effective_user.id
    if not (is_owner(user_id) or is_allowed(user_id)):
        await update.message.reply_text("Нет доступа. Используй /auth пароль.")
        return

    key = chat_key(update.message)
    history.add(key, "user", message_text(update.message))
    schedule_reply(context, update.message, key)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    if isinstance(context.error, TelegramError):
        logger.warning("Telegram error: %s", context.error)
    else:
        logger.exception("Unexpected error", exc_info=context.error)


async def post_init(application: Application) -> None:
    logger.info("Bot @%s started, Ollama %s, model %s", application.bot.username, OLLAMA_URL, OLLAMA_MODEL)
    logger.info("Memory: %s fragments, facts %s", len(memory), "loaded" if memory.facts else "not set")


def main() -> None:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN не задан")
    if not OWNER_TELEGRAM_ID and not ALLOWED_TELEGRAM_USER_IDS:
        raise RuntimeError("Задай OWNER_TELEGRAM_ID, иначе к боту сможет подключиться чужой аккаунт")

    builder = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .post_init(post_init)
    )
    if TELEGRAM_PROXY_URL:
        builder = builder.proxy(TELEGRAM_PROXY_URL).get_updates_proxy(TELEGRAM_PROXY_URL)

    application = builder.build()

    application.add_handler(BusinessConnectionHandler(handle_business_connection))
    application.add_handler(MessageHandler(filters.UpdateType.BUSINESS_MESSAGE, handle_business_message))

    private = filters.ChatType.PRIVATE & filters.UpdateType.MESSAGE
    application.add_handler(CommandHandler("start", start, filters=private))
    application.add_handler(CommandHandler("help", start, filters=private))
    application.add_handler(CommandHandler("whoami", whoami, filters=private))
    application.add_handler(CommandHandler("auth", auth, filters=private))
    application.add_handler(CommandHandler("logout", logout, filters=private))
    application.add_handler(CommandHandler("reset", reset, filters=private))
    application.add_handler(CommandHandler("pause", pause, filters=private))
    application.add_handler(CommandHandler("resume", resume, filters=private))
    application.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND, handle_direct_message))
    application.add_error_handler(error_handler)

    # Business-апдейты не приходят без явного allowed_updates.
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
