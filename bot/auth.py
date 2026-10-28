import json
from pathlib import Path

from bot.config import ALLOWED_TELEGRAM_USER_IDS, AUTHORIZED_USERS_FILE, BOT_ACCESS_PASSWORD


def load_authorized_users() -> set[int]:
    """Читает список пользователей, которые прошли пароль."""
    if not AUTHORIZED_USERS_FILE.exists():
        return set()

    data = json.loads(AUTHORIZED_USERS_FILE.read_text(encoding="utf-8"))
    return {int(user_id) for user_id in data.get("users", [])}


def save_authorized_users(users: set[int]) -> None:
    """Сохраняет список авторизованных пользователей."""
    Path(AUTHORIZED_USERS_FILE).parent.mkdir(parents=True, exist_ok=True)
    data = {"users": sorted(users)}
    AUTHORIZED_USERS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def is_allowed(user_id: int) -> bool:
    """Проверяет доступ пользователя."""
    if user_id in ALLOWED_TELEGRAM_USER_IDS:
        return True

    return user_id in load_authorized_users()


def authorize_by_password(user_id: int, password: str) -> bool:
    """Авторизует пользователя по паролю."""
    if not BOT_ACCESS_PASSWORD:
        return False

    if password != BOT_ACCESS_PASSWORD:
        return False

    users = load_authorized_users()
    users.add(user_id)
    save_authorized_users(users)
    return True


def logout_user(user_id: int) -> None:
    """Удаляет пользователя из локальной авторизации."""
    users = load_authorized_users()
    users.discard(user_id)
    save_authorized_users(users)
