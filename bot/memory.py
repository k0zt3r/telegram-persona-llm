"""Память бота: факты о владельце и поиск по его прошлой переписке (RAG).

Индекс строит tools.build_memory: пары «реплика собеседника → ответ владельца» с эмбеддингами.
"""

import sqlite3
from pathlib import Path

import numpy as np

from bot.config import FACTS_FILE, MEMORY_FILE, MEMORY_MIN_SCORE, MEMORY_TOP_K
from bot.ollama_client import embed


def load_facts() -> str:
    """Факты о владельце; строки с # — комментарии для себя, в промпт не идут."""
    if not FACTS_FILE.exists():
        return ""
    lines = FACTS_FILE.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#")).strip()


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class Memory:
    """Фрагменты прошлой переписки в памяти процесса: несколько десятков тысяч векторов — это мегабайты."""

    def __init__(self, path: Path = MEMORY_FILE) -> None:
        self.facts = load_facts()
        self.chat_ids = np.empty(0, dtype=np.int64)
        self.texts: list[str] = []
        self.vectors = np.empty((0, 0), dtype=np.float32)
        if not path.exists():
            return

        with sqlite3.connect(path) as db:
            rows = db.execute("SELECT chat_id, text, vector FROM fragments").fetchall()
        if rows:
            self.chat_ids = np.array([row[0] for row in rows], dtype=np.int64)
            self.texts = [row[1] for row in rows]
            self.vectors = np.stack([np.frombuffer(row[2], dtype=np.float32) for row in rows])

    def __len__(self) -> int:
        return len(self.texts)

    async def recall(self, query: str, chat_id: int | None) -> list[str]:
        """Похожие фрагменты из переписки с этим собеседником; chat_id=None — из всех чатов.

        В чужих чатах ищем только по их собственной истории: иначе бот пересказывал бы
        одному человеку переписку с другим.
        """
        # По «здарова» и «ахах» искать бессмысленно: находится случайный шум.
        if MEMORY_TOP_K <= 0 or not self.texts or len(query.strip()) < 12:
            return []
        mask = np.ones(len(self.texts), dtype=bool) if chat_id is None else self.chat_ids == chat_id
        if not mask.any():
            return []

        query_vector = normalize(np.asarray((await embed([query]))[0], dtype=np.float32))
        scores = np.where(mask, self.vectors @ query_vector, -1.0)
        best = np.argsort(scores)[::-1][:MEMORY_TOP_K]
        return [self.texts[i] for i in best if scores[i] >= MEMORY_MIN_SCORE]

    def prompt_block(self, fragments: list[str]) -> str:
        """Добавка к системному промпту: факты и найденные фрагменты."""
        parts = []
        if self.facts:
            parts.append(
                "Факты о тебе. Это фон, а не тема для разговора: не пересказывай их и не представляйся, "
                "используй, только когда к слову. Чего тут нет — не знаешь и не придумываешь:\n" + self.facts
            )
        if fragments:
            parts.append(
                "Похожие места из прошлой переписки — чтобы помнить контекст. Это было раньше, а не сегодня; "
                "дословно не повторяй:\n\n" + "\n\n".join(fragments)
            )
        return "\n\n" + "\n\n".join(parts) if parts else ""
