import re

import httpx

from bot.config import (
    EMBED_MODEL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
    OLLAMA_URL,
    REQUEST_TIMEOUT,
)


THINK_BLOCK = re.compile(r"^.*</think>", re.DOTALL)


def strip_thinking(answer: str) -> str:
    """Вырезает рассуждения, если модель все-таки начала думать вслух."""
    return THINK_BLOCK.sub("", answer).strip()


async def chat(messages: list[dict]) -> str:
    """Отправляет историю диалога в Ollama /api/chat и возвращает текст ответа."""
    if "qwen3" in OLLAMA_MODEL and messages and messages[0]["role"] == "system":
        # Qwen3 не всегда слушается параметра think, а этот тег в промпте отключает рассуждения надежно.
        messages = [{**messages[0], "content": messages[0]["content"] + "\n\n/no_think"}, *messages[1:]]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        # Отключает "размышления" у моделей с thinking-режимом (qwen3 и т.п.).
        "think": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            # По умолчанию Ollama берет короткий контекст и молча обрезает начало промпта с персоной.
            "num_ctx": OLLAMA_NUM_CTX,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }

    timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=20)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        if response.status_code == 400 and "think" in response.text:
            # Модель не поддерживает параметр think — повторяем без него.
            payload.pop("think")
            response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

    return strip_thinking((data.get("message") or {}).get("content", ""))


async def embed(texts: list[str]) -> list[list[float]]:
    """Эмбеддинги текстов через Ollama /api/embed (одним запросом на пачку)."""
    # На CPU: на маленькой видеокарте модель эмбеддингов вытесняла бы основную, и та грузилась бы заново
    # перед каждым ответом (+15–20 с). Одна фраза на CPU считается за доли секунды.
    payload = {"model": EMBED_MODEL, "input": texts, "keep_alive": OLLAMA_KEEP_ALIVE, "options": {"num_gpu": 0}}
    timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=20)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{OLLAMA_URL}/api/embed", json=payload)
        response.raise_for_status()
        return response.json()["embeddings"]
