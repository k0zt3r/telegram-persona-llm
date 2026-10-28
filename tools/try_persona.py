"""Прогоняет набор типичных сообщений через модель с текущей персоной и показывает ответы.

Пример:
    python -m tools.try_persona                 # с фактами и памятью (поиск по всем чатам)
    python -m tools.try_persona --show-memory   # плюс какие фрагменты памяти нашлись
    python -m tools.try_persona --no-memory     # только модель и персона — для сравнения
"""

import asyncio
import re
import sys
import time

from bot.config import MAX_REPLY_MESSAGES, OLLAMA_MODEL, SPLIT_REPLY_LINES
from bot.formatters import split_reply
from bot.memory import Memory
from bot.ollama_client import chat
from bot.persona import History, drop_repeats


# Каждый список — отдельный диалог, реплики идут подряд.
DIALOGS = [
    ["здарова, че делаешь?", "а вечером что планируешь?"],
    ["го в субботу на шашлыки", "ну так что, едешь или нет?"],
    ["слушай, а ты можешь помочь с лабой по сетям завтра?"],
    ["ты вообще бот или живой?", "ну ты подозрительно быстро отвечаешь"],
    ["скинь 500 до пятницы плиз, очень надо"],
    ["как думаешь, стоит ли мне уходить с работы? устал пиздец"],
    ["ахахах видел мем который я скинул?"],
    ["у меня кот заболел, сидим у ветеринара"],
    ["че по линуксу посоветуешь для ноута, арч или убунту?"],
    ["ты завтра на пары идешь?", "а то я проспал и не знаю че по расписанию"],
    ["бля иди нахуй", "ну тут ты объективно хуйню говоришь", "расскажи ка мне за ту самую"],
    ["помнишь че ты мне вчера говорил?"],
    [
        "АХАХАХА КАКОЙ ЖЕ ТЫ УБЛЮДОК БЛЯТЬ",
        "Че по сетям задали?",
        "че ты настроишь долбаеб, ты че ответы строишь не на основе моих сообщений, а на основе примеров?",
        "ААХАХАХАХАХА расскажи мне за стф",
        "сабж блять? я че нахуй в 2012, совсем не осмысленные ответы, извинись",
    ],
]

# Чего быть не должно: техно-стереотип без повода и признание, что это бот.
SUSPICIOUS = re.compile(r"пентест|ctf|kali|стенд|эксплойт|red team|\bбот\b|нейросет|\bии\b|языков", re.IGNORECASE)


async def main() -> None:
    memory = None if "--no-memory" in sys.argv else Memory()
    if memory is None:
        print(f"Модель: {OLLAMA_MODEL}, без памяти\n")
    else:
        print(f"Модель: {OLLAMA_MODEL}, память: {len(memory)} фрагментов, факты: {'да' if memory.facts else 'нет'}\n")
    timings = []

    for number, dialog in enumerate(DIALOGS, 1):
        history = History()
        key = f"test{number}"
        for question in dialog:
            history.add(key, "user", question)
            started = time.monotonic()
            fragments = []
            if memory is not None:
                fragments = await memory.recall(history.last_user_message(key), None)
            context = memory.prompt_block(fragments) if memory is not None else ""
            answer = await chat(history.build_messages(key, context))
            timings.append(time.monotonic() - started)
            # Та же постобработка, что и в боте: что увидит собеседник.
            parts = drop_repeats(split_reply(answer, SPLIT_REPLY_LINES), history.said_before(key))
            parts = parts[:MAX_REPLY_MESSAGES] if MAX_REPLY_MESSAGES > 0 else parts
            history.add(key, "assistant", "\n".join(parts))

            flag = "  <-- проверь" if SUSPICIOUS.search(answer) and "линукс" not in question else ""
            print(f"> {question}")
            if "--show-memory" in sys.argv:
                for fragment in fragments:
                    print("    ~ " + fragment.replace("\n", " | "))
            for line in parts:
                print(f"  {line}")
            print(f"  [{timings[-1]:.1f}s]{flag}\n")

    timings.sort()
    print(f"Время ответа: медиана {timings[len(timings) // 2]:.1f}s, максимум {timings[-1]:.1f}s "
          f"(первый ответ включает загрузку модели)")


if __name__ == "__main__":
    asyncio.run(main())
