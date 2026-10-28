"""Файн-тюн Qwen3 на своей переписке на локальной видеокарте NVIDIA (Windows, Linux или WSL2).

То же, что notebooks/finetune_colab_14b.ipynb, но без Colab и Google Drive. Режим подбирается по видеопамяти:
    40+ ГБ (A6000, A40, A100, RTX 6000 Ada) — LoRA поверх bf16-весов, без квантования: качество чуть выше;
    20–40 ГБ (RTX 3090/4090, A5000, L4)    — QLoRA 4 бита, батч 2;
    до 20 ГБ (A4000, T4, 4080)              — QLoRA 4 бита, батч 1.
Эффективный батч всегда 8, так что результат от режима почти не зависит.

Пример:
    python train/finetune.py --data data/finetune2 --out runs/14b

Если обучение прервалось, запусти ту же команду еще раз: продолжит с последнего чекпойнта в --out.
Если упал только экспорт в GGUF — допиши --export-only: обучение не повторится, GGUF соберется из --out/lora.

Установка на Windows (Python 3.11–3.12, драйвер NVIDIA посвежее):
    python -m venv .venv-train
    .venv-train\\Scripts\\activate
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install unsloth
Экспорт в GGUF на Windows собирает llama.cpp и требует Visual Studio Build Tools (C++) и CMake.
Если их нет и поставить нельзя — запускай с --skip-gguf и забирай папку --out/lora: GGUF соберем отдельно.
"""

import argparse
import glob
import inspect
import math
import os
import shutil

# Unsloth нужно импортировать до transformers.
from unsloth import FastLanguageModel, is_bfloat16_supported

import torch
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

ASSISTANT = "<|im_start|>assistant\n"
END = "<|im_end|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="папка с train.jsonl и val.jsonl")
    parser.add_argument("--out", required=True, help="папка для чекпойнтов, LoRA и GGUF")
    parser.add_argument("--model", default="Qwen3-14B", help="имя модели в unsloth/ (Qwen3-8B, Qwen3-14B, Qwen3-32B)")
    parser.add_argument("--mode", choices=["auto", "bf16", "4bit"], default="auto", help="bf16 — без квантования")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--gguf-name", default="neunizhayka-14b-q4_k_m.gguf")
    parser.add_argument("--skip-gguf", action="store_true", help="не собирать GGUF (можно потом из --out/lora)")
    parser.add_argument("--export-only", action="store_true", help="не обучать, только собрать GGUF из --out/lora")
    return parser.parse_args()


def pick_mode(requested: str) -> tuple[bool, int]:
    """Возвращает (load_in_4bit, batch_size) по объему видеопамяти."""
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}, {vram_gb:.0f} ГБ")
    if requested == "bf16" or (requested == "auto" and vram_gb >= 40):
        return False, 2 if vram_gb < 70 else 4
    return True, 2 if vram_gb >= 20 else 1


def make_args(**kwargs):
    """TrainingArguments без параметров, которых нет в установленной версии transformers."""
    known = inspect.signature(TrainingArguments.__init__).parameters
    dropped = [k for k in kwargs if k not in known]
    if dropped:
        print("Нет в этой версии transformers, пропускаю:", dropped)
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in known})


def make_tokenize(tokenizer, max_seq_length: int):
    def tokenize(example):
        # Поле weight есть только у коротких ответов: они остаются контекстом, но не идут в лосс.
        messages = [{"role": m["role"], "content": m["content"]} for m in example["messages"]]
        trainable = [m.get("weight") != 0 for m in example["messages"] if m["role"] == "assistant"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)

        spans, pos = [], 0
        for train_it in trainable:
            start = text.index(ASSISTANT, pos) + len(ASSISTANT)
            end = text.index(END, start) + len(END)
            if train_it:
                spans.append((start, end))
            pos = end

        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=True, max_length=max_seq_length)
        labels = [
            token if any(a < e and b > s for s, e in spans) else -100
            for token, (a, b) in zip(enc["input_ids"], enc["offset_mapping"])
        ]
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    return tokenize


def export_gguf(model, tokenizer, args: argparse.Namespace) -> None:
    gguf_dir = os.path.join(args.out, "gguf")
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
    found = [p for p in glob.glob(f"{gguf_dir}*/**/*.gguf", recursive=True) if "q4_k_m" in p.lower()]
    target = os.path.join(args.out, args.gguf_name)
    shutil.copy(found[0], target)
    print("Готово:", target)


def main() -> None:
    args = parse_args()
    load_in_4bit, batch_size = pick_mode(args.mode)
    lora_dir = os.path.join(args.out, "lora")

    if args.export_only:
        # Базовая модель берется из adapter_config.json в папке LoRA.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_dir, max_seq_length=args.max_seq_length, load_in_4bit=load_in_4bit,
        )
        export_gguf(model, tokenizer, args)
        return

    grad_accum = max(1, 8 // batch_size)
    model_name = f"unsloth/{args.model}-unsloth-bnb-4bit" if load_in_4bit else f"unsloth/{args.model}"
    print(f"Модель: {model_name}, батч {batch_size} x {grad_accum}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset("json", data_files={
        "train": os.path.join(args.data, "train.jsonl"),
        "validation": os.path.join(args.data, "val.jsonl"),
    })
    dataset = dataset.map(make_tokenize(tokenizer, args.max_seq_length), remove_columns=dataset["train"].column_names)
    dataset = dataset.filter(lambda x: any(label != -100 for label in x["labels"]))
    total = sum(len(x) for x in dataset["train"]["labels"])
    trained = sum(label != -100 for x in dataset["train"]["labels"] for label in x)
    print(dataset)
    print(f"Токенов: {total}, из них в обучении {trained} ({trained / total:.0%})")

    ckpt_dir = os.path.join(args.out, "checkpoints")
    # warmup_ratio убрали в transformers 5, поэтому 3% шагов считаем сами.
    warmup_steps = math.ceil(len(dataset["train"]) / (batch_size * grad_accum) * args.epochs * 0.03)
    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        args=make_args(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=0.01,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            group_by_length=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            output_dir=ckpt_dir,
            seed=3407,
            report_to="none",
        ),
    )

    resume = bool(glob.glob(os.path.join(ckpt_dir, "checkpoint-*")))
    print("Продолжаю с чекпойнта" if resume else "Обучение с нуля")
    trainer.train(resume_from_checkpoint=resume or None)

    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print("LoRA:", lora_dir, "| лучший eval_loss:", trainer.state.best_metric)

    if not args.skip_gguf:
        export_gguf(model, tokenizer, args)


if __name__ == "__main__":
    main()
