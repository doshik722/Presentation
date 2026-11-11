# Здесь подключаем SFT-разогрев: LoRA на эпизодах, собранных той же функцией среды, что и роллауты RL.
"""SFT-разогрев T-lite (decisions.md, раздел 9, решение 03.11.2026).

Примеры — ``data/tasks/sft/sft_dataset.jsonl`` (без поиска) и ``data/tasks/sft_search/sft_search_dataset.jsonl``
(с поиском). Каждый пример собирается в токены функцией эпизода (``environment.run_episode``), как роллаут: шаблон
чата — только для начала эпизода, реплики модели и выдача инструментов — как их собирает среда (диалог шаблоном
заново не пересобирается, ``subtleties.md``). Loss — только по репликам модели (маска эпизода).

Обучение — ``transformers.Trainer`` с LoRA (peft); логиты считаются только в позициях, где есть метка.
"""
from __future__ import annotations

import dataclasses
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..environment import END, Episode, run_episode
from ..sft_search import Example as SearchExample, Turn, replay
from ..tasks import PresentationRequest, load_tasks

PLAIN = Path("data/tasks/sft/sft_dataset.jsonl")
SEARCH = Path("data/tasks/sft_search/sft_search_dataset.jsonl")
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class SFTConfig:
    model_id: str = "t-tech/T-lite-it-2.1"
    revision: str = "main"
    lora_rank: int = 32
    lora_alpha: int = 32
    learning_rate: float = 2e-4
    epochs: int = 3
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    eval_tasks: int = 9  # отложенные темы SFT: по loss на них выбирается эпоха
    seed: int = 0
    max_tokens: int = 16000  # длиннее — пример не берётся (печатается); проверяется до обучения
    eval_topics: list | None = None  # отложенные темы списком (иначе — случайно по seed); с ними — их семейства
    save_epochs: bool = False  # сохранить адаптер каждой эпохи (epochs/epoch_<k>/), а не только лучший по loss

    @classmethod
    def load(cls, path: str | Path) -> "SFTConfig":
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Sample:
    id: str  # тема
    kind: str  # plain | search
    author: str  # кто составил пример
    ids: list[int]
    mask: list[int]  # 1 — токен реплики модели (в loss)

    @property
    def target_tokens(self) -> int:
        return sum(self.mask)


# --------------------------------------------------------------------------- эпизоды

def request_for(record: dict, tasks: dict) -> PresentationRequest:
    """Запрос примера: задание набора с режимом поиска и картинок примера; текст должен совпасть с примером."""
    request = dataclasses.replace(tasks[record["id"]].request, search="Поиск: есть" in record["request"],
                                  images=bool(record["images"]))
    if request.render() != record["request"]:
        raise ValueError(f"{record['id']}: запрос примера не совпадает с заданием набора")
    return request


def plain_episode(record: dict, prompt_ids: list[int], tokenizer) -> Episode:
    """Пример без поиска — эпизод из одной реплики модели."""
    reply = tokenizer.encode(record["answer"] + END, add_special_tokens=False)

    def generate(ids, budget):
        return list(reply), True

    episode = run_episode(prompt_ids, tokenizer, generate, None, max_new_tokens=10**6)
    if episode.finish != "stop" or episode.answer.strip() != record["answer"].strip():
        raise ValueError(f"{record['id']}: эпизод без поиска собрался не так, как написан ({episode.finish})")
    return episode


def search_episode(record: dict, prompt_ids: list[int], tokenizer) -> Episode:
    """Пример с поиском — реплики модели и выдача по порядку (``sft_search.replay``)."""
    fields = {f.name for f in dataclasses.fields(SearchExample)}
    example = SearchExample(**{**{k: v for k, v in record.items() if k in fields},
                               "turns": [Turn(t["role"], t["text"]) for t in record["turns"]]})
    return replay(example, tokenizer, prompt_ids)


def build_samples(root: str | Path, tokenizer, prompt: Callable[[PresentationRequest], list[int]],
                  tasks_path: str | Path = "data/tasks/tasks.json") -> list[Sample]:
    """Все примеры SFT токенами. ``prompt(request)`` — начало эпизода (``environment.build_prompt``
    с ``prompts.system_prompt``), как при роллаутах."""
    root = Path(root)
    tasks = {t.id: t for t in load_tasks(root / tasks_path, split="sft")}
    samples = []
    for path, kind, make in ((PLAIN, "plain", plain_episode), (SEARCH, "search", search_episode)):
        for line in (root / path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            episode = make(record, prompt(request_for(record, tasks)), tokenizer)
            samples.append(Sample(record["id"], kind, record["model"], episode.ids, episode.loss_mask))
    return samples


def split(samples: list[Sample], eval_tasks: int, seed: int, eval_topics: list[str] | None = None,
          families: dict[str, str] | None = None) -> tuple[list[Sample], list[Sample]]:
    """Отложенные темы — все их примеры (без поиска и с поиском) только в оценку. ``eval_topics`` — список тем
    (иначе ``eval_tasks`` тем случайно по ``seed``); с ``families`` (id → семейство) отложены и все задания
    тех же семейств: задание «один слайд» на тему из оценки не попадает в обучение."""
    topics = sorted({s.id for s in samples})
    held = set(eval_topics) if eval_topics else set(random.Random(seed).sample(topics, eval_tasks))
    if families:
        held_families = {families[t] for t in held}
        held = {t for t in topics if families[t] in held_families}
    return [s for s in samples if s.id not in held], [s for s in samples if s.id in held]


def task_families(root: str | Path, tasks_path: str | Path = "data/tasks/tasks.json") -> dict[str, str]:
    """id задания SFT → семейство тем."""
    return {t.id: t.family for t in load_tasks(Path(root) / tasks_path, split="sft")}


def drop_long(samples: list[Sample], max_tokens: int) -> tuple[list[Sample], list[Sample]]:
    """Эпизоды не длиннее ``max_tokens`` и отброшенные (их печатает тетрадка: обрезать эпизод нельзя — оборвётся ответ)."""
    return [s for s in samples if len(s.ids) <= max_tokens], [s for s in samples if len(s.ids) > max_tokens]


def lengths(samples: list[Sample]) -> dict:
    tokens = sorted(len(s.ids) for s in samples)
    return {"examples": len(samples), "max": tokens[-1] if tokens else 0,
            "median": tokens[len(tokens) // 2] if tokens else 0, "total": sum(tokens),
            "target_total": sum(s.target_tokens for s in samples)}


# --------------------------------------------------------------------------- обучение

def collate(batch: list[Sample]) -> dict:
    """Батч из одного эпизода: метки −100 там, где маска 0 (промпт, выдача инструментов, разметка)."""
    import torch

    if len(batch) != 1:
        raise ValueError("SFT идёт батчем из одного эпизода (накопление градиента — grad_accum)")
    sample = batch[0]
    ids = torch.tensor([sample.ids])
    labels = torch.tensor([[t if m else -100 for t, m in zip(sample.ids, sample.mask)]])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def masked_loss(model, inputs: dict, num_items_in_batch=None):
    """Кросс-энтропия только по позициям с меткой: логиты считаются лишь для них (``logits_to_keep``)."""
    import torch.nn.functional as F

    labels = inputs["labels"][0]
    keep = (labels[1:] != -100).nonzero().squeeze(1)  # позиция p предсказывает токен p + 1
    out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], logits_to_keep=keep,
                use_cache=False)
    logits = out.logits[0].float()
    loss = F.cross_entropy(logits, labels[1:][keep], reduction="sum")
    denominator = num_items_in_batch if num_items_in_batch is not None else len(keep)
    return loss / denominator


ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def export_epochs(out_dir: str | Path) -> list[Path]:
    """Адаптеры эпох из чекпоинтов Trainer (``checkpoints/checkpoint-<шаг>``, по одному на эпоху) —
    в ``epochs/epoch_<k>/``; чекпоинты (с состоянием оптимизатора) после этого удаляются."""
    import shutil

    out_dir = Path(out_dir)
    checkpoints = sorted((out_dir / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    exported = []
    for epoch, checkpoint in enumerate(checkpoints, 1):
        target = out_dir / "epochs" / f"epoch_{epoch}"
        target.mkdir(parents=True, exist_ok=True)
        for name in ADAPTER_FILES:
            shutil.copy2(checkpoint / name, target / name)
        exported.append(target)
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    return exported


def train(config: SFTConfig, train_samples: list[Sample], eval_samples: list[Sample], out_dir: str | Path,
          model=None, tokenizer=None):
    """LoRA-SFT; в ``out_dir`` — лучший по loss на отложенных темах адаптер (``adapter/``), журнал (``log.json``)
    и при ``save_epochs`` — адаптеры всех эпох (``epochs/epoch_<k>/``)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    from ..model_loader import load_model

    out_dir = Path(out_dir)
    if model is None:
        model, tokenizer = load_model(config.model_id, config.revision, dtype="float16", device_map={"": 0})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=0.0,
                                             target_modules=list(TARGET_MODULES), task_type="CAUSAL_LM"))
    for parameter in model.parameters():  # адаптер — во float32 (смешанная точность на V100)
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.print_trainable_parameters()

    class MaskedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = masked_loss(model, inputs, num_items_in_batch)
            return (loss, {}) if return_outputs else loss  # при оценке нужен только loss

    # разогрев — числом шагов: warmup_ratio убран из TrainingArguments в transformers 5
    steps = math.ceil(len(train_samples) / config.grad_accum) * config.epochs
    warmup_steps = max(1, math.ceil(config.warmup_ratio * steps))
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.grad_accum, num_train_epochs=config.epochs,
        learning_rate=config.learning_rate, lr_scheduler_type="cosine", warmup_steps=warmup_steps,
        fp16=True, logging_steps=1, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=None if config.save_epochs else 1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        remove_unused_columns=False, report_to="none", seed=config.seed, dataloader_pin_memory=False,
        prediction_loss_only=True,
    )
    trainer = MaskedTrainer(model=model, args=args, train_dataset=train_samples, eval_dataset=eval_samples,
                            data_collator=collate)
    trainer.train()
    model.save_pretrained(out_dir / "adapter")
    if config.save_epochs:
        export_epochs(out_dir)
    (out_dir / "log.json").write_text(json.dumps(trainer.state.log_history, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    torch.cuda.empty_cache()
    return model
