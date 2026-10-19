# Здесь загружаем модель для генерации и даём эпизоду функцию «токены → продолжение»; загрузку для RL реализуем отдельно.
"""Модель для генерации через ``transformers``.

``hf_generator`` возвращает функцию в формате, который ждёт ``environment.run_episode``:
``generate(ids, max_new_tokens) -> (новые id, закончила ли модель реплику)``. Реплика заканчивается токеном
``<|im_end|>`` (он входит в новые id); остановка по лимиту — ``False``.
"""
from __future__ import annotations


def load_model(model_id: str, revision: str = "main", dtype: str = "float16", device_map: str = "auto"):
    """Модель и токенизатор. На V100 — float16: bfloat16 эта карта не поддерживает."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=getattr(torch, dtype), device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def hf_generator(model, tokenizer, temperature: float = 0.9, top_p: float = 1.0):
    """Функция генерации для эпизода; temperature=0 — жадная генерация."""
    import torch

    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = [end_id] + ([tokenizer.eos_token_id] if tokenizer.eos_token_id not in (None, end_id) else [])

    def generate(ids: list[int], max_new_tokens: int) -> tuple[list[int], bool]:
        if max_new_tokens <= 0:
            return [], False
        input_ids = torch.tensor([ids], device=model.device)
        sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature > 0 else {"do_sample": False}
        with torch.no_grad():
            output = model.generate(
                input_ids, attention_mask=torch.ones_like(input_ids), max_new_tokens=max_new_tokens,
                eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id or end_id, **sampling,
            )
        new = output[0, len(ids):].tolist()
        for i, token in enumerate(new):
            if token in stop_ids:
                return new[:i + 1], True
        return new, False

    return generate
