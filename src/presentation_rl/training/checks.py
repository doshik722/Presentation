# Здесь собраны проверки до длинных запусков RL: память обеих карт, совпадение log p vLLM и обучения, смена адаптеров.
"""Проверки до длинных запусков (``docs/grpo.md``, раздел 13; пункты 7 и 2). Запускаются из ``notebooks/grpo.ipynb``.

Проверки на крошечной модели (маски, накопление, градиент k2, ratio) — ``tests/test_rl_torch.py``.
"""
from __future__ import annotations

import math
import random
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .common import (Job, RLConfig, Rollout, export_adapter, load_policy, mismatch_stats, mode_task, sequence_logprobs,
                     setup_torch, stop_ids, vllm_generator)


def memory_check(config: RLConfig, root: str | Path = ".", context: int | None = None,
                 model_tokens: int | None = None) -> dict:
    """Карта 0: полный шаг (опорная модель, политика, обратный проход, создание состояния Adam) на эпизоде
    с максимальным контекстом и максимумом токенов модели. Наивное ядро внимания выключено — если экономное
    не подходит, будет ошибка, а не тихий расход памяти. Результат — пиковая память и время."""
    import torch

    from .grpo import train_step

    setup_torch()
    context = context or config.max_context
    model_tokens = min(model_tokens or config.max_new_tokens, context - 1)
    model, tokenizer = load_policy(config, root)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate,
                                  betas=tuple(config.betas), eps=config.eps, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(0)
    ids = [rng.randrange(1000, 100000) for _ in range(context)]
    mask = [0] * (context - model_tokens) + [1] * model_tokens
    rollout = Rollout("memory", "search", 0, ids, mask, [math.nan] * model_tokens, "", "", "stop", [], 0, 0.0)
    start = time.monotonic()
    result = {"context": context, "model_tokens": model_tokens}
    try:
        stats = train_step(model, optimizer, scaler, [rollout], [1.0], config)
        result.update(ok=True, peak_memory_gb=stats["peak_memory_gb"], seconds=time.monotonic() - start)
    except Exception as error:  # нехватка памяти или неподходящее ядро внимания — это и есть ответ проверки
        result.update(ok=False, error=repr(error)[:1000])
    finally:
        del model, optimizer
        torch.cuda.empty_cache()
    return result


def gpu_memory_used(index: int) -> float | None:
    """Занятая память карты по nvidia-smi (ГБ) — видит и процесс vLLM."""
    try:
        out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return float(out) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def vllm_load_check(config: RLConfig, client, embedder, parallel: int = 8, gpu: int = 1) -> dict:
    """Карта 1: пиковая нагрузка — ``parallel`` одновременных запросов длиной почти в предел контекста
    и одновременно оценка награды (эмбеддинги bge-m3) на той же карте. Пик — по nvidia-smi раз в 0,5 с."""
    peak, stop = [0.0], threading.Event()

    def watch():
        while not stop.is_set():
            used = gpu_memory_used(gpu)
            if used:
                peak[0] = max(peak[0], used)
            time.sleep(0.5)

    rng = random.Random(1)
    prompts = [[rng.randrange(1000, 100000) for _ in range(config.max_context - 2000)] for _ in range(parallel)]
    errors: list[str] = []

    def request(ids):
        try:
            client.generate(ids, 1500, config.vllm_base_name, config.temperature, 0, [])
        except Exception as error:
            errors.append(repr(error)[:300])

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    start = time.monotonic()
    workers = [threading.Thread(target=request, args=(p,)) for p in prompts]
    for w in workers:
        w.start()
    embed_error = None
    try:
        for _ in range(5):
            embedder.encode(["Тестовый слайд о фотосинтезе. " * 40] * 64, "passage")
    except Exception as error:
        embed_error = repr(error)[:300]
    for w in workers:
        w.join()
    stop.set()
    watcher.join()
    return {"parallel": parallel, "prompt_tokens": len(prompts[0]), "seconds": time.monotonic() - start,
            "peak_gpu_gb": peak[0], "request_errors": errors, "embedder_error": embed_error}


def logprob_check(config: RLConfig, root: str | Path, client, tasks: dict, topics: list[str], model=None,
                  tokenizer=None, modes: tuple = ("search", "plain")) -> dict:
    """Совпадение log p vLLM и обучающего прохода на настоящих эпизодах (адаптер с нулевой поправкой): по ним
    задаётся порог ``max_logprob_mismatch`` до основных запусков."""
    import torch

    from ..tools.cache import WikiCache
    from ..tools.wiki import WikiClient
    from ..tools.wiki_tools import WikiTools
    from .common import collect, copy_snapshot, episode_prompt
    from .grpo import wiki_contact

    if model is None:
        model, tokenizer = load_policy(config, root)
    folder = Path(root) / config.runs_dir / "checks"
    folder.mkdir(parents=True, exist_ok=True)
    name = f"check_{int(time.time())}"
    path = export_adapter(model, folder / name)
    client.load_lora(name, str(path))
    cache = WikiCache(copy_snapshot(Path(root) / config.wiki_snapshot, folder), WikiClient(contact=wiki_contact()))
    stops = stop_ids(tokenizer)
    jobs = [Job(mode_task(tasks[t], mode), mode, 0, ("check", t, mode)) for t in topics for mode in modes]
    rollouts = collect(jobs, tokenizer, episode_prompt(tokenizer),
                       lambda job: vllm_generator(client, name, config.temperature, stops, job.seed_parts),
                       lambda: WikiTools(cache, max_calls=config.max_calls), config)
    train_logp, gen_logp = [], []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for r in rollouts:
            logp, _ = sequence_logprobs(model, r.ids, r.mask, config.temperature, config.logits_chunk)
            train_logp += logp.float().cpu().tolist()
            gen_logp += r.gen_logprobs
    client.unload_lora(name)
    shutil.rmtree(path, ignore_errors=True)
    return {"episodes": len(rollouts), **mismatch_stats(train_logp, gen_logp),
            "missing_logprobs": sum(math.isnan(g) for g in gen_logp)}


def adapter_switch_check(config: RLConfig, root: str | Path, client, model, tokenizer, prompt_ids: list[int]) -> dict:
    """Смена адаптеров в vLLM A → B → A возвращает прежние вероятности: один и тот же запрос с одним сидом под A
    до и после B даёт те же токены и log p; под B (поправка с шумом) — другие."""
    import torch

    folder = Path(root) / config.runs_dir / "checks"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    a = export_adapter(model, folder / f"switch_a_{stamp}")
    with torch.no_grad():
        saved = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 1e-2)
    b = export_adapter(model, folder / f"switch_b_{stamp}")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in saved:
                p.copy_(saved[n])
    stops = stop_ids(tokenizer)
    results = []
    for name, path in ((f"switch_a_{stamp}", a), (f"switch_b_{stamp}", b), (f"switch_a2_{stamp}", a)):
        client.load_lora(name, str(path))
        results.append(client.generate(prompt_ids, 200, name, config.temperature, 12345, stops))
        client.unload_lora(name)
    shutil.rmtree(a, ignore_errors=True)
    shutil.rmtree(b, ignore_errors=True)
    same = results[0][0] == results[2][0] and all(
        (math.isnan(x) and math.isnan(y)) or abs(x - y) < 1e-4 for x, y in zip(results[0][2], results[2][2]))
    return {"a_then_a_same": same, "b_differs": results[1][0] != results[0][0] or results[1][2] != results[0][2]}
