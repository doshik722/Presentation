# Здесь подключаем GRPO поверх training/common.py: групповое преимущество, loss, шаг, цикл обучения с validation.
"""GRPO с одним обновлением на пакет (``docs/grpo.md``).

Шаг: 8 заданий (4 с поиском, 4 без) × 8 эпизодов → награда → преимущество (среднее группы, разброс шага с нижней
границей) → loss по токенам модели с k2-регуляризацией к опорной модели → одно обновление → новый адаптер в vLLM.
Каждые 10 шагов: архив адаптера, 32 эпизода validation, полное состояние для продолжения.

Запуск — ``notebooks/grpo.ipynb``; продолжение после сбоя — ``train(config, root, run_dir=<папка запуска>)``.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import statistics
import time
from pathlib import Path

from .common import (Job, Journal, RLConfig, Rollout, RewardModel, TaskQueue, add_session, check_resume, collect,
                     copy_snapshot, episode_prompt, export_adapter, fixed_validation_topics, kl_terms, load_policy,
                     mismatch_exceeded, mismatch_stats, mode_task, new_run_dir, policy_terms, save_episodes,
                     sequence_logprobs, setup_torch, start_manifest, stop_ids, summarize, target_positions,
                     validation_jobs, vllm_generator, write_json)


class RunStopped(RuntimeError):
    """Запуск остановлен по правилу протокола (нехватка памяти, расхождение log p, неконечная награда)."""


# --------------------------------------------------------------------------- преимущество

def advantages(rewards: list[float], groups: list[int], floor: float = 0.02) -> tuple[list[float], dict]:
    """A_i = (r_i − среднее группы) / max(разброс шага, floor); разброс — по исходным наградам всех эпизодов шага
    (correction = 0). Группы не выбрасываются; в журнал — доля групп с низким разбросом и точные ничьи."""
    if any(not math.isfinite(r) for r in rewards):
        raise RunStopped("неконечная награда в шаге")
    by_group: dict[int, list[float]] = {}
    for r, g in zip(rewards, groups):
        by_group.setdefault(g, []).append(r)
    means = {g: statistics.fmean(v) for g, v in by_group.items()}
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    denominator = max(std, floor)
    result = [(r - means[g]) / denominator for r, g in zip(rewards, groups)]
    stats = {"std_batch": std, "floor_used": std < floor, "abs_mean": statistics.fmean(abs(a) for a in result),
             "abs_max": max(abs(a) for a in result),
             "low_spread_groups": sum(statistics.pstdev(v) < floor for v in by_group.values()) / len(by_group),
             "tied_groups": sum(max(v) - min(v) < 1e-6 for v in by_group.values())}
    return result, stats


# --------------------------------------------------------------------------- шаг обучения

def train_step(model, optimizer, scaler, rollouts: list[Rollout], advantage: list[float], config: RLConfig) -> dict:
    """Одно обновление по всем эпизодам шага: loss = (1/N) Σ_токенов [−A·ρ + β·k2], N — число токенов модели в шаге;
    градиенты копятся по одному эпизоду, затем unscale → отсечение → step → update. Нехватка памяти — обновление
    не делается, запуск останавливается (эпизоды не пропускаются)."""
    import torch

    device = next(model.parameters()).device
    cuda = device.type == "cuda"

    def autocast():  # на карте — float16 (V100 без bfloat16); на процессоре (тесты) — без смешанной точности
        return torch.autocast(device.type, dtype=torch.float16, enabled=cuda)

    total = sum(len(target_positions(r.mask)) for r in rollouts)
    trainable = [p for p in model.parameters() if p.requires_grad]
    T, chunk = config.temperature, config.logits_chunk
    sums = {"loss": 0.0, "pg": 0.0, "k2": 0.0, "k3": 0.0, "entropy": 0.0, "clip": 0.0}
    train_logp: list[float] = []
    gen_logp: list[float] = []
    kept: list[tuple[Rollout, "torch.Tensor"]] = []
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    try:
        for index, (rollout, a) in enumerate(zip(rollouts, advantage)):
            model.eval()
            with torch.no_grad(), model.disable_adapter(), autocast():
                ref, _ = sequence_logprobs(model, rollout.ids, rollout.mask, T, chunk)
            model.train()
            with autocast():
                logp, entropy = sequence_logprobs(model, rollout.ids, rollout.mask, T, chunk)
            pg, clip_fraction = policy_terms(logp, float(a), None, config.clip_eps)
            k2, k3 = kl_terms(logp, ref)
            loss = (pg + config.kl_beta * k2).sum() / total
            scaler.scale(loss).backward()
            values = logp.detach()
            sums["loss"] += float(loss.detach())
            sums["pg"] += float(pg.detach().sum())
            sums["k2"] += float(k2.detach().sum())
            sums["k3"] += float(k3.sum())
            sums["entropy"] += float(entropy.sum())
            sums["clip"] += float(clip_fraction) * len(values)
            train_logp += values.float().cpu().tolist()
            gen_logp += rollout.gen_logprobs
            if index < config.kl_old_new_episodes:
                kept.append((rollout, values))
            del ref, logp, entropy, pg, k2, k3, loss
    except torch.cuda.OutOfMemoryError as error:
        optimizer.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.empty_cache()
        raise RunStopped(f"нехватка памяти на эпизоде {rollout.mode}_{rollout.task_id}_{rollout.sample} "
                         f"({len(rollout.ids)} токенов): уменьшить предел контекста и начать новый запуск") from error

    scaler.unscale_(optimizer)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip))
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < scale  # GradScaler пропустил обновление (inf/nan в градиентах)
    optimizer.zero_grad(set_to_none=True)

    kl_old_new = {"k1": [], "k3": []}  # KL(old‖new) на токенах старой политики: насколько изменилась модель за шаг
    model.eval()
    with torch.no_grad(), autocast():
        for rollout, old in kept:
            new, _ = sequence_logprobs(model, rollout.ids, rollout.mask, T, chunk)
            d = new - old
            kl_old_new["k1"].append(float((-d).mean()))
            kl_old_new["k3"].append(float((torch.exp(d) - d - 1).mean()))
    model.train()

    return {"tokens": total, "loss": sums["loss"], "pg_per_token": sums["pg"] / total,
            "k2_per_token": sums["k2"] / total, "kl_ref_k3": sums["k3"] / total, "entropy": sums["entropy"] / total,
            "clip_fraction": sums["clip"] / total, "grad_norm": grad_norm, "grad_scale": scale, "skipped": skipped,
            "kl_old_new_k1": statistics.fmean(kl_old_new["k1"]) if kept else None,
            "kl_old_new_k3": statistics.fmean(kl_old_new["k3"]) if kept else None,
            "logprob_mismatch": mismatch_stats(train_logp, gen_logp),
            "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None}


# --------------------------------------------------------------------------- состояние

def save_state(run_dir: Path, step: int, model, optimizer, scaler, queue: TaskQueue, counters: dict,
               keep: int) -> Path:
    """Полное состояние для продолжения на границе шага; хранятся ``keep`` последних."""
    import torch

    states = run_dir / "state"
    states.mkdir(exist_ok=True)
    folder = states / f"step_{step:04d}"
    tmp = states / f"step_{step:04d}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    model.save_pretrained(tmp / "adapter")
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "python_rng": random.getstate()}, tmp / "training.pt")
    write_json(tmp / "state.json", {"step": step, "queue": queue.state(), "counters": counters})
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(tmp, folder)
    for old in sorted(p for p in states.glob("step_*") if not p.name.endswith(".tmp"))[:-keep]:
        shutil.rmtree(old)
    return folder


def latest_state(run_dir: Path) -> Path | None:
    done = sorted(p for p in (run_dir / "state").glob("step_*") if (p / "state.json").exists())
    return done[-1] if done else None


def cancel_after(run_dir: Path, step: int, session: int) -> None:
    """При продолжении чекпоинты шагов после восстановленной точки (отменённая история) переименовываются."""
    for folder in sorted((run_dir / "checkpoints").glob("step_*")):
        if int(folder.name.split("_")[1]) > step:
            os.replace(folder, folder.with_name(f"cancelled_{folder.name}_s{session}"))


# --------------------------------------------------------------------------- validation

def validate(client, model_name: str, tasks: dict, topics: list[str], samples: int, tokenizer, cache,
             reward: RewardModel, config: RLConfig, folder: Path) -> list[Rollout]:
    """Ответы validation одной модели (адаптера) с сидами без номера шага, награда и аналитика; эпизоды
    сохраняются в ``folder`` (их читает судья)."""
    from ..tools.wiki_tools import WikiTools

    jobs = [Job(mode_task(tasks[topic], mode), mode, s, ("validation", topic, mode, s))
            for mode, topic, s in validation_jobs(topics, samples)]
    prompt, stops = episode_prompt(tokenizer), stop_ids(tokenizer)
    rollouts = collect(jobs, tokenizer, prompt,
                       lambda job: vllm_generator(client, model_name, config.temperature, stops, job.seed_parts),
                       lambda: WikiTools(cache, max_calls=config.max_calls), config)
    for job, rollout in zip(jobs, rollouts):
        reward(job.task, rollout)
    save_episodes(folder, rollouts)
    return rollouts


def generate_reference(config: RLConfig, root: str | Path = ".", client=None, reward: RewardModel | None = None,
                       samples: int = 2) -> Path:
    """Ответы исходной модели на весь validation (25 тем × 2 режима × ``samples``) — один раз для всех запусков:
    с ними судья сравнивает чекпоинты (малый validation — их часть, полный — ответы с номером 0)."""
    from transformers import AutoTokenizer

    from ..tasks import load_tasks
    from ..tools.cache import WikiCache
    from ..tools.wiki import WikiClient
    from .vllm_client import VLLMClient

    root = Path(root).resolve()
    folder = root / config.runs_dir / "reference"
    folder.mkdir(parents=True, exist_ok=True)
    tasks = {t.id: t for t in load_tasks(root / "data" / "tasks" / "tasks.json", split="validation")}
    tokenizer = AutoTokenizer.from_pretrained(config.policy_model, revision=config.policy_revision)
    client = client or VLLMClient(config.vllm_url)
    reward = reward or RewardModel(root, device=config.embedder_device, floor=config.reward_floor)
    cache = WikiCache(copy_snapshot(root / config.wiki_snapshot, folder), WikiClient(contact=wiki_contact()))
    rollouts = validate(client, config.vllm_base_name, tasks, sorted(tasks), samples, tokenizer, cache, reward,
                        config, folder / "episodes")
    write_json(folder / "summary.json", {"model": config.vllm_base_name, "policy_model": config.policy_model,
                                         "samples": samples, "summary": summarize(rollouts),
                                         "network_seconds": cache.network_seconds})
    return folder


def wiki_contact() -> str:
    return os.environ.get("WIKI_CONTACT", "https://github.com/doshik722")


# --------------------------------------------------------------------------- цикл обучения

def train(config: RLConfig, root: str | Path = ".", run_dir: str | Path | None = None, client=None,
          reward: RewardModel | None = None) -> Path:
    """Новый запуск (``run_dir=None``) или продолжение запуска из его папки с последнего полного состояния."""
    import torch

    from ..tasks import load_tasks
    from ..tools.cache import WikiCache
    from ..tools.wiki import WikiClient
    from ..tools.wiki_tools import WikiTools
    from .vllm_client import VLLMClient

    root = Path(root).resolve()
    setup_torch()
    all_tasks = {t.id: t for t in load_tasks(root / "data" / "tasks" / "tasks.json")}
    train_ids = sorted(k for k, t in all_tasks.items() if t.split == "train")
    val_topics = fixed_validation_topics([k for k, t in all_tasks.items() if t.split == "validation"],
                                         config.val_topics)
    client = client or VLLMClient(config.vllm_url)

    if run_dir is None:
        run_dir = new_run_dir(config, root)
        manifest = start_manifest(run_dir, config, root, {"validation_topics": val_topics,
                                                          "vllm_version": client.version()})
        state_dir, start = None, 0
    else:
        run_dir = Path(run_dir)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        check_resume(manifest, config, root)
        state_dir = latest_state(run_dir)
        start = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))["step"] if state_dir else 0
    session = add_session(run_dir, manifest, start, root, {"vllm_version": client.version(),
                                                           "wiki_proxy": os.environ.get("WIKI_PROXY", "")})
    journals = {name: Journal(run_dir / f"{name}.jsonl", session) for name in ("steps", "validation", "events")}
    if session > 1:
        for journal in journals.values():
            journal.resume(start)
        cancel_after(run_dir, start, session)

    cache = WikiCache(copy_snapshot(root / config.wiki_snapshot, run_dir), WikiClient(contact=wiki_contact()))
    model, tokenizer = load_policy(config, root, adapter=state_dir / "adapter" if state_dir else None)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, betas=tuple(config.betas), eps=config.eps,
                                  weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    queue = TaskQueue(train_ids, config.tasks_per_mode, config.seed)
    counters = {"updates": 0, "skipped": 0, "episodes": 0, "generated_tokens": 0, "train_tokens": 0,
                "seconds": 0.0, "network_seconds": 0.0}
    if state_dir:
        saved = torch.load(state_dir / "training.pt", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
        queue.load_state(state["queue"])
        counters.update(state["counters"])

    reward = reward or RewardModel(root, device=config.embedder_device, floor=config.reward_floor)
    prompt, stops = episode_prompt(tokenizer), stop_ids(tokenizer)
    adapters = run_dir / "adapters"
    adapters.mkdir(exist_ok=True)

    def publish(step: int, previous: str | None) -> str:
        """Адаптер шага → vLLM (атомарная запись, загрузка, подтверждение), прежний выгружается."""
        name = f"{run_dir.name}_s{session}_step{step}"
        path = export_adapter(model, adapters / f"step_{step:04d}")
        client.load_lora(name, str(path))
        if previous:
            client.unload_lora(previous)
        for old in adapters.glob("step_*"):
            if old != path and not old.name.endswith(".tmp"):
                shutil.rmtree(old, ignore_errors=True)
        return name

    for name in client.models():  # адаптеры прерванной сессии этого запуска, оставшиеся на сервере
        if name.startswith(run_dir.name + "_"):
            client.unload_lora(name)
    current = publish(start, None)
    journals["events"].write({"type": "start", "step": start, "run": run_dir.name})
    step = start
    try:
        for step in range(start + 1, config.steps + 1):
            began = time.monotonic()
            network_before = cache.network_seconds
            batch = queue.next()
            jobs = [Job(mode_task(all_tasks[task_id], mode), mode, s, (config.seed, "train", step, task_id, mode, s))
                    for mode, task_id in batch for s in range(config.group_size)]
            rollouts = collect(jobs, tokenizer, prompt,
                               lambda job: vllm_generator(client, current, config.temperature, stops, job.seed_parts),
                               lambda: WikiTools(cache, max_calls=config.max_calls), config)
            generated = time.monotonic()
            for job, rollout in zip(jobs, rollouts):
                reward(job.task, rollout)
            rewarded = time.monotonic()
            groups = [batch.index((r.mode, r.task_id)) for r in rollouts]
            advantage, advantage_stats = advantages([r.reward for r in rollouts], groups, config.std_floor)
            stats = train_step(model, optimizer, scaler, rollouts, advantage, config)
            trained = time.monotonic()
            problem = mismatch_exceeded(stats["logprob_mismatch"], config.max_logprob_mismatch or {})
            if problem:
                raise RunStopped(problem)
            current = publish(step, current)
            published = time.monotonic()

            counters["updates"] += not stats["skipped"]
            counters["skipped"] += stats["skipped"]
            counters["episodes"] += len(rollouts)
            counters["generated_tokens"] += sum(r.model_tokens for r in rollouts)
            counters["train_tokens"] += stats["tokens"]
            counters["seconds"] += published - began
            counters["network_seconds"] += cache.network_seconds - network_before
            journals["steps"].write({
                "step": step, "tasks": batch, "advantage": advantage_stats, **stats, **summarize(rollouts),
                "seconds": {"generation": generated - began, "reward": rewarded - generated, "train": trained - rewarded,
                            "adapter": published - trained, "network": cache.network_seconds - network_before},
                "counters": dict(counters), "queue": queue.state(),
            })
            if counters["skipped"] > config.max_skipped_updates * step:
                journals["events"].write({"type": "warning", "step": step,
                                          "message": f"пропущено обновлений {counters['skipped']} из {step}"})

            if step % config.eval_every == 0 or step == config.steps:
                folder = run_dir / "checkpoints" / f"step_{step:04d}"
                if folder.exists():
                    shutil.rmtree(folder)
                folder.mkdir(parents=True)
                shutil.copytree(adapters / f"step_{step:04d}", folder / "adapter")
                val = validate(client, current, all_tasks, val_topics, config.val_samples, tokenizer, cache, reward,
                               config, folder / "validation")
                journals["validation"].write({"step": step, **summarize(val)})
                write_json(folder / "done.json", {"step": step, "session": session,
                                                  "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
                save_state(run_dir, step, model, optimizer, scaler, queue, counters, config.keep_states)
    except BaseException as error:
        journals["events"].write({"type": "stopped", "at_step": step, "error": repr(error)[:2000]})
        raise
    journals["events"].write({"type": "finished", "step": config.steps})
    return run_dir
