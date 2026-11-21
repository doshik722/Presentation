# Здесь подключаем PPO поверх training/common.py: разогрев критиков, GAE, два прохода политики и цикл.
"""PPO с критиком на префиксных признаках или скрытых состояниях опорной T-lite.

Шаг: 8 заданий × 8 эпизодов (как у GRPO) → подготовка награды, признаков и опорных log p → V_old, преимущества
GAE (γ = λ = 1: A = R − V_old), нормировка по токенам шага → обучение критика → проход 1 политики
(ρ = 1, старые log p сохраняются) → шаг оптимизатора → проход 2 (отсечение ε, KL k2 с весом sg(ρ)) →
шаг оптимизатора → адаптер в vLLM.

Разогрев (до основного запуска, у каждого сида свой): эпизоды исходной модели из отдельной очереди, признаки
в случайной выборке позиций, регрессия головы на итоговую награду. Выбор вида головы и числа эпох — EXP-028
(``notebooks/ppo_exp028_v2.ipynb``). Части без torch (GAE, нормировка, меры критика, выборка позиций, деление по темам)
проверяются в ``tests/test_ppo.py``; части с torch — в ``tests/test_ppo_torch.py`` на крошечной модели (на сервере).

Общий код (``common.py``, ``grpo.py``) не меняется: его хеши проверяет продолжение уже идущих запусков GRPO.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import shutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .common import (Job, Journal, RLConfig, Rollout, RewardModel, TaskQueue, add_session, check_resume, collect,
                     copy_snapshot, episode_prompt, export_adapter, file_hash, fixed_validation_topics, kl_terms,
                      load_policy, mismatch_exceeded, mismatch_stats, mode_task, new_run_dir, read_journal, save_episodes,
                     sequence_logprobs, setup_torch, stable_seed, start_manifest, stop_ids, summarize, target_positions,
                     vllm_generator, write_json)
from .grpo import RunStopped, cancel_after, fmt, validate, wiki_contact

POSITIONS = ("first", "q25", "q50", "q75", "last")  # описательные срезы ответа (docs/ppo.md, раздел 16)


# --------------------------------------------------------------------------- конфиг

@dataclass
class PPOConfig(RLConfig):
    """Параметры PPO (``configs/algorithms/ppo.yaml``): общие — как у GRPO, плюс критик, разогрев и проходы."""

    algorithm: str = "ppo"
    runs_dir: str = "runs/ppo"
    ppo_epochs: int = 2  # μ: проходов по пакету, один шаг оптимизатора на проход
    gae_lambda: float = 1.0  # решение 18.11.2026: λ = 1 у обоих сидов
    # голова критика (docs/ppo.md, разделы 4, 8)
    head: str = "linear"  # linear | mlp
    head_hidden: int = 256
    head_lr: float = 1e-4
    head_betas: tuple = (0.9, 0.999)
    head_eps: float = 1e-8
    head_weight_decay: float = 1e-4
    head_batch: int = 1024
    head_epochs_step: int = 1
    head_grad_clip: float = 1.0
    head_device: str = "cuda:0"
    # разогрев (docs/ppo.md, раздел 5)
    warmup_steps: int = 10  # × 64 эпизода
    warmup_max_positions: int = 200_000
    warmup_max_epochs: int = 10
    warmup_patience: int = 2
    warmup_holdout: float = 0.2
    warmup_epochs: int | None = None  # выбранное в EXP-028 число эпох (None — выбор по отложенным темам)
    warmup_dir: str | None = None  # папка готового разогрева этого сида (для основного запуска)

    @classmethod
    def load(cls, path: str | Path, **overrides) -> "PPOConfig":
        config = super().load(path, **overrides)
        config.head_betas = tuple(config.head_betas)
        return config

    def to_json(self) -> dict:
        data = super().to_json()
        data["head_betas"] = list(self.head_betas)
        return data


# --------------------------------------------------------------------------- GAE, нормировка (без torch)

def gae(values: list[float], reward: float, lam: float) -> tuple[list[float], list[float]]:
    """Преимущества и цели по действиям эпизода (γ = 1): награда — на последнем действии, после него V = 0.
    δ_t = V_{t+1} − V_t (δ_T = R − V_T), A_t = δ_t + λ·A_{t+1}; G_t = A_t + V_t. При λ = 1: A_t = R − V_t, G_t = R."""
    n = len(values)
    advantages = [0.0] * n
    following = 0.0
    for t in range(n - 1, -1, -1):
        delta = (reward if t == n - 1 else values[t + 1]) - values[t]
        following = delta + lam * following
        advantages[t] = following
    return advantages, [a + v for a, v in zip(advantages, values)]


def normalize(advantages: list[list[float]], floor: float) -> tuple[list[list[float]], dict]:
    """(A − среднее) / max(std, floor) по всем действиям шага (std с correction = 0). Вычитание среднего может менять
    знаки, порядок сохраняется (``subtleties.md``)."""
    flat = [a for episode in advantages for a in episode]
    if not flat or any(not math.isfinite(a) for a in flat):
        raise RunStopped("неконечные или пустые преимущества шага")
    mean = statistics.fmean(flat)
    std = statistics.pstdev(flat) if len(flat) > 1 else 0.0
    denominator = max(std, floor)
    result = [[(a - mean) / denominator for a in episode] for episode in advantages]
    stats = {"raw_mean": mean, "raw_std": std, "floor_used": std < floor, "raw_min": min(flat), "raw_max": max(flat),
             "positive_share": sum(a > 0 for a in flat) / len(flat)}
    return result, stats


# --------------------------------------------------------------------------- меры критика (без torch)

def slice_index(n: int, name: str) -> int:
    """Номер действия среза ответа из ``n`` действий: первое, 25/50/75% и последнее."""
    return {"first": 0, "q25": int(0.25 * (n - 1)), "q50": int(0.5 * (n - 1)), "q75": int(0.75 * (n - 1)),
            "last": n - 1}[name]


def _ev(errors: list[float], targets: list[float]) -> float | None:
    if len(targets) < 2:
        return None
    var = statistics.pvariance(targets)
    return None if var <= 0 else 1 - statistics.pvariance(errors) / var


def critic_metrics(values: list[list[float]], rewards: list[float], modes: list[str], constant: float,
                   mode_means: dict | None = None) -> dict:
    """Меры критика на пакете (``docs/ppo.md``, разделы 14, 16): MSE по всем действиям (как обучающая потеря),
    по срезам ответа и режимам; смещение V − R; EV; то же для константы и средних по режимам. ``values`` —
    V по действиям каждого эпизода (полные или выборка), ``rewards`` — R эпизодов."""
    out: dict = {}

    def block(prefix: str, rows: list[tuple[float, float]]):
        if not rows:
            return
        errors = [r - v for v, r in rows]
        out[f"{prefix}_mse"] = statistics.fmean(e * e for e in errors)
        out[f"{prefix}_bias"] = statistics.fmean(-e for e in errors)
        out[f"{prefix}_ev"] = _ev(errors, [r for _, r in rows])

    pairs = [(v, r) for vs, r in zip(values, rewards) for v in vs]
    block("all", pairs)
    out["all_mse_const"] = statistics.fmean((r - constant) ** 2 for _, r in pairs) if pairs else None
    if mode_means and pairs:
        out["all_mse_mode"] = sum(len(vs) * (r - mode_means[m]) ** 2
                                  for vs, r, m in zip(values, rewards, modes)) / len(pairs)
    for mode in sorted(set(modes)):
        rows = [(v, r) for vs, r, m in zip(values, rewards, modes) if m == mode for v in vs]
        block(f"{mode}_all", rows)
        out[f"{mode}_all_mse_const"] = statistics.fmean((r - constant) ** 2 for _, r in rows) if rows else None
    for name in POSITIONS:
        rows = [(vs[slice_index(len(vs), name)], r) for vs, r in zip(values, rewards) if vs]
        block(name, rows)
        out[f"{name}_mse_const"] = statistics.fmean((r - constant) ** 2 for _, r in rows) if rows else None
        if mode_means:
            out[f"{name}_mse_mode"] = statistics.fmean((r - mode_means[m]) ** 2
                                                       for vs, r, m in zip(values, rewards, modes) if vs)
    return out


def paired_improvement(rows: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Основная мера EXP-028: d = (R − c)² − (R − V)², усреднённая внутри эпизода, затем внутри темы и режима;
    режимы — с весом по половине; 95% интервал — парный бутстреп по темам (тема несёт оба режима).
    ``rows``: {"task", "mode", "d"} — d эпизода (среднее по его позициям)."""
    by_topic: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        by_topic.setdefault(r["task"], {}).setdefault(r["mode"], []).append(r["d"])
    topics = sorted(by_topic)
    modes = sorted({r["mode"] for r in rows})

    def estimate(sample: list[str]) -> float:
        per_mode = []
        for mode in modes:
            means = [statistics.fmean(by_topic[t][mode]) for t in sample if mode in by_topic[t]]
            if means:
                per_mode.append(statistics.fmean(means))
        return statistics.fmean(per_mode)

    rng = random.Random(seed)
    boot = sorted(estimate([rng.choice(topics) for _ in topics]) for _ in range(n_boot))
    return {"estimate": estimate(topics), "low": boot[int(0.025 * n_boot)], "high": boot[int(0.975 * n_boot)],
            "topics": len(topics)}


def exp028_verdict(metrics: dict) -> str:
    """Правило EXP-028 (``docs/ppo.md``, раздел 16): «пригоден» — интервал основной меры целиком выше нуля и ни
    в одном режиме и ни в одном срезе точечная MSE критика не хуже константы; «не пригоден» — интервал целиком
    ниже нуля; иначе — «данных недостаточно»."""
    paired = metrics["paired"]
    if paired["high"] < 0:
        return "не пригоден"
    if paired["low"] <= 0:
        return "данных недостаточно"
    keys = [k[:-len("_mse_const")] for k in metrics if k.endswith("_mse_const")]
    worse = [k for k in keys if metrics.get(f"{k}_mse") is not None and metrics[f"{k}_mse_const"] is not None
             and metrics[f"{k}_mse"] > metrics[f"{k}_mse_const"]]
    return "пригоден" if not worse else "данных недостаточно (хуже константы: " + ", ".join(sorted(worse)) + ")"


def leave_one_out_means(rewards: list[float], groups: list) -> list[float | None]:
    """Среднее награды других ответов того же задания (точка сравнения GRPO, справочно)."""
    by_group: dict = {}
    for r, g in zip(rewards, groups):
        by_group.setdefault(g, []).append(r)
    out = []
    for r, g in zip(rewards, groups):
        others = len(by_group[g]) - 1
        out.append((sum(by_group[g]) - r) / others if others else None)
    return out


# --------------------------------------------------------------------------- разогрев: выборка позиций, деление

def sample_positions(counts: list[int], cap: int, seed: int) -> list[list[int]]:
    """Номера действий для буфера разогрева: равномерная выборка среди всех действий (около ``cap``), первое
    и последнее действие эпизода — всегда (на них меры EXP-028). Детерминирована по сиду."""
    total = sum(counts)
    keep = 1.0 if total <= cap else cap / total
    rng = random.Random(seed)
    result = []
    for n in counts:
        chosen = {k for k in range(n) if keep >= 1.0 or rng.random() < keep}
        if n:
            chosen |= {0, n - 1}
        result.append(sorted(chosen))
    return result


def split_topics(task_ids: list[str], holdout: float, seed: int) -> set[str]:
    """Темы отложенной части (все эпизоды темы — в одной части, оба режима)."""
    topics = sorted(set(task_ids))
    rng = random.Random(seed)
    rng.shuffle(topics)
    return set(topics[:max(1, round(holdout * len(topics)))])


# --------------------------------------------------------------------------- torch: голова, проход 0, шаг

def make_head(kind: str, dim: int, hidden: int, init: float):
    """Голова критика во float32: линейная d → 1 или MLP d → hidden → 1 (GELU); последний слой — нули, сдвиг — ``init``."""
    import torch

    if kind == "linear":
        head = torch.nn.Linear(dim, 1)
        last = head
    elif kind == "mlp":
        head = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1))
        last = head[2]
    else:
        raise ValueError(f"неизвестная голова критика: {kind}")
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(init)
    return head.float()


def head_optimizer(head, config: PPOConfig):
    """AdamW головы: weight decay — только на матрицы весов, сдвиги без него."""
    import torch

    decay = [p for n, p in head.named_parameters() if n.endswith("weight")]
    plain = [p for n, p in head.named_parameters() if not n.endswith("weight")]
    return torch.optim.AdamW([{"params": decay, "weight_decay": config.head_weight_decay},
                              {"params": plain, "weight_decay": 0.0}],
                             lr=config.head_lr, betas=tuple(config.head_betas), eps=config.head_eps)


def head_values(head, features, device: str, chunk: int = 4096) -> list[float]:
    """V по признакам (CPU, float16): кусками на ``device`` во float32, без градиента."""
    import torch

    out = []
    head.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], chunk):
            part = features[start:start + chunk].to(device, non_blocking=True).float()
            out.append(head(part)[:, 0].float().cpu())
    return torch.cat(out).tolist() if out else []


def reference_pass(model, ids: list[int], mask: list[int], temperature: float, chunk: int = 1024,
                   keep: list[int] | None = None, logp: bool = True, *, return_features: bool = True):
    """Проход 0 по эпизоду: опорная модель (адаптер выключен, ``eval``, без градиента). Возвращает log p опорной
    модели в позициях действий (на процессоре, float32; None при ``logp=False``) и признаки критика —
    ``last_hidden_state`` в позиции p − 1 для действия в позиции p (на процессоре, float16; ``keep`` — номера
    действий, иначе все). При ``return_features=False`` признаки не копируются на CPU. Режим модели восстанавливается."""
    import torch

    from .common import _chunk_logprobs, backbone_and_head

    device = next(model.parameters()).device
    positions_all = target_positions(mask)
    if not positions_all:
        return (torch.zeros(0) if logp else None), (torch.zeros((0, 0), dtype=torch.float16) if return_features else None)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), model.disable_adapter(), torch.autocast(device.type, dtype=torch.float16,
                                                                       enabled=device.type == "cuda"):
            body, head = backbone_and_head(model)
            input_ids = torch.tensor([ids], device=device)
            hidden = body(input_ids=input_ids, use_cache=False).last_hidden_state[0]
            positions = torch.tensor([p - 1 for p in positions_all], device=device)
            states = hidden.index_select(0, positions)
            del hidden
            ref = None
            if logp:
                targets = input_ids[0].index_select(0, positions + 1)
                parts = [_chunk_logprobs(states[s:s + chunk], targets[s:s + chunk], head.weight, temperature)[0]
                         for s in range(0, len(positions_all), chunk)]
                ref = torch.cat(parts).float().cpu()
            features = None
            if return_features:
                chosen = states if keep is None else states.index_select(0, torch.tensor(keep, device=device))
                features = chosen.to(torch.float16).cpu()
    finally:
        model.train(was_training)
    return ref, features


def train_head(head, optimizer, features: list, targets: list[list[float]], config: PPOConfig, epochs: int,
               seed: int) -> dict:
    """Эпохи регрессии головы (½·MSE) по буферу: позиции перемешиваются по всем эпизодам (не подряд из одного),
    мини-пакеты по ``head_batch``; признаки — с процессора, во float32 — только очередной мини-пакет."""
    import torch

    index = [(e, k) for e, f in enumerate(features) for k in range(f.shape[0])]
    rng = random.Random(seed)
    losses, norms = [], []
    head.train()
    for epoch in range(epochs):
        rng.shuffle(index)
        for start in range(0, len(index), config.head_batch):
            by_episode: dict[int, list[int]] = {}
            for e, k in index[start:start + config.head_batch]:
                by_episode.setdefault(e, []).append(k)
            rows, goal = [], []
            for e, ks in by_episode.items():
                rows.append(features[e].index_select(0, torch.tensor(ks)))
                goal += [targets[e][k] for k in ks]
            x = torch.cat(rows).to(config.head_device).float()
            y = torch.tensor(goal, device=config.head_device, dtype=torch.float32)
            loss = 0.5 * ((head(x)[:, 0] - y) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norms.append(float(torch.nn.utils.clip_grad_norm_(head.parameters(), config.head_grad_clip)))
            optimizer.step()
            losses.append(float(loss.detach()))
    head.eval()
    return {"head_loss": statistics.fmean(losses) if losses else None,
            "head_grad_norm": statistics.fmean(norms) if norms else None, "head_updates": len(losses)}


def ppo_policy_terms(logp, advantage, old_logp, clip_eps: float):
    """−min(ρA, clip(ρ)A) по токенам; доли: ρ вне диапазона и токены, где отсечённая ветвь действительно
    ограничивает цель."""
    import torch

    ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    unclipped_obj, clipped_obj = ratio * advantage, clipped * advantage
    terms = -torch.minimum(unclipped_obj, clipped_obj)
    outside = ((ratio - 1).abs() > clip_eps).float().mean()
    active = (clipped_obj < unclipped_obj).float().mean()
    return terms, ratio.detach(), outside.detach(), active.detach()


def ppo_train_step(model, optimizer, scaler, rollouts: list[Rollout], advantages: list[list[float]], ref_logps: list,
                   config: PPOConfig) -> dict:
    """μ проходов по пакету, один шаг оптимизатора на проход (градиенты копятся по одному эпизоду). Проход 1:
    ρ = 1, старые log p сохраняются (без графа); проходы 2…μ: отсечение ε и KL k2 с весом sg(ρ).
    loss = (1/N) Σ [−min(ρA, clip(ρ)A) + sg(ρ)·β·k2], N — фактическое число действий шага."""
    import torch

    device = next(model.parameters()).device
    cuda = device.type == "cuda"

    def autocast():
        return torch.autocast(device.type, dtype=torch.float16, enabled=cuda)

    total = sum(len(target_positions(r.mask)) for r in rollouts)
    trainable = [p for p in model.parameters() if p.requires_grad]
    T, chunk = config.temperature, config.logits_chunk
    old: list = []
    passes = []
    train_logp: list[float] = []
    gen_logp: list[float] = []
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for number in range(1, config.ppo_epochs + 1):
        sums = {"loss": 0.0, "pg": 0.0, "k2": 0.0, "k3": 0.0, "entropy": 0.0, "outside": 0.0, "active": 0.0}
        ratios = []
        optimizer.zero_grad(set_to_none=True)
        rollout = None
        try:
            for index, rollout in enumerate(rollouts):
                with autocast():
                    logp, entropy = sequence_logprobs(model, rollout.ids, rollout.mask, T, chunk)
                a = torch.tensor(advantages[index], device=device, dtype=torch.float32)
                ref = ref_logps[index].to(device)
                if number == 1:
                    previous = logp.detach()
                    old.append(previous.float().cpu())
                    train_logp += previous.float().cpu().tolist()
                    gen_logp += rollout.gen_logprobs
                else:
                    previous = old[index].to(device)
                pg, ratio, outside, active = ppo_policy_terms(logp, a, previous, config.clip_eps)
                k2, k3 = kl_terms(logp, ref)
                weight = ratio if number > 1 else torch.ones_like(ratio)
                loss = (pg + config.kl_beta * weight * k2).sum() / total
                scaler.scale(loss).backward()
                count = len(ratio)
                sums["loss"] += float(loss.detach())
                sums["pg"] += float(pg.detach().sum())
                sums["k2"] += float((weight * k2.detach()).sum())
                sums["k3"] += float(k3.sum())
                sums["entropy"] += float(entropy.sum())
                sums["outside"] += float(outside) * count
                sums["active"] += float(active) * count
                if number > 1:
                    ratios.append(ratio.float().cpu())
                del logp, entropy, pg, k2, k3, loss, ratio, ref, previous
        except torch.cuda.OutOfMemoryError as error:
            optimizer.zero_grad(set_to_none=True)
            if cuda:
                torch.cuda.empty_cache()
            raise RunStopped(f"нехватка памяти, проход {number}, эпизод {rollout.mode}_{rollout.task_id}_"
                             f"{rollout.sample} ({len(rollout.ids)} токенов): уменьшить предел контекста "
                             f"и начать новый запуск") from error
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, config.grad_clip))
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < scale
        optimizer.zero_grad(set_to_none=True)
        record = {"pass": number, "loss": sums["loss"], "pg_per_token": sums["pg"] / total,
                  "k2_per_token": sums["k2"] / total, "entropy": sums["entropy"] / total,
                  "ratio_outside": sums["outside"] / total, "clip_active": sums["active"] / total,
                  "grad_norm": grad_norm, "grad_scale": scale, "skipped": skipped}
        if number == 1:
            record["kl_ref_k3_before"] = sums["k3"] / total  # KL к опорной до обновлений
        if ratios:
            r = torch.cat(ratios)
            record.update(ratio_max=float(r.max()), ratio_min=float(r.min()),
                          ratio_q99=float(torch.quantile(r, 0.99)) if len(r) else None)
        passes.append(record)

    kl_old_new = {"k1": [], "k3": []}  # KL(old‖new) после всех проходов: exp(d) − 1 − d, d = log π_new − log π_old
    model.eval()
    with torch.no_grad(), autocast():
        for index, rollout in enumerate(rollouts[:config.kl_old_new_episodes]):
            new, _ = sequence_logprobs(model, rollout.ids, rollout.mask, T, chunk)
            d = new.float().cpu() - old[index]
            kl_old_new["k1"].append(float((-d).mean()))
            kl_old_new["k3"].append(float((torch.exp(d) - d - 1).mean()))
    model.train()
    return {"tokens": total, "passes": passes, "kl_ref_k3": passes[0]["kl_ref_k3_before"],
            "updates": sum(not p["skipped"] for p in passes), "skipped": sum(p["skipped"] for p in passes),
            "kl_old_new_k1": statistics.fmean(kl_old_new["k1"]) if kl_old_new["k1"] else None,
            "kl_old_new_k3": statistics.fmean(kl_old_new["k3"]) if kl_old_new["k3"] else None,
            "logprob_mismatch": mismatch_stats(train_logp, gen_logp),
            "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None}


# --------------------------------------------------------------------------- разогрев

def warmup_generate(config: PPOConfig, root: str | Path, client=None, reward: RewardModel | None = None) -> Path:
    """Эпизоды разогрева сида: исходная модель (``vllm_base_name``), очередь со своим сидом, ``warmup_steps`` × 64.
    Сохраняются ids, маска, награда, задание и режим (``episodes.jsonl``) и тексты (``episodes/``)."""
    from transformers import AutoTokenizer

    from ..tasks import load_tasks
    from ..tools.cache import WikiCache
    from ..tools.wiki import WikiClient
    from ..tools.wiki_tools import WikiTools
    from .vllm_client import VLLMClient

    root = Path(root).resolve()
    folder = root / config.runs_dir / f"warmup_seed{config.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    tasks = {t.id: t for t in load_tasks(root / "data" / "tasks" / "tasks.json")}
    train_ids = sorted(k for k, t in tasks.items() if t.split == "train")
    tokenizer = AutoTokenizer.from_pretrained(config.policy_model, revision=config.policy_revision)
    client = client or VLLMClient(config.vllm_url)
    reward = reward or RewardModel(root, device=config.embedder_device, floor=config.reward_floor)
    cache = WikiCache(copy_snapshot(root / config.wiki_snapshot, folder), WikiClient(contact=wiki_contact()))
    queue = TaskQueue(train_ids, config.tasks_per_mode, stable_seed(config.seed, "warmup-queue"))
    prompt, stops = episode_prompt(tokenizer), stop_ids(tokenizer)
    out = folder / "episodes.jsonl"
    done = sum(1 for _ in out.open(encoding="utf-8")) // (2 * config.tasks_per_mode * config.group_size) \
        if out.exists() else 0
    for _ in range(done):
        queue.next()
    for step in range(done + 1, config.warmup_steps + 1):
        began = time.monotonic()
        batch = queue.next()
        jobs = [Job(mode_task(tasks[task_id], mode), mode, s, (config.seed, "warmup", step, task_id, mode, s))
                for mode, task_id in batch for s in range(config.group_size)]
        rollouts = collect(jobs, tokenizer, prompt,
                           lambda job: vllm_generator(client, config.vllm_base_name, config.temperature, stops,
                                                      job.seed_parts),
                           lambda: WikiTools(cache, max_calls=config.max_calls), config)
        for job, rollout in zip(jobs, rollouts):
            reward(job.task, rollout)
        with out.open("a", encoding="utf-8") as sink:
            for r in rollouts:
                sink.write(json.dumps({"step": step, "task": r.task_id, "mode": r.mode, "sample": r.sample,
                                       "reward": r.reward, "raw_total": r.components.get("total"), "ids": r.ids,
                                       "mask": r.mask, "finish": r.finish}, ensure_ascii=False) + "\n")
        save_episodes(folder / "episodes", rollouts)
        print(f"разогрев: шаг {step}/{config.warmup_steps} · {(time.monotonic() - began) / 60:.1f} мин · "
              f"награда {statistics.fmean(r.reward for r in rollouts):.3f}", flush=True)
    write_json(folder / "generation.json", {"seed": config.seed, "steps": config.warmup_steps,
                                            "episodes": config.warmup_steps * 2 * config.tasks_per_mode
                                            * config.group_size, "network_seconds": cache.network_seconds})
    return folder


def load_warmup_episodes(folder: Path) -> list[dict]:
    return [json.loads(line) for line in (Path(folder) / "episodes.jsonl").open(encoding="utf-8") if line.strip()]


def warmup_features(model, episodes: list[dict], config: PPOConfig, folder: Path) -> dict:
    """Признаки опорной модели для разогрева: равномерная выборка позиций (≤ ``warmup_max_positions``; первое
    и последнее действие — всегда), без log p. Сохраняются в ``features.pt`` с номерами позиций и числом действий."""
    import torch

    episodes = [e for e in episodes if target_positions(e["mask"])]  # эпизод без действий критику не нужен
    counts = [len(target_positions(e["mask"])) for e in episodes]
    keep = sample_positions(counts, config.warmup_max_positions, stable_seed(config.seed, "warmup-positions"))
    features = []
    for episode, chosen in zip(episodes, keep):
        _, f = reference_pass(model, episode["ids"], episode["mask"], config.temperature, config.logits_chunk,
                              keep=chosen, logp=False)
        features.append(f)
    data = {"features": features, "positions": keep, "counts": counts,
            "rewards": [e["reward"] for e in episodes], "tasks": [e["task"] for e in episodes],
            "modes": [e["mode"] for e in episodes], "ids_hash": [stable_seed(*e["ids"]) for e in episodes],
            "revision": config.policy_revision, "layer": "last_hidden_state", "dtype": "float16"}
    torch.save(data, Path(folder) / "features.pt")
    return data


def _subset(data: dict, take: list[int]) -> dict:
    return {k: [data[k][i] for i in take] for k in ("features", "positions", "counts", "rewards", "tasks", "modes")}


def evaluate_head(head, data: dict, constant: float, mode_means: dict, device: str) -> dict:
    """Меры критика на части разогрева (по выбранным позициям): общие (``critic_metrics``) и строки для
    основной меры EXP-028 (d эпизода), сравнение со средним других ответов задания (справочно)."""
    values = [head_values(head, f, device) for f in data["features"]]
    # срезы ответа считаются по номерам действий: берём ближайшую выбранную позицию к срезу
    sliced = []
    for vs, pos, n in zip(values, data["positions"], data["counts"]):
        full = {}
        for name in POSITIONS:
            target = slice_index(n, name)
            nearest = min(range(len(pos)), key=lambda k: abs(pos[k] - target))
            full[name] = vs[nearest]
        sliced.append(full)
    metrics = critic_metrics(values, data["rewards"], data["modes"], constant, mode_means)
    for name in POSITIONS:  # срезы — по ближайшим выбранным позициям
        rows = [(s[name], r) for s, r in zip(sliced, data["rewards"])]
        errors = [r - v for v, r in rows]
        metrics[f"{name}_mse"] = statistics.fmean(e * e for e in errors)
        metrics[f"{name}_bias"] = statistics.fmean(-e for e in errors)
        metrics[f"{name}_ev"] = _ev(errors, [r for _, r in rows])
    rows = [{"task": t, "mode": m, "d": statistics.fmean((r - constant) ** 2 - (r - v) ** 2 for v in vs)}
            for vs, r, t, m in zip(values, data["rewards"], data["tasks"], data["modes"]) if vs]
    metrics["paired"] = paired_improvement(rows)
    groups = [(t, m) for t, m in zip(data["tasks"], data["modes"])]
    loo = leave_one_out_means(data["rewards"], groups)
    pairs = [(r, b) for r, b in zip(data["rewards"], loo) if b is not None]
    metrics["first_mse_group"] = statistics.fmean((r - b) ** 2 for r, b in pairs) if pairs else None
    # внутри заданий: R и V минус средние задания (в первой позиции выигрыша не ожидаем)
    by_group: dict = {}
    for s, r, g in zip(sliced, data["rewards"], groups):
        by_group.setdefault(g, []).append((s, r))
    for name in POSITIONS:
        errs, targ = [], []
        for items in by_group.values():
            mr = statistics.fmean(r for _, r in items)
            mv = statistics.fmean(s[name] for s, _ in items)
            for s, r in items:
                errs.append((r - mr) - (s[name] - mv))
                targ.append(r - mr)
        metrics[f"{name}_ev_within"] = _ev(errs, targ)
    return metrics


def fit_warmup_head(data: dict, config: PPOConfig, kind: str, lr: float, holdout: set[str] | None,
                    epochs: int | None = None) -> dict:
    """Голова разогрева: при ``holdout`` — обучение на остальных темах с выбором числа эпох по ошибке на отложенных
    (не больше ``warmup_max_epochs``, остановка после ``warmup_patience`` эпох без улучшения); без ``holdout`` —
    заново на всех эпизодах ровно ``epochs`` эпох. Возвращает голову, оптимизатор и историю."""
    import torch

    names = list(range(len(data["rewards"])))
    train_idx = [i for i in names if not holdout or data["tasks"][i] not in holdout]
    hold_idx = [i for i in names if holdout and data["tasks"][i] in holdout]
    train, hold = _subset(data, train_idx), _subset(data, hold_idx) if hold_idx else None
    constant = statistics.fmean(train["rewards"])
    mode_means = {m: statistics.fmean(r for r, mm in zip(train["rewards"], train["modes"]) if mm == m)
                  for m in set(train["modes"])}
    cfg = dataclasses.replace(config, head=kind, head_lr=lr)
    torch.manual_seed(stable_seed(config.seed, "warmup-head", kind, lr))
    dim = train["features"][0].shape[1]
    head = make_head(kind, dim, config.head_hidden, constant).to(config.head_device)
    optimizer = head_optimizer(head, cfg)
    targets = [[r] * f.shape[0] for f, r in zip(train["features"], train["rewards"])]
    history, best, best_state, best_epoch, waited = [], math.inf, None, 0, 0
    limit = epochs if epochs is not None else config.warmup_max_epochs
    for epoch in range(1, limit + 1):
        stats = train_head(head, optimizer, train["features"], targets, cfg, 1,
                           stable_seed(config.seed, "warmup-order", kind, lr, epoch))
        record = {"epoch": epoch, **stats}
        if hold:
            m = evaluate_head(head, hold, constant, mode_means, config.head_device)
            record.update(holdout_mse=m["all_mse"], holdout_mse_const=m["all_mse_const"])
            if m["all_mse"] < best - 1e-12:
                best, best_epoch, waited = m["all_mse"], epoch, 0
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            else:
                waited += 1
        history.append(record)
        if hold and waited >= config.warmup_patience:
            break
    result = {"kind": kind, "lr": lr, "constant": constant, "mode_means": mode_means, "history": history,
              "head": head, "optimizer": optimizer}
    if hold:
        head.load_state_dict(best_state)
        result.update(best_epoch=best_epoch, holdout=evaluate_head(head, hold, constant, mode_means,
                                                                   config.head_device))
    return result


def save_warmup(folder: Path, fitted: dict, config: PPOConfig, extra: dict | None = None) -> Path:
    """Итог разогрева сида для основного запуска: голова, её оптимизатор, вид, lr, число эпох, константа."""
    import torch

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "head.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"head": fitted["head"].state_dict(), "optimizer": fitted["optimizer"].state_dict(),
                "kind": fitted["kind"], "lr": fitted["lr"], "constant": fitted["constant"],
                "hidden": config.head_hidden, "mode_means": fitted.get("mode_means"),
                "feature_scaler": fitted.get("feature_scaler"), "input_dim": fitted.get("input_dim")}, tmp)
    tmp.replace(path)
    write_json(folder / "head.json", {k: v for k, v in fitted.items() if k not in ("head", "optimizer")}
               | (extra or {}) | {"head_sha256": file_hash(folder / "head.pt")})
    return folder / "head.pt"


# --------------------------------------------------------------------------- цикл обучения


def _prepare_critic_inputs(model, jobs, rollouts: list[Rollout], config: PPOConfig, reward,
                           extract=None, scaling=None, *, overlap: bool = False) -> tuple:
    """Готовлю награды, признаки и опорные log p. При двух GPU совмещаю независимые фазы."""
    import torch

    timings = {}

    def score_batch():
        began = time.monotonic()
        for job, rollout in zip(jobs, rollouts):
            reward(job.task, rollout)
        rewards = [r.reward for r in rollouts]
        if any(not math.isfinite(r) for r in rewards):
            raise RunStopped("неконечная награда в шаге")
        timings["reward"] = time.monotonic() - began
        began = time.monotonic()
        features = []
        if extract is not None:
            mean, scale = scaling
            for job, rollout in zip(jobs, rollouts):
                values = extract(job.task, rollout.ids, rollout.mask).values
                features.append((torch.from_numpy(values) - mean) / scale)
        timings["prefix_features"] = time.monotonic() - began if extract is not None else 0.0
        return rewards, features

    def reference_batch():
        began = time.monotonic()
        refs, features = [], []
        for rollout in rollouts:
            ref, x = reference_pass(model, rollout.ids, rollout.mask, config.temperature, config.logits_chunk,
                                    return_features=extract is None)
            refs.append(ref)
            if x is not None:
                features.append(x)
        timings["pass0"] = time.monotonic() - began
        return refs, features

    if overlap:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(score_batch)
            refs, hidden = reference_batch()
            rewards, prefix = future.result()
    else:
        rewards, prefix = score_batch()
        refs, hidden = reference_batch()
    return rewards, refs, prefix if extract is not None else hidden, timings

def save_state(run_dir: Path, step: int, model, optimizer, scaler, head, head_opt, queue: TaskQueue, counters: dict,
               keep: int) -> Path:
    """Полное состояние на границе шага: согласованная пара политика + критик (адаптер, голова, оба оптимизатора,
    scaler, генераторы, очередь, счётчики); хранятся ``keep`` последних."""
    import torch

    states = run_dir / "state"
    states.mkdir(exist_ok=True)
    folder = states / f"step_{step:04d}"
    tmp = states / f"step_{step:04d}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    model.save_pretrained(tmp / "adapter")
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "head": head.state_dict(),
                "head_optimizer": head_opt.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate()}, tmp / "training.pt")
    write_json(tmp / "state.json", {"step": step, "queue": queue.state(), "counters": counters})
    if folder.exists():
        shutil.rmtree(folder)
    os.replace(tmp, folder)
    for old in sorted(p for p in states.glob("step_*") if not p.name.endswith(".tmp"))[:-keep]:
        shutil.rmtree(old)
    return folder


def ppo_code_hash(root: str | Path) -> str:
    return file_hash(Path(root) / "src" / "presentation_rl" / "training" / "ppo.py")


def latest_ppo_state(run_dir: Path) -> Path | None:
    states = (run_dir / "state").glob("step_*")
    complete = [p for p in states if p.name.removeprefix("step_").isdigit()
                and (p / "state.json").is_file() and (p / "training.pt").is_file() and (p / "adapter").is_dir()]
    return max(complete, key=lambda p: int(p.name.removeprefix("step_")), default=None)


def progress(step: int, seconds: float, stats: dict, critic: dict, rollouts: list[Rollout], counters: dict,
             config: PPOConfig) -> None:
    summary = summarize(rollouts)
    left = (config.steps - step) * counters["seconds"] / max(1, step) / 3600
    p2 = stats["passes"][-1]
    print(f"шаг {step}/{config.steps} · {seconds / 60:.1f} мин · награда: поиск {fmt(summary.get('search_reward'))}, "
          f"без {fmt(summary.get('plain_reward'))} · KL {stats['kl_ref_k3']:.1e} · критик MSE "
          f"{fmt(critic.get('all_mse'), 4)} (константа {fmt(critic.get('all_mse_const'), 4)}) · отсечено "
          f"{fmt(p2.get('clip_active'), 3)} · log p |Δ| {fmt(stats['logprob_mismatch'].get('mean_abs'), 4)} · пик "
          f"{fmt(stats['peak_memory_gb'], 1)} ГБ · осталось ≈ {left:.1f} ч", flush=True)


def train(config: PPOConfig, root: str | Path = ".", run_dir: str | Path | None = None, client=None,
          reward: RewardModel | None = None) -> Path:
    """Новый запуск (``run_dir=None``, нужен ``config.warmup_dir`` с ``head.pt``) или продолжение запуска из его
    папки с последнего полного состояния."""
    import torch

    from ..tasks import load_tasks
    from ..tools.cache import WikiCache
    from ..tools.wiki import WikiClient
    from ..tools.wiki_tools import WikiTools
    from .prefix_features import PrefixFeatureExtractor, prefix_scaler_tensors
    from .vllm_client import VLLMClient

    root = Path(root).resolve()
    setup_torch()
    all_tasks = {t.id: t for t in load_tasks(root / "data" / "tasks" / "tasks.json")}
    train_ids = sorted(k for k, t in all_tasks.items() if t.split == "train")
    val_topics = fixed_validation_topics([k for k, t in all_tasks.items() if t.split == "validation"],
                                         config.val_topics)
    client = client or VLLMClient(config.vllm_url)
    if not config.warmup_dir:
        raise ValueError("нужен config.warmup_dir — папка разогрева этого сида с head.pt")
    warmup_file = Path(config.warmup_dir) / "head.pt"
    warmup_hash = file_hash(warmup_file)
    warm = torch.load(warmup_file, map_location="cpu", weights_only=True)
    scaling = prefix_scaler_tensors(warm["feature_scaler"]) if warm.get("feature_scaler") is not None else None
    prefix_code = file_hash(Path(__file__).with_name("prefix_features.py")) if scaling is not None else None

    if run_dir is None:
        run_dir = new_run_dir(config, root)
        manifest = start_manifest(run_dir, config, root, {
            "validation_topics": val_topics, "vllm_version": client.version(), "ppo_code": ppo_code_hash(root),
            "prefix_code": prefix_code,
            "warmup": {"dir": str(config.warmup_dir), "head_sha256": warmup_hash,
                       "info": json.loads((Path(config.warmup_dir) / "head.json").read_text(encoding="utf-8"))}})
        state_dir, start = None, 0
    else:
        run_dir = Path(run_dir)
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        check_resume(manifest, config, root)
        if manifest.get("ppo_code") != ppo_code_hash(root):
            raise ValueError("код PPO изменился — это новый запуск, а не продолжение")
        if manifest.get("prefix_code") != prefix_code or manifest["warmup"]["head_sha256"] != warmup_hash:
            raise ValueError("изменились признаки или разогрев критика — это новый запуск")
        state_dir = latest_ppo_state(run_dir)
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
    head_cfg = dataclasses.replace(config, head=warm["kind"], head_lr=warm["lr"])
    dim = scaling[0].numel() if scaling is not None else model.config.hidden_size
    if scaling is not None and warm.get("input_dim") != dim:
        raise ValueError("размер входа сохранённого критика не совпадает с префиксными признаками")
    head = make_head(warm["kind"], dim, warm["hidden"], 0.0).to(config.head_device)
    head.load_state_dict(warm["head"])
    head_opt = head_optimizer(head, head_cfg)
    head_opt.load_state_dict(warm["optimizer"])
    queue = TaskQueue(train_ids, config.tasks_per_mode, config.seed)
    counters = {"updates": 0, "skipped": 0, "episodes": 0, "generated_tokens": 0, "train_tokens": 0,
                "seconds": 0.0, "network_seconds": 0.0, "last_mean_reward": warm["constant"]}
    if state_dir:
        saved = torch.load(state_dir / "training.pt", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        head.load_state_dict(saved["head"])
        head_opt.load_state_dict(saved["head_optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        random.setstate(saved["python_rng"])
        state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
        queue.load_state(state["queue"])
        counters.update(state["counters"])

    reward = reward or RewardModel(root, device=config.embedder_device, floor=config.reward_floor)
    extract = (PrefixFeatureExtractor(tokenizer, reward, max_new_tokens=config.max_new_tokens,
                                      max_context=config.max_context) if scaling is not None else None)
    device = next(model.parameters()).device
    overlap = (extract is not None and device.type == "cuda" and config.embedder_device.startswith("cuda:")
               and torch.device(config.embedder_device) != device)
    mode_means = warm.get("mode_means") if extract is not None else None
    prompt, stops = episode_prompt(tokenizer), stop_ids(tokenizer)
    adapters = run_dir / "adapters"
    adapters.mkdir(exist_ok=True)

    def publish(step: int, previous: str | None) -> str:
        name = f"{run_dir.name}_s{session}_step{step}"
        path = export_adapter(model, adapters / f"step_{step:04d}")
        client.load_lora(name, str(path))
        if previous:
            client.unload_lora(previous)
        for old in adapters.glob("step_*"):
            if old != path and not old.name.endswith(".tmp"):
                shutil.rmtree(old, ignore_errors=True)
        return name

    for name in client.models():
        if name.startswith(run_dir.name + "_"):
            client.unload_lora(name)
    current = publish(start, None)
    journals["events"].write({"type": "start", "step": start, "run": run_dir.name})
    print(f"запуск {run_dir.name}, сессия {session}, с шага {start + 1} до {config.steps}", flush=True)
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
            rewards, refs, features, prep_times = _prepare_critic_inputs(
                model, jobs, rollouts, config, reward, extract, scaling, overlap=overlap,
            )
            values = [head_values(head, f, config.head_device) for f in features]
            modes = [r.mode for r in rollouts]
            baseline = warm["constant"] if extract is not None else counters["last_mean_reward"]
            critic_before = critic_metrics(values, rewards, modes, baseline, mode_means)
            raw, targets = zip(*(gae(v, r, config.gae_lambda) for v, r in zip(values, rewards)))
            normalized, advantage_stats = normalize(list(raw), config.std_floor)
            prepared = time.monotonic()

            head_stats = train_head(head, head_opt, features, list(targets), head_cfg, config.head_epochs_step,
                                    stable_seed(config.seed, "head-order", step))
            after = [head_values(head, f, config.head_device) for f in features]
            critic_after = critic_metrics(after, rewards, modes, baseline, mode_means)
            del features, after
            criticized = time.monotonic()

            stats = ppo_train_step(model, optimizer, scaler, rollouts, normalized, refs, config)
            del refs
            trained = time.monotonic()
            problem = mismatch_exceeded(stats["logprob_mismatch"], config.max_logprob_mismatch or {})
            if problem:
                raise RunStopped(problem)
            current = publish(step, current)
            published = time.monotonic()

            counters["updates"] += stats["updates"]
            counters["skipped"] += stats["skipped"]
            counters["episodes"] += len(rollouts)
            counters["generated_tokens"] += sum(r.model_tokens for r in rollouts)
            counters["train_tokens"] += stats["tokens"]
            counters["seconds"] += published - began
            counters["network_seconds"] += cache.network_seconds - network_before
            counters["last_mean_reward"] = statistics.fmean(rewards)
            step_stats = {
                "step": step, "tasks": batch, "advantage": advantage_stats, **stats,
                "critic_before": critic_before, "critic_after": critic_after, **head_stats,
                "values": {"mean": statistics.fmean(v for vs in values for v in vs) if any(values) else None},
                **summarize(rollouts),
                "seconds": {"generation": generated - began, **prep_times, "prepare_wall": prepared - generated,
                             "head": criticized - prepared, "train": trained - criticized,
                            "adapter": published - trained, "network": cache.network_seconds - network_before},
                "preparation_overlapped": overlap,
                "counters": dict(counters), "queue": queue.state(),
                "embedder_oom_retries": getattr(getattr(reward, "embedder", None), "oom_retries", None),
            }
            progress(step, published - began, stats, critic_before, rollouts, counters, config)
            if counters["skipped"] > config.max_skipped_updates * config.ppo_epochs * step:
                journals["events"].write({"type": "warning", "step": step,
                                          "message": f"пропущено обновлений {counters['skipped']}"})

            step_stats["seconds"]["validation"] = 0.0
            if step % config.eval_every == 0 or step == config.steps:
                validating = time.monotonic()
                folder = run_dir / "checkpoints" / f"step_{step:04d}"
                if folder.exists():
                    shutil.rmtree(folder)
                folder.mkdir(parents=True)
                shutil.copytree(adapters / f"step_{step:04d}", folder / "adapter")
                torch.save(head.state_dict(), folder / "critic_head.pt")
                val = validate(client, current, all_tasks, val_topics, config.val_samples, tokenizer, cache, reward,
                               config, folder / "validation")
                summary = summarize(val)
                journals["validation"].write({"step": step, **summary})
                print(f"  validation, шаг {step}: награда поиск {fmt(summary.get('search_reward'))}, "
                      f"без {fmt(summary.get('plain_reward'))}", flush=True)
                write_json(folder / "done.json", {"step": step, "session": session,
                                                  "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
                step_stats["seconds"]["validation"] = time.monotonic() - validating
            checkpointing = time.monotonic()
            # Сначала журнал: если сохранение оборвётся, продолжение отменит запись этого шага.
            journals["steps"].write(step_stats)
            if step % config.save_every == 0 or step % config.eval_every == 0 or step == config.steps:
                save_state(run_dir, step, model, optimizer, scaler, head, head_opt, queue, counters,
                           config.keep_states)
                journals["events"].write({"type": "saved", "step": step,
                                          "seconds": time.monotonic() - checkpointing})
    except BaseException as error:
        journals["events"].write({"type": "stopped", "at_step": step, "error": repr(error)[:2000]})
        raise
    journals["events"].write({"type": "finished", "step": config.steps})
    return run_dir


PREFIX_CRITICS = {
    "linear": {"label": "Линейный критик на префиксных признаках", "epochs": 4, "hidden": 0, "lr": 1e-3},
    "mlp": {"label": "Критик с MLP на префиксных признаках", "epochs": 2, "hidden": 64, "lr": 1e-3},
}


def prepare_prefix_critics(config: PPOConfig, root: str | Path, kinds: tuple[str, ...]) -> dict[str, PPOConfig]:
    """Сохраняю разогрев каждого варианта отдельно. Оба варианта читают один буфер этого сида."""
    import torch

    from .prefix_features import fit_prefix_head, load_or_prepare_prefix_warmup

    root = Path(root).resolve()
    folder = root / "runs" / "ppo" / f"warmup_seed{config.seed}"
    for kind in kinds:
        if kind not in PREFIX_CRITICS:
            raise ValueError(f"неизвестный вариант критика: {kind}")
    data, provenance = load_or_prepare_prefix_warmup(root, config, folder)
    configs = {}
    for kind in kinds:
        settings = PREFIX_CRITICS[kind]
        saved = folder / f"prefix_{kind}"
        cfg = dataclasses.replace(config, head=kind, head_hidden=settings["hidden"], head_lr=settings["lr"],
                                  head_device="cpu", warmup_epochs=settings["epochs"], warmup_dir=str(saved))
        manifest = {"seed": config.seed, "settings": settings, "provenance": provenance,
                    "head_batch": cfg.head_batch, "head_betas": list(cfg.head_betas),
                    "head_eps": cfg.head_eps, "head_weight_decay": cfg.head_weight_decay,
                    "head_grad_clip": cfg.head_grad_clip}
        path, info_path = saved / "head.pt", saved / "head.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
        if path.is_file() and info.get("prefix_warmup") == manifest and info.get("head_sha256") == file_hash(path):
            print(f"читаю разогретый критик: {settings['label']} · {saved}", flush=True)
        else:
            started = time.monotonic()
            fit = fit_prefix_head(data, cfg, None, cfg.head_lr, kind=kind, epochs=cfg.warmup_epochs)
            save_warmup(saved, fit, cfg, {"prefix_warmup": manifest, "episodes": len(data["rewards"])})
            print(f"разогретый критик: {settings['label']} · {time.monotonic() - started:.1f} с · {saved}",
                  flush=True)
        configs[kind] = cfg
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return configs


def _pilot_step_seconds(row: dict) -> float:
    times = row["seconds"]
    # Награда, признаки префикса и проход 0 частично параллельны. Считаю их по общему времени.
    return sum(times[k] for k in ("generation", "prepare_wall", "head", "train", "adapter", "validation"))


def _pilot_summary(run_dir: Path, config: PPOConfig) -> dict:
    steps = [r for r in read_journal(run_dir / "steps.jsonl") if "critic_before" in r]
    val = [r for r in read_journal(run_dir / "validation.jsonl") if r.get("step") == config.steps]
    saved = [r["seconds"] for r in read_journal(run_dir / "events.jsonl") if r.get("type") == "saved"]
    if [r["step"] for r in steps] != list(range(1, config.steps + 1)) or len(val) != 1:
        raise ValueError(f"неполный журнал пилота: {run_dir}")
    totals = [_pilot_step_seconds(r) for r in steps]
    return {
        "run_dir": str(run_dir), "time_seconds": sum(totals) + sum(saved),
        "mean_step_seconds": statistics.fmean(total - r["seconds"]["validation"] for total, r in zip(totals, steps)),
        "mean_prefix_seconds": statistics.fmean(r["seconds"]["prefix_features"] for r in steps),
        "peak_memory_gb": max(r["peak_memory_gb"] for r in steps),
        "skipped_updates": sum(r["skipped"] for r in steps),
        "critic_before": {key: statistics.fmean(r["critic_before"][key] for r in steps)
                          for key in ("all_mse", "all_mse_const", "all_mse_mode", "first_mse", "first_mse_mode")},
        "mean_clip_active": statistics.fmean(r["passes"][-1]["clip_active"] for r in steps),
        "max_kl_old_new": max((r["kl_old_new_k3"] for r in steps if r["kl_old_new_k3"] is not None), default=None),
        "validation": val[0], "steps": steps,
    }


def run_prefix_pilots(config: PPOConfig, root: str | Path, client, steps: int = 5) -> dict:
    """Два пилота сида 0 с общего старта. Повторный запуск продолжает незавершённый пилот."""
    import torch

    if steps < 1:
        raise ValueError("число шагов пилота должно быть положительным")
    root = Path(root).resolve()
    configs = prepare_prefix_critics(dataclasses.replace(config, seed=0), root, tuple(PREFIX_CRITICS))
    folder = Path("runs") / "ppo_pilot" / f"prefix_{steps}_steps"
    report = {}
    for kind, cfg in configs.items():
        target = folder / kind
        pilot = dataclasses.replace(cfg, steps=steps, eval_every=steps, save_every=1, runs_dir=str(target))
        existing = sorted(p.parent for p in (root / target).glob("*/manifest.json"))
        run = existing[-1] if existing else None
        if run is not None:
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            check_resume(manifest, pilot, root)
            if (manifest.get("ppo_code") != ppo_code_hash(root)
                    or manifest.get("prefix_code") != file_hash(Path(__file__).with_name("prefix_features.py"))
                    or manifest["warmup"]["head_sha256"] != file_hash(Path(pilot.warmup_dir) / "head.pt")):
                raise ValueError(f"изменились код или разогрев пилота в {run}")
        state = latest_ppo_state(run) if run is not None else None
        finished = state and json.loads((state / "state.json").read_text(encoding="utf-8"))["step"] == steps
        print(f"{PREFIX_CRITICS[kind]['label']} · {steps} шагов", flush=True)
        if not finished:
            run = train(pilot, root, run_dir=run, client=client)
        report[kind] = _pilot_summary(run, pilot)
        for name in client.models():
            if name.startswith(run.name + "_"):
                client.unload_lora(name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    schedules = [[row["tasks"] for row in arm["steps"]] for arm in report.values()]
    if any(schedule != schedules[0] for schedule in schedules[1:]):
        raise ValueError("очереди заданий в двух пилотах различаются")
    write_json(root / folder / "comparison.json", report)
    return report


# --------------------------------------------------------------------------- проверка памяти

def memory_check(config: PPOConfig, root: str | Path = ".", context: int | None = None,
                 model_tokens: int | None = None) -> dict:
    """Карта 0: полный цикл PPO на эпизоде с максимальным контекстом — проход 0 (опорная, признаки), голова,
    два прохода политики с созданием состояний обоих Adam, затем ещё один шаг (повторные шаги). Пик и время."""
    import torch

    setup_torch()
    context = context or config.max_context
    model_tokens = min(model_tokens or config.max_new_tokens, context - 1)
    model, _ = load_policy(config, root)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate,
                                  betas=tuple(config.betas), eps=config.eps, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda")
    head = make_head(config.head, model.config.hidden_size, config.head_hidden, 0.0).to(config.head_device)
    head_opt = head_optimizer(head, config)
    rng = random.Random(0)
    ids = [rng.randrange(1000, 100000) for _ in range(context)]
    mask = [0] * (context - model_tokens) + [1] * model_tokens
    rollout = Rollout("memory", "search", 0, ids, mask, [math.nan] * model_tokens, "", "", "stop", [], 0, 0.0)
    result = {"context": context, "model_tokens": model_tokens}
    start = time.monotonic()
    try:
        for repeat in range(2):
            ref, f = reference_pass(model, ids, mask, config.temperature, config.logits_chunk)
            values = head_values(head, f, config.head_device)
            raw, targets = gae(values, 0.5, config.gae_lambda)
            normalized, _ = normalize([raw], config.std_floor)
            train_head(head, head_opt, [f], [targets], config, 1, repeat)
            stats = ppo_train_step(model, optimizer, scaler, [rollout], normalized, [ref], config)
            result[f"peak_memory_gb_{repeat + 1}"] = stats["peak_memory_gb"]
        result.update(ok=True, seconds=time.monotonic() - start,
                      reserved_gb=torch.cuda.memory_reserved(0) / 2**30)
    except Exception as error:
        result.update(ok=False, error=repr(error)[:1000])
    finally:
        del model, optimizer
        torch.cuda.empty_cache()
    return result
