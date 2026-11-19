# Здесь проверяем части PPO с torch на крошечной модели (на сервере; без torch и peft тесты пропускаются).
"""Проверки до длинных запусков PPO (``docs/ppo.md``, раздел 15): признаки критика — состояния опорной модели
в позиции перед действием и не зависят от будущего; голова на признаках с процессора = на тех же признаках
на карте; обучение головы не трогает модель; вес sg(ρ) у k2 даёт градиент точного KL при выборке старой
политикой; в первом проходе ρ ≡ 1, во втором — старые log p; шаг сдвигает вероятности в нужную сторону."""
import math

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from presentation_rl.training.common import (Rollout, TARGET_MODULES, kl_terms, sequence_logprobs,  # noqa: E402
                                             target_positions)
from presentation_rl.training.ppo import (PPOConfig, head_optimizer, head_values, make_head,  # noqa: E402
                                          ppo_policy_terms, ppo_train_step, reference_pass, train_head)

IDS = [3, 17, 5, 44, 9, 81, 2, 60, 33, 7, 12, 90, 41, 8]
MASK = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1]


def tiny_lora_model(seed=0):
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                      max_position_embeddings=256, tie_word_embeddings=False)
    model = transformers.Qwen3ForCausalLM(config).float()
    model.config.use_cache = False
    torch.manual_seed(seed + 1)
    lora = peft.LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, bias="none", target_modules=list(TARGET_MODULES),
                           task_type="CAUSAL_LM")
    model = peft.get_peft_model(model, lora)
    for name, parameter in model.named_parameters():  # ненулевой адаптер: политика ≠ опорная
        if parameter.requires_grad:
            parameter.data = (parameter.data + 0.05 * torch.randn_like(parameter)).float()
    return model


def cpu_config(**kw):
    return PPOConfig(head_device="cpu", kl_old_new_episodes=2, **kw)


def test_reference_pass_features_are_reference_states_before_each_action():
    model = tiny_lora_model()
    model.train()
    ref, features = reference_pass(model, IDS, MASK, 0.9)
    assert model.training  # режим восстановлен
    with torch.no_grad(), model.disable_adapter():
        hidden = model.get_base_model().model(input_ids=torch.tensor([IDS])).last_hidden_state[0]
        expected_ref, _ = sequence_logprobs(model, IDS, MASK, 0.9)
    positions = [p - 1 for p in target_positions(MASK)]
    assert torch.allclose(features.float(), hidden[positions].half().float(), atol=1e-3)
    assert torch.allclose(ref, expected_ref, atol=1e-5)
    _, some = reference_pass(model, IDS, MASK, 0.9, keep=[0, 3], logp=False)
    assert torch.allclose(some.float(), features[[0, 3]].float())


def test_features_do_not_depend_on_the_future():
    model = tiny_lora_model()
    _, a = reference_pass(model, IDS, MASK, 0.9)
    changed = IDS[:9] + [50, 51, 52, 53, 54]  # после позиции 8 — другое продолжение
    _, b = reference_pass(model, changed, MASK, 0.9)
    before = sum(1 for p in target_positions(MASK) if p - 1 < 9)  # действия, чьё состояние кончается до позиции 9
    assert torch.allclose(a[:before].float(), b[:before].float(), atol=1e-3)


def test_head_from_cpu_cache_equals_direct_and_training_head_leaves_model_untouched():
    model = tiny_lora_model()
    config = cpu_config(head="mlp", head_hidden=8, head_lr=1e-2, head_batch=4)
    _, features = reference_pass(model, IDS, MASK, 0.9)
    head = make_head("mlp", 32, 8, 0.3)
    assert head_values(head, features, "cpu") == pytest.approx([0.3] * features.shape[0])
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = head_optimizer(head, config)
    first = train_head(head, optimizer, [features], [[0.8] * features.shape[0]], config, 1, seed=0)
    for epoch in range(30):
        last = train_head(head, optimizer, [features], [[0.8] * features.shape[0]], config, 1, seed=epoch)
    assert last["head_loss"] < first["head_loss"]
    direct = head(features.float())[:, 0].tolist()
    assert head_values(head, features, "cpu") == pytest.approx(direct, rel=1e-5, abs=1e-6)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all(p.grad is None for p in model.parameters())


def test_weighted_k2_gives_exact_kl_gradient_under_old_policy_sampling():
    torch.manual_seed(0)
    old = torch.log_softmax(torch.randn(6), -1)
    ref = torch.log_softmax(torch.randn(6), -1)
    theta = torch.randn(6, requires_grad=True)
    logp = torch.log_softmax(theta, -1)
    rho = torch.exp(logp - old).detach()
    k2, _ = kl_terms(logp, ref)
    (old.exp() * rho * k2).sum().backward()  # E_{q0}[sg(ρ)·k2]
    weighted = theta.grad.clone()
    theta.grad = None
    logp = torch.log_softmax(theta, -1)
    (logp.exp() * (logp - ref)).sum().backward()  # точный KL(q‖p)
    assert torch.allclose(weighted, theta.grad, atol=1e-6)


def test_policy_terms_clip_shares():
    logp = torch.log(torch.tensor([0.5, 0.5, 0.5]))
    old = torch.log(torch.tensor([0.5, 0.3, 0.7]))  # ρ = 1; 1,67; 0,71
    terms, ratio, outside, active = ppo_policy_terms(logp, torch.tensor([1.0, 1.0, -1.0]), old, 0.2)
    assert float(outside) == pytest.approx(2 / 3) and float(active) == pytest.approx(2 / 3)
    assert terms.tolist() == pytest.approx([-1.0, -1.2, 0.8])


def test_ppo_step_first_pass_ratio_one_second_uses_old_and_moves_probabilities():
    model = tiny_lora_model()
    config = cpu_config(kl_beta=0.0, learning_rate=1e-2, ppo_epochs=2)
    good = Rollout("t", "plain", 0, IDS, MASK, [math.nan] * sum(MASK), "", "", "stop", [], 0, 0.0)
    bad_ids = IDS[:3] + [70, 71, 72] + IDS[6:]
    bad = Rollout("t", "plain", 1, bad_ids, MASK, [math.nan] * sum(MASK), "", "", "stop", [], 0, 0.0)
    refs = [reference_pass(model, r.ids, r.mask, 0.9)[0] for r in (good, bad)]
    n = len(target_positions(MASK))
    with torch.no_grad():
        before = [sequence_logprobs(model, r.ids, r.mask, 0.9)[0].sum() for r in (good, bad)]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=False)
    stats = ppo_train_step(model, optimizer, scaler, [good, bad], [[1.0] * n, [-1.0] * n], refs, config)
    with torch.no_grad():
        after = [sequence_logprobs(model, r.ids, r.mask, 0.9)[0].sum() for r in (good, bad)]
    first, second = stats["passes"]
    assert first["ratio_outside"] == 0 and first["clip_active"] == 0  # ρ ≡ 1 в первом проходе
    assert "ratio_max" in second and second["ratio_max"] != pytest.approx(1.0)  # во втором — старые log p
    assert after[0] > before[0] and after[1] < before[1]
    assert stats["updates"] == 2 and stats["skipped"] == 0 and stats["tokens"] == 2 * n
    assert stats["kl_old_new_k3"] > 0 and stats["kl_ref_k3"] > 0
