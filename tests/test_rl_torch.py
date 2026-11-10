# Здесь проверяем части RL с torch на крошечной случайной модели (на сервере; без torch и peft тесты пропускаются).
"""Проверки до длинных запусков (``docs/grpo.md``, раздел 13): log p кусками = полные логиты, нулевой адаптер =
опорная модель, градиент ratio = градиент политики, накопление по эпизодам = целый пакет, средний градиент k2 =
градиент точного KL, обновление сдвигает вероятности в нужную сторону. Модель — крошечная Qwen3 из конфига
(без загрузки весов), на процессоре во float32."""
import math

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from presentation_rl.training.common import (RLConfig, Rollout, TARGET_MODULES, kl_terms, policy_terms,  # noqa: E402
                                             sequence_logprobs, target_positions)
from presentation_rl.training.grpo import train_step  # noqa: E402


def tiny_model(seed=0):
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                      max_position_embeddings=256, tie_word_embeddings=False)
    model = transformers.Qwen3ForCausalLM(config).float()
    model.config.use_cache = False
    return model


def with_lora(model, seed=1):
    torch.manual_seed(seed)
    config = peft.LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, bias="none", target_modules=list(TARGET_MODULES),
                             task_type="CAUSAL_LM")
    return peft.get_peft_model(model, config)


IDS = [3, 17, 5, 44, 9, 81, 2, 60, 33, 7, 12, 90, 41, 8]
MASK = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1]


def reference_logprobs(model, ids, mask, temperature):
    logits = model(input_ids=torch.tensor([ids])).logits[0].float() / temperature
    logp = torch.log_softmax(logits, -1)
    positions = target_positions(mask)
    return torch.stack([logp[p - 1, ids[p]] for p in positions])


def test_chunked_logprobs_match_full_logits_and_gradients():
    model = tiny_model()
    for chunk in (1, 3, 100):
        model.zero_grad()
        logp, entropy = sequence_logprobs(model, IDS, MASK, 0.9, chunk)
        logp.sum().backward()
        grads = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        expected = reference_logprobs(model, IDS, MASK, 0.9)
        expected.sum().backward()
        assert torch.allclose(logp, expected.detach(), atol=1e-5)
        assert all(torch.allclose(a, p.grad, atol=1e-5) for a, p in zip(grads, model.parameters()))
        assert entropy.shape == logp.shape and bool((entropy > 0).all())


def test_zero_adapter_equals_reference_and_k2_gradient_is_zero():
    model = with_lora(tiny_model())
    with torch.no_grad(), model.disable_adapter():
        ref, _ = sequence_logprobs(model, IDS, MASK, 0.9)
    logp, _ = sequence_logprobs(model, IDS, MASK, 0.9)
    assert torch.allclose(logp, ref, atol=1e-6)
    k2, k3 = kl_terms(logp, ref)
    k2.sum().backward()
    assert all(p.grad is None or float(p.grad.abs().max()) < 1e-8 for p in model.parameters() if p.requires_grad)
    assert float(k3.abs().max()) < 1e-8


def test_ratio_has_policy_gradient_not_zero():
    model = tiny_model()
    logp, _ = sequence_logprobs(model, IDS, MASK, 0.9)
    terms, clip_fraction = policy_terms(logp, 1.5)
    assert torch.allclose(terms.detach(), torch.full_like(terms, -1.5)) and float(clip_fraction) == 0.0
    terms.sum().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    logp, _ = sequence_logprobs(model, IDS, MASK, 0.9)
    (-1.5 * logp).sum().backward()
    assert any(float(g.abs().max()) > 0 for g in grads)
    assert all(torch.allclose(a, p.grad, atol=1e-6) for a, p in zip(grads, model.parameters()))


def test_accumulation_by_episode_equals_whole_batch():
    model = tiny_model()
    episodes = [(IDS, MASK, 1.0), (IDS[::-1], MASK[::-1], -0.5)]
    total = sum(len(target_positions(m)) for _, m, _ in episodes)
    for ids, mask, a in episodes:
        logp, _ = sequence_logprobs(model, ids, mask, 0.9)
        (policy_terms(logp, a)[0].sum() / total).backward()
    accumulated = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    whole = sum(policy_terms(sequence_logprobs(model, ids, mask, 0.9)[0], a)[0].sum() for ids, mask, a in episodes)
    (whole / total).backward()
    assert all(torch.allclose(a, p.grad, atol=1e-6) for a, p in zip(accumulated, model.parameters()))


def test_expected_k2_gradient_is_exact_reverse_kl_gradient():
    torch.manual_seed(0)
    theta = torch.randn(6, requires_grad=True)
    ref = torch.log_softmax(torch.randn(6), -1)
    logp = torch.log_softmax(theta, -1)
    probs = logp.exp().detach()  # выборка из своей политики: вероятности выборки не дифференцируются
    k2, k3 = kl_terms(logp, ref)
    (probs * k2).sum().backward()
    k2_grad = theta.grad.clone()
    theta.grad = None
    logp = torch.log_softmax(theta, -1)
    (logp.exp() * (logp - ref)).sum().backward()  # точный KL(π_θ‖π_ref)
    assert torch.allclose(k2_grad, theta.grad, atol=1e-6)


def test_train_step_moves_probabilities_in_the_right_direction():
    model = with_lora(tiny_model())
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    good = Rollout("t", "plain", 0, IDS, MASK, [math.nan] * sum(MASK), "", "", "stop", [], 0, 0.0)
    bad_ids = IDS[:3] + [70, 71, 72] + IDS[6:]
    bad = Rollout("t", "plain", 1, bad_ids, MASK, [math.nan] * sum(MASK), "", "", "stop", [], 0, 0.0)
    config = RLConfig(kl_beta=0.0, learning_rate=1e-2, kl_old_new_episodes=2)
    with torch.no_grad():
        before_good = sequence_logprobs(model, good.ids, good.mask, config.temperature)[0].sum()
        before_bad = sequence_logprobs(model, bad.ids, bad.mask, config.temperature)[0].sum()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, betas=config.betas, weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=False)
    stats = train_step(model, optimizer, scaler, [good, bad], [1.0, -1.0], config)
    with torch.no_grad():
        after_good = sequence_logprobs(model, good.ids, good.mask, config.temperature)[0].sum()
        after_bad = sequence_logprobs(model, bad.ids, bad.mask, config.temperature)[0].sum()
    assert after_good > before_good and after_bad < before_bad
    assert not stats["skipped"] and stats["grad_norm"] > 0 and stats["kl_old_new_k3"] > 0
    assert stats["tokens"] == 2 * len(target_positions(MASK)) and stats["logprob_mismatch"] == {"tokens": 0}
