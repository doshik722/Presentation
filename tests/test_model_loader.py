"""Генератор эпизода задаёт все параметры выборки явно (без подмешивания generation_config модели)."""
import contextlib
import sys
import types

from presentation_rl.model_loader import hf_generator


class _Tensor:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        row, cut = index
        return _Row(self.rows[row][cut])


class _Row(list):
    def tolist(self):
        return list(self)


class _Model:
    device = "cpu"

    def __init__(self):
        self.kwargs = None

    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        return _Tensor([input_ids.rows[0] + [7, 99, 5]])


class _Tokenizer:
    eos_token_id = 99
    pad_token_id = None

    def convert_tokens_to_ids(self, token):
        return 99


@contextlib.contextmanager
def _fake_torch():
    torch = types.ModuleType("torch")
    torch.tensor = lambda rows, device=None: _Tensor(rows)
    torch.ones_like = lambda tensor: tensor
    torch.no_grad = contextlib.nullcontext
    saved = sys.modules.get("torch")
    sys.modules["torch"] = torch
    try:
        yield
    finally:
        if saved is None:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = saved


def test_sampling_parameters_explicit():
    model = _Model()
    with _fake_torch():
        new, finished = hf_generator(model, _Tokenizer())([1, 2], 10)
        assert (new, finished) == ([7, 99], True)
        assert model.kwargs["top_k"] == 0 and model.kwargs["top_p"] == 1.0 and model.kwargs["temperature"] == 0.9
        assert model.kwargs["do_sample"] is True and model.kwargs["repetition_penalty"] == 1.0
        hf_generator(model, _Tokenizer(), top_k=20)([1, 2], 10)
        assert model.kwargs["top_k"] == 20


def test_greedy():
    model = _Model()
    with _fake_torch():
        hf_generator(model, _Tokenizer(), temperature=0)([1], 5)
    assert model.kwargs["do_sample"] is False and "top_k" not in model.kwargs
