# Здесь общаемся с vLLM-сервером: генерация реплики по токенам, подгрузка и выгрузка LoRA во время обучения.
"""Клиент OpenAI-совместимого сервера vLLM 0.8.5 (``docs/grpo.md``, раздел 4).

vLLM работает отдельным процессом в своём окружении на карте 1 (у него свои версии torch и transformers)::

    VLLM_USE_V1=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True CUDA_VISIBLE_DEVICES=1 \\
    python -m vllm.entrypoints.openai.api_server --model <слитая SFT-модель> --served-model-name sft \\
        --dtype float16 --enable-lora --max-lora-rank 32 --max-loras 1 --max-model-len 25600 \\
        --gpu-memory-utilization 0.80 --generation-config vllm --port 8000 --seed 0
    # без кеширования префикса и chunked prefill: на V100 в 0.8.5 их нет

Реплика запрашивается токенами (``prompt`` — список id), все параметры выборки — явно (``--generation-config vllm``
и явные значения: иначе подмешиваются настройки модели — top-k 20, top-p 0,8, T 0,7). Ответ — id выбранных токенов
(``return_tokens_as_token_ids``) и их log p после деления на температуру.

Запросы идут через ``urllib`` без прокси: сервер локальный, а переменные прокси на сервере его бы не пустили.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request


class VLLMError(RuntimeError):
    pass


class VLLMClient:
    def __init__(self, url: str = "http://127.0.0.1:8000", timeout: float = 1800.0, retries: int = 3):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # --- HTTP

    def _request(self, path: str, payload: dict | None = None, method: str | None = None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.url + path, data=data, method=method or ("POST" if data else "GET"),
                                         headers={"Content-Type": "application/json"})
        last = None
        for attempt in range(self.retries):
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    body = response.read().decode()
                    return json.loads(body) if body.strip().startswith(("{", "[")) else body
            except urllib.error.HTTPError as error:  # ошибка запроса — повтор не поможет
                raise VLLMError(f"{path}: HTTP {error.code}: {error.read().decode(errors='replace')[:500]}") from error
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                last = error
                time.sleep(2 * (attempt + 1))
        raise VLLMError(f"{path}: сервер недоступен ({last})")

    # --- сервер и адаптеры

    def version(self) -> str:
        result = self._request("/version")
        return result.get("version", "") if isinstance(result, dict) else str(result)

    def models(self) -> list[str]:
        return [m["id"] for m in self._request("/v1/models")["data"]]

    def load_lora(self, name: str, path: str) -> None:
        self._request("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
        if name not in self.models():  # подтверждение: следующий сбор — только с загруженной версией
            raise VLLMError(f"адаптер {name} не появился в списке моделей сервера")

    def unload_lora(self, name: str) -> None:
        self._request("/v1/unload_lora_adapter", {"lora_name": name})

    # --- генерация

    def generate(self, ids: list[int], max_tokens: int, model: str, temperature: float, seed: int,
                 stop_ids: list[int]) -> tuple[list[int], bool, list[float]]:
        """Одна реплика: (новые id, закончила ли модель реплику, log p выбранных токенов)."""
        payload = {
            "model": model, "prompt": list(ids), "max_tokens": int(max_tokens), "temperature": temperature,
            "top_p": 1.0, "top_k": -1, "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "repetition_penalty": 1.0, "seed": int(seed), "stop_token_ids": list(stop_ids), "logprobs": 0,
            "return_tokens_as_token_ids": True, "skip_special_tokens": False, "ignore_eos": False,
        }
        choice = self._request("/v1/completions", payload)["choices"][0]
        new, logprobs = parse_tokens(choice.get("logprobs") or {})
        finished = choice.get("finish_reason") == "stop"
        if finished and stop_ids and (not new or new[-1] not in stop_ids):
            # стоп-токен выбран моделью, но сервер не вернул его — дописываем (log p неизвестен)
            new.append(stop_ids[0])
            logprobs.append(math.nan)
        return new, finished, logprobs


def parse_tokens(logprobs: dict) -> tuple[list[int], list[float]]:
    """id из ``tokens`` вида «token_id:151645» и их log p (``None`` → nan)."""
    tokens = logprobs.get("tokens") or []
    values = logprobs.get("token_logprobs") or [None] * len(tokens)
    ids = [int(t.rsplit(":", 1)[1]) for t in tokens]
    return ids, [math.nan if v is None else float(v) for v in values]
