<!-- Здесь коротко описываем проект, карту файлов и правила работы; подробности — в docs/. -->
# Агенты для создания презентаций: PPO, GRPO и LangChain

Дипломная работа. Агенты получают тему и создают презентацию в формате PPTX.
Сравниваем агентов PPO и GRPO (RL-часть, защищается как курсовая), позже — агента на LangChain,
а также пользу доступа к внешним источникам.

## С чего читать

| Документ | Что в нём |
|---|---|
| [docs/decisions.md](docs/decisions.md) | все согласованные решения и порядок реализации — **главный документ** |
| [docs/dsl_spec.md](docs/dsl_spec.md) | грамматика DSL, на котором модель пишет презентацию |
| [docs/metrics.md](docs/metrics.md) | метрики и ручная рубрика оценки |
| [docs/model_selection.md](docs/model_selection.md) | протокол выбора базовой модели (EXP-001) |
| [docs/experiments.md](docs/experiments.md) | журнал экспериментов |
| [data/examples/](data/examples/) | примеры презентаций в DSL для проверки парсера и сборщика |

## Как устроено

```text
тема ──► агент ──► текст в DSL ──► dsl.parse ──► презентация ──► builder ──► .pptx
                                                      │                      │
                                                      └────► metrics ◄───────┘
                                                               │
                                                           rewards ──► PPO / GRPO
```

## Карта файлов

```text
Diplom/
├── configs/            # модели, алгоритмы, эксперименты, профили запуска
├── data/
│   ├── examples/       # ручные примеры в DSL
│   ├── model_selection/# задания для выбора базовой модели
│   ├── sources/        # сохранённые ответы Википедии и данные пилотов (не в git)
│   └── train / validation / test .yaml
├── docs/
├── notebooks/          # эксперименты и запуск в Kaggle; код только импортируют из src/
├── runs/               # результаты запусков (не в git)
├── src/presentation_rl/
│   ├── dsl.py              # структура презентации и парсер DSL
│   ├── rendering/
│   │   ├── builder.py      # сборка PPTX
│   │   ├── visuals.py      # формулы и графики функций → PNG
│   │   └── images.py       # картинки из Wikimedia Commons
│   ├── metrics.py          # метрики
│   ├── rewards.py          # награда из выбранных метрик
│   ├── evaluation.py       # применение метрик к сохранённым ответам
│   ├── environment.py      # среда: промпт, инструменты, эпизод
│   ├── tools/
│   │   ├── wiki.py         # клиент API Википедии, статья по разделам
│   │   ├── cache.py        # сохранённые ответы Википедии
│   │   └── wiki_tools.py   # инструменты модели: search, open, read
│   ├── model_loader.py     # загрузка модели, выбор устройства
│   ├── training/
│   │   ├── common.py       # общая основа RL: rollout'ы, log-probs, KL, сохранение
│   │   ├── sft.py          # SFT-разогрев на примерах в DSL
│   │   ├── grpo.py
│   │   └── ppo.py
│   ├── experiments/model_selection.py
│   ├── config.py, cli.py
├── archive/            # код и тетрадки отменённых подходов (для отчётов), в проекте не используются
├── templates/          # оформление презентаций
└── tests/
```

Готовы парсер (`dsl.py`) и сборщик (`rendering/`), кроме поиска картинок: на месте `#IMAGE` пока заглушка.
Остальные модули — заглушки с описанием назначения; код пишем по шагам из `docs/decisions.md`.

## Сервер

Работа с сервером вуза: [docs/server.md](docs/server.md).

## Правила работы

- Эталонные функции живут только в `src/`; ноутбуки их импортируют и не содержат своих копий алгоритмов.
- Один шаг — один модуль, тест к нему и проверка перед переходом дальше.
- Метрика и награда — разные вещи: метрики показываем все, награду собираем из явно выбранных компонентов.
- Награду и гиперпараметры подбираем на train/validation; test — только для итоговых выводов.
- Исходные ответы моделей сохраняем без исправлений; неудачные попытки не выбрасываем из отчёта.
- Каждый запуск сохраняет конфигурацию, версии, устройство, seed и результаты в `runs/<run_id>/`
  и не перезаписывает прежние запуски.
- Код запускается на трёх профилях: `local` (GTX 1660 Super), `server` (Intel Arc A770), `kaggle` (T4);
  устройство выбирается автоматически: CUDA → XPU → CPU.

## Запуск

Из папки `Diplom`:

```bash
pip install -e ".[dev]"
pytest                          # тесты парсера и сборщика
```

Сборка презентации из текста DSL:

```python
from presentation_rl.rendering import build_from_text

text = open("data/examples/serf_reform.txt", encoding="utf-8").read()
deck, report = build_from_text(text, "runs/examples/serf_reform.pptx")
print(deck.parse_rate, report.status_counts())
```

`notebooks/explore_models.ipynb` — разведка: как исходные модели раскрывают тему без DSL.
