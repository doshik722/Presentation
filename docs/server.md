<!-- Здесь записываю порядок работы с сервером вуза: код, запуск, результаты. -->
# Сервер

С 01.11.2026 — облачная платформа МГТУ TeraCloud (https://panel.teracloud.bmstu.ru, вход через SSO, из сети МГТУ или
через VPN). Прежний сервер (195.19.43.12) закрыт. Пространство `diplom`, профиль GPU Small; квоты проекта увеличены
06.11.2026: 2×V100 32 ГБ, 128 ГБ памяти, домашний каталог 100 ГБ (JuiceFS, сохраняется после остановки), `/shared` 10 ГБ,
доступ к Hugging Face. Корень контейнера (`/`, `/tmp`) стирается при остановке. Работа — через JupyterLab (терминал,
тетрадки); SSH-доступа в контейнер нет. Python 3.11 и PyTorch с CUDA — в образе. Репозиторий —
https://github.com/doshik722/Presentation (открытый).

- Код — только на компьютере → push; на сервере `git pull`, ничего не правлю.
- Результаты (`runs/`) в git не идут — забираю через JupyterLab (правой кнопкой → Download) или архивом.
- Веса и кеш Hugging Face — в `~/hf` (`HF_HOME`), ответы Википедии — в `data/sources/wiki_cache`.
- Пароли и токены — не в репозиторий.

## Настройка пространства (один раз)

```
cd ~ && git clone https://github.com/doshik722/Presentation.git Diplom
cd Diplom && pip install --user -e ".[models,training,reward]" pytest "brotli>=1.2"
echo 'export HF_HOME=$HOME/hf' >> ~/.bashrc && source ~/.bashrc
python -c "from huggingface_hub import snapshot_download as s; print(s('t-tech/T-lite-it-2.1', allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja']))"
```

Пакеты — только с `--user` (в домашний каталог), иначе пропадут при остановке пространства. `brotli>=1.2` — новая
`huggingface_hub` распаковывает ответы с параметром, которого нет в `brotli` образа (`process() takes no keyword
arguments`). Ответы T-lite из EXP-010 и EXP-012 (`runs/variance/20261020_0331`, `runs/reward_sources/20261027_2150`)
перенесены архивом с компьютера.

## Запуск

```
cd ~/Diplom && git pull
nvidia-smi                                # какая карта свободна → CUDA_VISIBLE_DEVICES в тетрадке
cd notebooks && jupyter nbconvert --to script <тетрадка>.ipynb
nohup python -u <тетрадка>.py > $HOME/<тетрадка>.log 2>&1 &
tail -f $HOME/<тетрадка>.log              # Ctrl+C — выйти из просмотра, работа продолжается
```

Тетрадка запускается скриптом в фоне: закрытый браузер её не прерывает, ход — в журнале.
