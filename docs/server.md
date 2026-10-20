<!-- Здесь записываю порядок работы с сервером вуза: код, запуск, результаты. -->
# Сервер

Сервер вуза: 8×V100 32 ГБ (общий), диск 50 ГБ, вход по SSH через VPN (`ssh -p 2622 ryzhkova@195.19.43.12`). Репозиторий —
https://github.com/doshik722/Presentation (открытый).

- Код — только на компьютере → push; на сервере `git pull`, ничего не правлю.
- На сервере репозиторий склонирован по HTTPS без входа: только `git pull`, push оттуда невозможен.
- Результаты (`runs/`) в git не идут — забираю через Download в окне SSH или `scp`.
- Веса, чекпоинты, кэш HF, ответы Википедии — только на сервере. Чекпоинты: последний и лучший.
- Пароли и токены — не в репозиторий.

## Настройка сервера (один раз)

```
mv ~/Diplom ~/Diplom_old                  # старая копия из архива
git clone https://github.com/doshik722/Presentation.git ~/Diplom
~/venvs/diplom/bin/pip install -e "$HOME/Diplom[models]"
echo 'export WIKI_CONTACT="…"' >> ~/.bashrc
cp -rn ~/Diplom_old/runs/. ~/Diplom/runs/  # runs/ и ответов Википедии в git нет — из старой копии
mkdir -p ~/Diplom/data/sources && cp -rn ~/Diplom_old/data/sources/wiki_cache ~/Diplom/data/sources/
```

## Запуск

```
cd ~/Diplom && git pull
nvidia-smi                                # свободная карта → CUDA_VISIBLE_DEVICES в тетрадке
tmux new -s run                           # долгое — только в tmux; Ctrl+B D — выйти, tmux attach -t run — вернуться
~/venvs/diplom/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 notebooks/<тетрадка>.ipynb --output <тетрадка>_run.ipynb
```

## Результаты на компьютер

```
scp -P 2622 -r ryzhkova@195.19.43.12:~/Diplom/runs/<…>/<время> runs/<…>/
```
