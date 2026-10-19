<!-- Здесь записываю порядок работы с сервером вуза: код, запуск, результаты. -->
# Сервер

Сервер вуза: 8×V100 32 ГБ (общий), диск 50 ГБ, вход по SSH через VPN (`ssh -p 2622 ryzhkova@195.19.43.12`). Репозиторий —
https://github.com/doshik722/Presentation (приватный).

- Код — только на компьютере → push; на сервере `git pull`, ничего не правлю.
- Ключ сервера для GitHub — только чтение (deploy key без write access).
- Результаты (`runs/`) в git не идут — забираю через Download в окне SSH или `scp`.
- Веса, чекпоинты, кэш HF, ответы Википедии — только на сервере. Чекпоинты: последний и лучший.
- Пароли и токены — не в репозиторий.

## Настройка сервера (один раз)

```
ssh-keygen -t ed25519 -C "vuz-v100" -f ~/.ssh/github_presentation -N ""
cat ~/.ssh/github_presentation.pub        # → GitHub: Settings → Deploy keys, без write access
printf 'Host github.com\n  IdentityFile ~/.ssh/github_presentation\n  IdentitiesOnly yes\n' >> ~/.ssh/config
chmod 600 ~/.ssh/config ~/.ssh/github_presentation
ssh -T git@github.com                     # порт 22 закрыт → Hostname ssh.github.com, Port 443

mv ~/Diplom ~/Diplom_old
git clone git@github.com:doshik722/Presentation.git ~/Diplom
~/venvs/diplom/bin/pip install -e "$HOME/Diplom[models]"
echo 'export WIKI_CONTACT="…"' >> ~/.bashrc
```

## Запуск

```
cd ~/Diplom && git pull
nvidia-smi                                # свободная карта → CUDA_VISIBLE_DEVICES в тетрадке
tmux new -s run                           # долгое — только в tmux; Ctrl+B D — выйти, tmux attach -t run — вернуться
~/venvs/diplom/bin/jupyter nbconvert --to notebook --execute notebooks/<тетрадка>.ipynb --output <тетрадка>_run.ipynb
```

## Результаты на компьютер

```
scp -P 2622 -r ryzhkova@195.19.43.12:~/Diplom/runs/<…>/<время> runs/<…>/
```
