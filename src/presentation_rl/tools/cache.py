# Здесь сохраняем ответы API Википедии: повторный запрос берётся из файла, а не из сети.
"""Сохранённые ответы Википедии.

Каждый ответ API записывается в папку и повторно не запрашивается:

- ``searches.jsonl`` — выдача поиска: язык, запрос, лимит, язык межъязыковых ссылок и статьи (без текста);
- ``pages.jsonl`` — статьи по названиям (введение, номер правки, URL); нет статьи — ``page: null``;
- ``articles.jsonl`` — статьи целиком по разделам, с таблицами и списками в колонках (``tables`` — версия формата
  ``ARTICLE_FORMAT``); записи старого формата не читаются — такие статьи запрашиваются заново: 20.10.2026 добавились
  таблицы (``tables: true``), 29.10.2026 — списки в колонках (``tables: 2``).

Так роллауты одного задания видят одну и ту же выдачу, а повторные прогоны и сравнения моделей идут по тем же
ответам. ``client=None`` — только сохранённое: промах — ошибка ``CacheMiss`` (для точного воспроизведения).
Обращения к Википедии идут по одному (общая блокировка), запись в файлы — построчно, сборку можно прервать.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .wiki import Article, PageInfo, WikiClient


ARTICLE_FORMAT = 2  # 1 (true) — таблицы и карточка; 2 — ещё списки в колонках (wiki_tables.list_text)


class CacheMiss(LookupError):
    """Ответа нет среди сохранённых, а обращаться к Википедии нельзя."""


class WikiCache:
    def __init__(self, folder: str | Path, client: WikiClient | None = None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.client = client
        self._lock = threading.Lock()
        self.searches: dict[tuple, list[PageInfo]] = {}
        self.page_infos: dict[tuple[str, str], PageInfo | None] = {}
        self.articles: dict[tuple[str, str], Article | None] = {}  # (язык, запрошенное или настоящее название)
        self.misses = 0  # сколько раз пришлось обращаться к Википедии
        for record in self._read("searches.jsonl"):
            key = (record["lang"], record["query"], record["limit"], record["target"])
            self.searches[key] = [PageInfo(**p) for p in record["pages"]]
        for record in self._read("pages.jsonl"):
            page = PageInfo(**record["page"]) if record["page"] else None
            self.page_infos[(record["lang"], record["requested"])] = page
        for record in self._read("articles.jsonl"):
            if not record.get("missing") and int(record.get("tables", 0)) < ARTICLE_FORMAT:
                continue  # сохранено в старом формате (без таблиц или без списков в колонках) — запросим заново
            if record.get("missing"):
                self.articles[(record["lang"], record["requested"])] = None
                continue
            article = Article.from_json(record)
            self.articles[(article.lang, record.get("requested", article.title))] = article
            self.articles[(article.lang, article.title)] = article

    # --- файлы

    def _read(self, name: str) -> list[dict]:
        path = self.folder / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

    def _append(self, name: str, record: dict) -> None:
        with (self.folder / name).open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _client(self, what: str) -> WikiClient:
        if self.client is None:
            raise CacheMiss(what)
        self.misses += 1
        return self.client

    # --- запросы

    def search(self, lang: str, query: str, limit: int = 6, target: str = "") -> list[PageInfo]:
        key = (lang, query.strip(), limit, target)
        with self._lock:
            if key not in self.searches:
                pages = self._client(f"search {key}").search_pages(lang, key[1], limit, target)
                self.searches[key] = pages
                self._append("searches.jsonl", {"lang": lang, "query": key[1], "limit": limit, "target": target,
                                                "pages": [p.to_json() for p in pages]})
            return self.searches[key]

    def pages(self, lang: str, titles: list[str]) -> dict[str, PageInfo | None]:
        with self._lock:
            missing = [t for t in dict.fromkeys(titles) if (lang, t) not in self.page_infos]
            if missing:
                for title, page in self._client(f"pages {lang} {missing}").pages(lang, missing).items():
                    self.page_infos[(lang, title)] = page
                    self._append("pages.jsonl", {"lang": lang, "requested": title,
                                                 "page": page.to_json() if page else None})
            return {t: self.page_infos[(lang, t)] for t in titles}

    def article(self, lang: str, title: str) -> Article | None:
        with self._lock:
            if (lang, title) not in self.articles:
                article = self._client(f"article {lang} {title}").article(lang, title)
                self.articles[(lang, title)] = article
                if article is None:
                    self._append("articles.jsonl", {"lang": lang, "requested": title, "missing": True})
                else:
                    self.articles[(lang, article.title)] = article
                    self._append("articles.jsonl", article.to_json() | {"requested": title, "tables": ARTICLE_FORMAT})
            return self.articles[(lang, title)]
