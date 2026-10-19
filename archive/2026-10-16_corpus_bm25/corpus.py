# Здесь собираем корпус Википедии под темы заданий и сохраняем выдачу живого поиска для подбора весов BM25.
"""Сборка корпуса Википедии.

Для каждой темы:

1. запросы — тема и короткие ключевые слова (``terms`` из тезисов, если есть); по каждому — поиск
   в русской Википедии (по теме — ``topic_k`` статей, по ключевым словам — ``term_k``); выдача
   сохраняется как есть, по ней потом подбираются веса нашего BM25. Длинные фразы (например, заголовки
   тезисов) поиск Википедии понимает плохо, поэтому запросы — короткие;
2. статьи, на которые ссылаются введения ``top_n`` лучших статей по теме, — до ``link_k`` с каждой;
3. английские версии всех найденных русских статей по межъязыковым ссылкам и поиск в английской
   Википедии по английским названиям ``top_n`` лучших статей по теме;
4. каждая статья скачивается один раз (общая для всех тем) и делится на разделы.

Файлы в папке корпуса: ``articles.jsonl`` — статьи по разделам, ``topics.jsonl`` — по теме: запросы,
выдача живого поиска, ключи статей. Сборку можно прервать и продолжить: уже скачанные статьи и уже
собранные темы повторно не запрашиваются.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .wiki import Article, WikiClient


@dataclass
class TopicPlan:
    id: str
    topic: str
    queries: list[str] = field(default_factory=list)  # короткие запросы, например terms из тезисов


class CorpusStore:
    def __init__(self, folder: str | Path, topics_name: str = "topics.jsonl"):
        """``topics_name`` — другой файл тем позволяет пересобрать темы по новым запросам, не скачивая статьи заново."""
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.articles_path = self.folder / "articles.jsonl"
        self.topics_path = self.folder / topics_name
        self.articles: dict[tuple[str, str], Article] = {}  # (язык, название) → статья
        self.missing: set[tuple[str, str]] = set()  # нет такой статьи или это страница значений
        self.topics: dict[str, dict] = {}
        if self.articles_path.exists():
            for line in self.articles_path.open(encoding="utf-8"):
                record = json.loads(line)
                if record.get("missing"):
                    self.missing.add((record["lang"], record["requested"]))
                    continue
                article = Article.from_json(record)
                self.articles[(article.lang, record.get("requested", article.title))] = article
                self.articles[(article.lang, article.title)] = article
        if self.topics_path.exists():
            for line in self.topics_path.open(encoding="utf-8"):
                record = json.loads(line)
                self.topics[record["id"]] = record

    def get_article(self, client: WikiClient, lang: str, title: str) -> Article | None:
        if (lang, title) in self.articles:
            return self.articles[(lang, title)]
        if (lang, title) in self.missing:
            return None
        article = client.article(lang, title)
        with self.articles_path.open("a", encoding="utf-8") as out:
            if article is None:
                self.missing.add((lang, title))
                out.write(json.dumps({"lang": lang, "requested": title, "missing": True}, ensure_ascii=False) + "\n")
                return None
            record = article.to_json() | {"requested": title}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.articles[(lang, title)] = article
        self.articles[(lang, article.title)] = article
        return article

    def save_topic(self, record: dict) -> None:
        self.topics[record["id"]] = record
        with self.topics_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    def unique_articles(self) -> list[Article]:
        return list({a.key: a for a in self.articles.values()}.values())


def collect_topic(
    client: WikiClient, store: CorpusStore, plan: TopicPlan,
    topic_k: int = 5, term_k: int = 3, top_n: int = 3, link_k: int = 10, en_k: int = 3,
) -> dict:
    """Собирает статьи одной темы; возвращает запись для ``topics.jsonl``."""
    live: dict[str, list[dict]] = {}
    ru_titles: list[str] = []

    def add(titles: list[str], title: str) -> None:
        if title not in titles:
            titles.append(title)

    for number, query in enumerate([plan.topic] + plan.queries):
        hits = client.search("ru", query, topic_k if number == 0 else term_k)
        live[query] = [{"title": h.title, "section": h.section} for h in hits]
        for h in hits:
            add(ru_titles, h.title)

    top = [h["title"] for h in live[plan.topic][:top_n]]
    for title in top:
        for linked in client.lead_links("ru", title, link_k):
            add(ru_titles, linked)

    ru_keys, en_titles = [], []
    for title in ru_titles:
        article = store.get_article(client, "ru", title)
        if article:
            ru_keys.append(article.key)
    links = client.langlinks("ru", ru_titles, "en")
    for title in links.values():
        add(en_titles, title)
    live_en: dict[str, list[dict]] = {}
    for title in top:
        if title in links:
            hits = client.search("en", links[title], en_k)
            live_en[links[title]] = [{"title": h.title, "section": h.section} for h in hits]
            for h in hits:
                add(en_titles, h.title)
    en_keys = []
    for title in en_titles:
        article = store.get_article(client, "en", title)
        if article:
            en_keys.append(article.key)

    return {
        "id": plan.id, "topic": plan.topic, "queries": [plan.topic] + plan.queries,
        "live_ru": live, "live_en": live_en, "top_ru": top,
        "articles_ru": ru_keys, "articles_en": en_keys,
        "collected": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def collect(client: WikiClient, store: CorpusStore, plans: list[TopicPlan], **kwargs) -> None:
    """Собирает все темы, пропуская уже собранные; печатает ход работы."""
    started = time.time()
    for number, plan in enumerate(plans, start=1):
        if plan.id in store.topics:
            print(f"[{number}/{len(plans)}] {plan.id}: уже собрана")
            continue
        record = collect_topic(client, store, plan, **kwargs)
        store.save_topic(record)
        print(
            f"[{number}/{len(plans)}] {plan.id}: лучшие по теме {record['top_ru']}, "
            f"статей ru {len(record['articles_ru'])}, en {len(record['articles_en'])}; "
            f"запросов к API {client.requests}, {time.time() - started:.0f} с"
        )
