# Здесь ищем разделы статей по корпусу Википедии: BM25 по разделам с весами названия статьи и заголовка раздела.
"""Поиск по корпусу Википедии для обучения.

Повторяет устройство поиска Википедии (CirrusSearch), насколько это просто сделать:

- слова приводятся к основе стеммером Snowball — его же использует анализатор Elasticsearch для русского
  и английского, на котором работает поиск Википедии;
- документ — раздел статьи; поля: текст раздела, название статьи, заголовок раздела и текст всей статьи
  (поиск Википедии ранжирует статьи целиком, поэтому совпадение в других разделах тоже даёт вклад);
- оценка — BM25F: частоты слов по полям складываются с весами, длина нормируется только у текстовых полей.

Чего нет: популярности статьи (число входящих ссылок, просмотры) — у Википедии она заметно влияет на порядок.
Веса полей подбираются по совпадению с живой выдачей (``data/sources/wiki_pilot/topics_v2.jsonl``).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .wiki import Article

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Самые частые служебные слова; остальное гасит idf.
STOP_WORDS = {
    "ru": set("и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было "
              "вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас "
              "нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их "
              "чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой "
              "совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при "
              "наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве "
              "три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда "
              "конечно всю между это также который которые которая которого является".split()),
    "en": set("a an and are as at be but by for from has have he her his i in is it its of on or she that the "
              "their them they this to was were which who will with".split()),
}
_STEMMER_LANG = {"ru": "russian", "en": "english"}


class Analyzer:
    """Текст → основы слов: нижний регистр, ё → е, без служебных слов, стеммер Snowball (с кэшем)."""

    def __init__(self, lang: str):
        import snowballstemmer  # pip install snowballstemmer

        self.lang = lang
        self._stemmer = snowballstemmer.stemmer(_STEMMER_LANG[lang])
        self._stop = STOP_WORDS[lang]
        self._cache: dict[str, str] = {}

    def __call__(self, text: str) -> list[str]:
        result = []
        for word in _WORD_RE.findall(text.lower().replace("ё", "е")):
            if word in self._stop:
                continue
            stem = self._cache.get(word)
            if stem is None:
                stem = self._cache[word] = self._stemmer.stemWord(word)
            result.append(stem)
        return result


@dataclass
class Weights:
    """Параметры BM25F. ``text`` и ``article`` нормируются по длине (``b``), ``title`` и ``heading`` — нет."""
    # Подобраны 16.10.2026 на пилоте (notebooks/wiki_search_calibration.ipynb): совпадение с живой выдачей
    # в топ-3 — 0.66, первое место совпадает в 59% запросов. Соседние значения дают то же в пределах ±0.01.
    k1: float = 1.2
    b: float = 0.75
    text: float = 1.0
    title: float = 20.0
    heading: float = 3.0
    article: float = 0.3


@dataclass
class Hit:
    article: Article
    section: int  # номер раздела в статье
    score: float


class SectionIndex:
    """Обратный индекс по разделам статей одного языка."""

    FIELDS = ("text", "title", "heading")

    def __init__(self, articles: Iterable[Article], lang: str, weights: Weights | None = None):
        self.lang = lang
        self.weights = weights or Weights()
        self.analyze = Analyzer(lang)
        self.articles: list[Article] = []
        self.sections: list[tuple[int, int]] = []  # раздел → (номер статьи, номер раздела в ней)
        self.text_len: list[int] = []
        self.article_len: list[int] = []
        # слово → {раздел: частота} для полей раздела, слово → {статья: частота} для всей статьи
        self.postings = {f: defaultdict(dict) for f in self.FIELDS}
        self.article_postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.df: dict[str, int] = defaultdict(int)  # в скольких разделах слово есть хотя бы в одном поле

        seen = set()
        for article in articles:
            if article.lang != lang or article.key in seen:
                continue
            seen.add(article.key)
            a = len(self.articles)
            self.articles.append(article)
            title = self.analyze(article.title)
            total: dict[str, int] = defaultdict(int)
            length = 0
            for section in article.sections:
                s = len(self.sections)
                self.sections.append((a, section.index))
                fields = {"text": self.analyze(section.text), "title": title, "heading": self.analyze(section.heading)}
                self.text_len.append(len(fields["text"]))
                length += len(fields["text"])
                for name, tokens in fields.items():
                    for token in tokens:
                        self.postings[name][token][s] = self.postings[name][token].get(s, 0) + 1
                for token in set().union(*fields.values()):
                    self.df[token] += 1
                for token in fields["text"]:
                    total[token] += 1
            for token, count in total.items():
                self.article_postings[token][a] = count
            self.article_len.append(length)
        self.avg_text = sum(self.text_len) / max(len(self.text_len), 1)
        self.avg_article = sum(self.article_len) / max(len(self.article_len), 1)
        # раздел по номеру статьи — чтобы поле «вся статья» давало вклад всем её разделам
        self.article_sections: list[list[int]] = [[] for _ in self.articles]
        for s, (a, _) in enumerate(self.sections):
            self.article_sections[a].append(s)

    def idf(self, token: str) -> float:
        n = len(self.sections)
        df = self.df.get(token, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str, weights: Weights | None = None) -> dict[int, float]:
        """Оценки разделов, где есть хотя бы одно слово запроса."""
        w = weights or self.weights
        tokens = list(dict.fromkeys(self.analyze(query)))
        scores: dict[int, float] = defaultdict(float)
        for token in tokens:
            tf: dict[int, float] = defaultdict(float)
            for s, count in self.postings["text"].get(token, {}).items():
                tf[s] += w.text * count / (1 - w.b + w.b * self.text_len[s] / self.avg_text)
            for name in ("title", "heading"):
                weight = getattr(w, name)
                for s, count in self.postings[name].get(token, {}).items():
                    tf[s] += weight * count
            if w.article:
                for a, count in self.article_postings.get(token, {}).items():
                    value = w.article * count / (1 - w.b + w.b * self.article_len[a] / self.avg_article)
                    for s in self.article_sections[a]:
                        tf[s] += value
            idf = self.idf(token)
            for s, value in tf.items():
                scores[s] += idf * value / (w.k1 + value)
        return scores

    def search(self, query: str, k: int = 3, weights: Weights | None = None) -> list[Hit]:
        """Лучшие разделы, не больше одного из статьи."""
        scores = self.score(query, weights)
        hits: list[Hit] = []
        used = set()
        for s in sorted(scores, key=lambda s: (-scores[s], s)):
            a, index = self.sections[s]
            if a in used:
                continue
            used.add(a)
            hits.append(Hit(self.articles[a], index, scores[s]))
            if len(hits) == k:
                break
        return hits
