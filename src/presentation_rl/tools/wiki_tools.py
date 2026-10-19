# Здесь реализуем инструменты модели search, open и read поверх сохранённых ответов Википедии.
"""Инструменты модели для одного эпизода (эпизод — один прогон модели на одном задании).

- ``search(запрос)`` — до 3 статей из поиска Википедии на языке запроса (кириллица — русская, иначе английская)
  и до 3 их версий на другом языке; по каждой — id статьи, язык, название, начало введения;
- ``open(id статьи)`` — введение и полное оглавление с id разделов и подводками разделов, у которых есть
  подразделы; всё в пределах ``limit`` символов (обрезаются только самые длинные тексты);
- ``read(id раздела)`` — текст одного раздела без подразделов, до ``limit`` символов.

Id живут только внутри эпизода: статьи — ``d1``, ``d2``… по порядку появления, разделы — ``d1#3`` (номер раздела
в статье, ``#0`` — введение). Результаты — простой текст для модели. Таблица id и проверка ``#SOURCE`` — для сборщика.
"""
from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field

from .cache import WikiCache
from .wiki import Article, PageInfo

_CYRILLIC = re.compile("[а-яё]", re.IGNORECASE)
_REF = re.compile(r"^d(\d+)(?:#(\d+))?$")

# Описания для промпта модели (формат описания функций, как в шаблонах чата с инструментами).
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "search",
        "description": "Поиск статей Википедии. Запрос — 1–4 ключевых слова, а не фраза. Кириллица — поиск по русской "
                       "Википедии, латиница — по английской; в ответе также версии найденных статей на другом языке. "
                       "Возвращает id статей (d1, d2…), названия и начало введения.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "open",
        "description": "Открыть статью по id из search: введение и оглавление с id разделов (d1#3).",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    }},
    {"type": "function", "function": {
        "name": "read",
        "description": "Прочитать раздел статьи по id из оглавления (d1#3).",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
    }},
]


@dataclass
class _Doc:
    number: int
    page: PageInfo
    opened: bool = False
    read: set[int] = field(default_factory=set)

    @property
    def id(self) -> str:
        return f"d{self.number}"


class WikiTools:
    def __init__(self, cache: WikiCache, max_calls: int = 10, limit: int = 6000, per_lang: int = 3,
                 snippet: int = 200):
        self.cache = cache
        self.max_calls = max_calls
        self.limit = limit
        self.per_lang = per_lang
        self.snippet = snippet
        self.reset()

    def reset(self) -> None:
        """Новый эпизод: id начинаются заново."""
        self.docs: list[_Doc] = []
        self._by_key: dict[tuple[str, int], _Doc] = {}
        self.calls: list[dict] = []

    # --- вызов по имени (так его выполняет среда)

    def call(self, name: str, argument: str) -> str:
        if len(self.calls) >= self.max_calls:
            return f"Лимит вызовов исчерпан ({self.max_calls}). Пишите презентацию по уже найденному."
        handler = {"search": self.search, "open": self.open, "read": self.read}.get(name)
        result = handler(argument) if handler else f"Нет инструмента «{name}». Есть: search, open, read."
        self.calls.append({"name": name, "argument": argument, "chars": len(result)})
        return result

    # --- инструменты

    def search(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "Пустой запрос."
        lang = "ru" if _CYRILLIC.search(query) else "en"
        other = "en" if lang == "ru" else "ru"
        found = [p for p in self.cache.search(lang, query, self.per_lang * 2, other) if not p.disambiguation]
        found = found[:self.per_lang]
        linked = [p.langlink for p in found if p.langlink]
        others = [p for p in self.cache.pages(other, linked).values() if p and not p.disambiguation] if linked else []
        docs = [self._doc(p) for p in found + others]
        if not docs:
            return f"По запросу «{query}» ничего не найдено. Попробуйте другие ключевые слова."
        lines = [f"Результаты поиска «{query}»:"]
        lines += [f"{d.id} | {d.page.lang} | {d.page.title} | {_shorten(d.page.intro, self.snippet)}" for d in docs]
        return "\n".join(lines)

    def open(self, ref: str) -> str:
        doc, _, error = self._resolve(ref)
        if error:
            return error
        article = self._article(doc)
        if article is None:
            return f"Статья {doc.id} сейчас недоступна."
        doc.opened = True
        sections = article.sections
        heads = [f"{doc.id}#{i} " + "  " * (s.level - 1) + (s.heading or "Введение") for i, s in enumerate(sections)]
        # тексты в оглавлении: введение и подводки разделов, у которых есть подразделы
        texts = {i: s.text for i, s in enumerate(sections) if s.text and (i == 0 or _has_children(sections, i))}
        title = f"{doc.id} | {doc.page.lang} | {article.title}"
        fixed = len(title) + sum(len(h) + 1 for h in heads) + 2 * len(texts)
        cap = _fit_cap([len(t) for t in texts.values()], self.limit - fixed - 40 * len(texts))
        lines = [title]
        for i, head in enumerate(heads):
            lines.append(head)
            if i in texts:
                text = texts[i]
                if cap is not None and len(text) > cap:
                    text = _shorten(text, max(cap, 0), keep_lines=True) + f" [дальше: read {doc.id}#{i}]"
                lines.append(text)
        result = "\n".join(lines)
        if len(result) > self.limit:  # огромное оглавление: не помещаются даже заголовки
            result = result[:self.limit].rsplit("\n", 1)[0] + "\n[оглавление обрезано]"
        return result

    def read(self, ref: str) -> str:
        doc, index, error = self._resolve(ref)
        if error:
            return error
        article = self._article(doc)
        if article is None:
            return f"Статья {doc.id} сейчас недоступна."
        index = index or 0
        if index >= len(article.sections):
            return f"В статье {doc.id} нет раздела #{index}; разделы — от #0 до #{len(article.sections) - 1}."
        doc.read.add(index)
        section = article.sections[index]
        head = f"{doc.id}#{index} | {doc.page.lang} | {article.title} — {section.heading or 'Введение'}"
        text = section.text
        if not text:
            children = [f"{doc.id}#{j} {article.sections[j].heading}" for j in _children(article.sections, index)]
            return head + "\nУ раздела нет собственного текста. Подразделы: " + "; ".join(children) + "."
        if len(text) > self.limit:
            cut = text.rfind("\n", 0, self.limit)
            if cut < self.limit // 2:
                cut = text.rfind(". ", 0, self.limit) + 1
            if cut <= 0:
                cut = self.limit
            text = text[:cut].rstrip() + f"\n[обрезано: ещё {len(text) - cut} символов]"
        return head + "\n" + text

    # --- источники (для сборщика)

    def valid_source(self, ref: str) -> bool:
        """Id для ``#SOURCE``: статья — если была в выдаче, раздел — если статья открыта или раздел прочитан."""
        match = _REF.match(ref.strip())
        if not match or not 1 <= int(match.group(1)) <= len(self.docs):
            return False
        doc = self.docs[int(match.group(1)) - 1]
        if match.group(2) is None:
            return True
        index = int(match.group(2))
        if index in doc.read:
            return True
        article = self.cache.articles.get((doc.page.lang, doc.page.title)) if doc.opened else None
        return article is not None and index < len(article.sections)

    def source(self, ref: str) -> dict | None:
        """Название, раздел, ссылка и номер правки для оформления источника; None — id недопустим."""
        if not self.valid_source(ref):
            return None
        number, _, index = ref.strip()[1:].partition("#")
        doc = self.docs[int(number) - 1]
        page = doc.page
        result = {"lang": page.lang, "title": page.title, "url": page.url, "revid": page.revid, "section": ""}
        if index:
            article = self._article(doc)
            section = article.sections[int(index)]
            result["revid"] = article.revid
            if section.heading:
                result["section"] = section.heading
                result["url"] = page.url + "#" + urllib.parse.quote(section.heading.replace(" ", "_"))
        return result

    def table(self) -> dict[str, dict]:
        """Таблица id эпизода — пишется в лог вместе с ответом модели."""
        return {d.id: {"lang": d.page.lang, "title": d.page.title, "pageid": d.page.pageid,
                       "opened": d.opened, "read": sorted(d.read)} for d in self.docs}

    # --- служебное

    def _doc(self, page: PageInfo) -> _Doc:
        key = (page.lang, page.pageid)
        if key not in self._by_key:
            doc = _Doc(len(self.docs) + 1, page)
            self.docs.append(doc)
            self._by_key[key] = doc
        return self._by_key[key]

    def _resolve(self, ref: str) -> tuple[_Doc | None, int | None, str]:
        match = _REF.match(ref.strip())
        if not match:
            return None, None, f"Неверный id «{ref}»: статьи — d1, d2… (из search), разделы — d1#3 (из open)."
        number = int(match.group(1))
        if not 1 <= number <= len(self.docs):
            return None, None, f"Статьи d{number} в этом поиске не было."
        index = int(match.group(2)) if match.group(2) is not None else None
        return self.docs[number - 1], index, ""

    def _article(self, doc: _Doc) -> Article | None:
        return self.cache.article(doc.page.lang, doc.page.title)


def _has_children(sections, i: int) -> bool:
    return i + 1 < len(sections) and sections[i + 1].level > sections[i].level


def _children(sections, i: int) -> list[int]:
    """Прямые подразделы раздела ``i``."""
    result = []
    for j in range(i + 1, len(sections)):
        if sections[j].level <= sections[i].level:
            break
        if sections[j].level == sections[i].level + 1:
            result.append(j)
    return result


def _fit_cap(lengths: list[int], budget: int) -> int | None:
    """Наибольший порог c, при котором sum(min(длина, c)) <= budget; None — всё помещается без обрезки."""
    if sum(lengths) <= budget:
        return None
    low, high = 0, max(lengths)
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(n, middle) for n in lengths) <= budget:
            low = middle
        else:
            high = middle - 1
    return low


def _shorten(text: str, limit: int, keep_lines: bool = False) -> str:
    """Обрезка по границе слова с многоточием; без ``keep_lines`` переносы строк заменяются пробелами."""
    if not keep_lines:
        text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[:cut if cut > limit // 2 else limit].rstrip() + "…"
