# Здесь обращаемся к Википедии через её API: поиск, статья по разделам, межъязыковые ссылки, ссылки из введения.
"""Клиент API Википедии (MediaWiki Action API).

Нужен в двух местах: при сборке корпуса для обучения и как живой поиск при использовании.
Правила API Википедии: в заголовке User-Agent — название проекта и контакт (переменная окружения
``WIKI_CONTACT``), запросы по одному с паузой, повтор с ожиданием при перегрузке (429, 5xx, maxlag).

Статья хранится по разделам: введение — раздел 0, дальше разделы в порядке текста; служебные разделы
(«Примечания», «Литература», «Ссылки» и т. п.) отбрасываются вместе с подразделами.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Callable

USER_AGENT = "presentation-rl/0.1 (diploma research; contact: {contact})"

# Служебные разделы: в корпус не берём ни их, ни их подразделы.
SKIP_HEADINGS = {
    "примечания", "литература", "ссылки", "см. также", "источники", "комментарии", "сноски",
    "библиография", "дополнительная литература", "внешние ссылки",
    "references", "see also", "external links", "further reading", "notes", "bibliography",
    "sources", "citations", "footnotes", "notes and references",
}
_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")

_DISPLAY = "{\\displaystyle"
_JOIN_AFTER = (" ", ".", ",", ";", ":", ")", "!", "?")


def clean_math(text: str) -> str:
    """Формулы в простом тексте Википедии приходят «разобранными» MathML (по символу на строке с отступом),
    а следом — LaTeX в ``{\\displaystyle …}``. Оставляем только LaTeX в ``$…$`` в строке текста."""
    if _DISPLAY not in text:
        return text
    out: list[str] = []
    after_formula = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(_DISPLAY) and stripped.endswith("}"):
            formula = "$" + stripped[len(_DISPLAY):-1].strip() + "$"
            # строки разобранного MathML перед формулой: пустые и с отступом от двух пробелов
            while out and (not out[-1].strip() or out[-1].startswith("  ")):
                out.pop()
            if out:
                out[-1] = out[-1].rstrip() + " " + formula
            else:
                out.append(formula)
            after_formula = True
            continue
        if after_formula:
            if not stripped:
                continue  # хвост MathML или пустая строка сразу после формулы
            if line.startswith(_JOIN_AFTER):
                out[-1] += line.rstrip() if line.startswith(" ") else line.strip()
                after_formula = False
                continue
            after_formula = False
        out.append(line)
    return "\n".join(out)



@dataclass
class Section:
    index: int  # 0 — введение
    level: int  # 1 — введение и разделы верхнего уровня (== … ==), 2 — подразделы (=== … ===) и т. д.
    heading: str
    text: str


@dataclass
class Article:
    lang: str
    pageid: int
    title: str
    revid: int
    url: str
    sections: list[Section] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.lang}:{self.pageid}"

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Article":
        # формулы чистим и при загрузке: статьи, сохранённые до clean_math, тоже читаются без мусора MathML
        sections = [Section(**(s | {"text": clean_math(s["text"])})) for s in data["sections"]]
        return cls(data["lang"], data["pageid"], data["title"], data["revid"], data["url"], sections)


@dataclass
class PageInfo:
    """Статья без текста: то, что нужно для выдачи search и для ссылки на источник."""
    lang: str
    title: str
    pageid: int
    revid: int
    url: str
    intro: str  # введение статьи простым текстом
    disambiguation: bool = False  # страница значений
    langlink: str = ""  # название той же статьи на другом языке (если запрашивали)

    def to_json(self) -> dict:
        return asdict(self)


def _page_info(lang: str, page: dict) -> PageInfo:
    links = page.get("langlinks") or [{}]
    return PageInfo(
        lang, page["title"], page["pageid"], page.get("lastrevid", 0), page.get("fullurl", ""),
        clean_math(page.get("extract", "")).strip(), "disambiguation" in page.get("pageprops", {}), links[0].get("title", ""),
    )


_INTRO_PARAMS = {
    "prop": "extracts|pageprops|info", "exintro": "1", "explaintext": "1", "exlimit": "max",
    "ppprop": "disambiguation", "inprop": "url", "redirects": "1",
}


@dataclass
class SearchHit:
    title: str
    pageid: int
    section: str  # заголовок раздела, где совпадение лучше всего; пусто — совпадение во введении или в названии
    snippet: str


def split_sections(text: str) -> list[Section]:
    """Делит простой текст статьи с заголовками вида ``== Раздел ==`` на разделы; служебные отбрасывает."""
    sections = [Section(0, 1, "", "")]
    lines: list[str] = []
    skip_level: int | None = None  # уровень служебного раздела, подразделы которого тоже пропускаем

    def close() -> None:
        sections[-1].text = clean_math("\n".join(lines)).strip()
        lines.clear()

    for line in text.splitlines():
        match = _HEADING_RE.match(line.strip())
        if not match:
            if skip_level is None:
                lines.append(line)
            continue
        level = len(match.group(1)) - 1
        heading = match.group(2).strip()
        if skip_level is not None and level > skip_level:
            continue
        skip_level = None
        if heading.lower() in SKIP_HEADINGS:
            close()
            skip_level = level
            continue
        close()
        sections.append(Section(len(sections), level, heading, ""))
    if skip_level is None:
        close()
    # Пустые разделы (заголовок без текста и без подразделов с текстом) не нужны модели.
    kept = [s for i, s in enumerate(sections) if s.text or s.index == 0 or _has_text_below(sections, i)]
    for new_index, section in enumerate(kept):
        section.index = new_index
    return kept


def _has_text_below(sections: list[Section], i: int) -> bool:
    level = sections[i].level
    for s in sections[i + 1:]:
        if s.level <= level:
            return False
        if s.text:
            return True
    return False


def _clean_snippet(snippet: str) -> str:
    return re.sub(r"<[^>]+>", "", snippet).replace("&quot;", '"').replace("&amp;", "&").strip()


class WikiClient:
    """Запросы к API Википедии с паузой между ними и повтором при перегрузке."""

    def __init__(
        self,
        contact: str | None = None,
        min_interval: float = 0.5,
        retries: int = 4,
        fetch: Callable[[str, dict], dict] | None = None,
    ):
        contact = contact or os.environ.get("WIKI_CONTACT", "")
        if not contact and fetch is None:
            raise ValueError("укажите контакт для User-Agent: WikiClient(contact=...) или переменная WIKI_CONTACT")
        self.headers = {"User-Agent": USER_AGENT.format(contact=contact)}
        self.min_interval = min_interval
        self.retries = retries
        self._fetch = fetch or self._http_get
        self._last = 0.0
        self.requests = 0  # попытки, включая повторы
        self.failures: list[str] = []  # почему попытки не удались — чтобы видеть, откуда задержки

    # --- транспорт

    def _http_get(self, url: str, headers: dict) -> dict:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def get(self, lang: str, params: dict) -> dict:
        params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
        delay = 2.0
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests += 1
            try:
                data = self._fetch(url, self.headers)
            except urllib.error.HTTPError as exc:
                self.failures.append(f"HTTP {exc.code}")
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as exc:
                self.failures.append(f"сеть: {getattr(exc, 'reason', exc)}")
                if attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
            if data.get("error", {}).get("code") == "maxlag" and attempt < self.retries:
                self.failures.append("maxlag: серверы Википедии перегружены")
                time.sleep(delay)
                delay *= 2
                continue
            if "error" in data:
                raise RuntimeError(f"API Википедии: {data['error']}")
            return data
        raise RuntimeError("API Википедии не ответило")

    # --- запросы

    def search(self, lang: str, query: str, limit: int = 5) -> list[SearchHit]:
        """Встроенный поиск Википедии по статьям (основное пространство имён)."""
        data = self.get(lang, {
            "action": "query", "list": "search", "srsearch": query, "srlimit": str(limit),
            "srnamespace": "0", "srprop": "snippet|sectiontitle",
        })
        return [
            SearchHit(h["title"], h["pageid"], h.get("sectiontitle", ""), _clean_snippet(h.get("snippet", "")))
            for h in data.get("query", {}).get("search", [])
        ]

    def search_pages(self, lang: str, query: str, limit: int = 6, target: str = "") -> list[PageInfo]:
        """Поиск Википедии одним запросом: статьи в порядке выдачи с введением, признаком страницы значений
        и (если задан ``target``) названием той же статьи на языке ``target``."""
        params = {
            "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": str(limit),
            "gsrnamespace": "0", **_INTRO_PARAMS,
        }
        if target:
            params.update(prop=_INTRO_PARAMS["prop"] + "|langlinks", lllang=target, lllimit="max")
        pages = self.get(lang, params).get("query", {}).get("pages", [])
        pages = sorted(pages, key=lambda p: p.get("index", 0))
        return [_page_info(lang, p) for p in pages if not p.get("missing")]

    def pages(self, lang: str, titles: list[str]) -> dict[str, PageInfo | None]:
        """Статьи по названиям (до 20 за запрос): {запрошенное название: статья или None, если её нет}."""
        result: dict[str, PageInfo | None] = {}
        for start in range(0, len(titles), 20):
            chunk = titles[start:start + 20]
            query = self.get(lang, {"action": "query", "titles": "|".join(chunk), **_INTRO_PARAMS}).get("query", {})
            renamed = {x["from"]: x["to"] for x in query.get("normalized", []) + query.get("redirects", [])}
            found = {p["title"]: _page_info(lang, p) for p in query.get("pages", []) if not p.get("missing")}
            for title in chunk:
                resolved = title
                while resolved in renamed:
                    resolved = renamed[resolved]
                result[title] = found.get(resolved)
        return result

    def article(self, lang: str, title: str) -> Article | None:
        """Статья целиком, по разделам; None — нет такой статьи или это страница значений."""
        data = self.get(lang, {
            "action": "query", "titles": title, "redirects": "1",
            "prop": "extracts|revisions|info|pageprops", "explaintext": "1", "exsectionformat": "wiki",
            "rvprop": "ids", "inprop": "url", "ppprop": "disambiguation",
        })
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            return None
        page = pages[0]
        if page.get("missing") or "disambiguation" in page.get("pageprops", {}):
            return None
        revid = page.get("revisions", [{}])[0].get("revid", 0)
        return Article(lang, page["pageid"], page["title"], revid, page.get("fullurl", ""),
                       split_sections(page.get("extract", "")))

    def langlinks(self, lang: str, titles: list[str], target: str) -> dict[str, str]:
        """Названия тех же статей на языке ``target``: {название на lang: название на target}."""
        result: dict[str, str] = {}
        for start in range(0, len(titles), 50):
            chunk = titles[start:start + 50]
            data = self.get(lang, {
                "action": "query", "titles": "|".join(chunk), "redirects": "1",
                "prop": "langlinks", "lllang": target, "lllimit": "max",
            })
            query = data.get("query", {})
            # Запрошенное название могло быть нормализовано или быть перенаправлением.
            renamed = {x["from"]: x["to"] for x in query.get("normalized", []) + query.get("redirects", [])}
            found = {p["title"]: p["langlinks"][0]["title"] for p in query.get("pages", []) if p.get("langlinks")}
            for title in chunk:
                resolved = title
                while resolved in renamed:
                    resolved = renamed[resolved]
                if resolved in found:
                    result[title] = found[resolved]
        return result

    def lead_links(self, lang: str, title: str, limit: int = 10) -> list[str]:
        """Статьи, на которые ссылается введение статьи, в порядке появления."""
        data = self.get(lang, {"action": "parse", "page": title, "redirects": "1", "prop": "links", "section": "0"})
        links = data.get("parse", {}).get("links", [])
        titles = [l["title"] for l in links if l.get("ns") == 0 and l.get("exists", True)]
        return titles[:limit]
