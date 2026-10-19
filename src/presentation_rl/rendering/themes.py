# Здесь описываем цветовые палитры презентаций и выбор палитры для запроса.
"""Палитры оформления.

Палитры различаются только цветами: шрифты, размеры и раскладка у всех общие,
поэтому палитра не влияет на метрики заполнения и переполнения. Модель палитру
не выбирает и не видит. Пользователь может указать её в запросе; если не указал
(``auto``), палитра выбирается по тексту запроса: один и тот же запрос всегда
получает одну и ту же палитру, разные запросы — разные.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Цвета и шрифты одной палитры (цвета — hex без #)."""

    ink: str = "1F2A44"  # заголовки, основной тёмный
    body: str = "2E3A4F"  # основной текст
    muted: str = "636C7E"  # подписи, источники, номера слайдов
    accent: str = "2A9D8F"  # маркеры, линии, первая серия
    accent_soft: str = "E3F2EF"  # заливка узлов схем и цитат
    grid: str = "E6E9EE"
    connector: str = "5B6475"
    title_bg: str = "1F2A44"  # фон титульного слайда
    title_text: str = "FFFFFF"
    title_body: str = "D5DEEA"
    placeholder_fill: str = "EEF1F4"
    placeholder_line: str = "B8BFC9"
    palette: tuple[str, ...] = ("2A9D8F", "E76F51", "E9C46A", "264653", "F4A261", "8AB17D", "6C8EBF", "B5838D")
    heading_font: str = "Cambria"
    body_font: str = "Calibri"


THEMES: dict[str, Theme] = {
    "navy": Theme(),
    "forest": Theme(
        ink="1E3B2F", body="2F3E36", muted="5F6E66", accent="2F7D4A", accent_soft="E4F1E8",
        connector="56655D", title_bg="1E3B2F", title_body="D3E6D8",
        palette=("2F7D4A", "D08C60", "4F6D7A", "C9A227", "8AA29E", "7A5C61", "5B8E7D", "B5C99A"),
    ),
    "terracotta": Theme(
        ink="3D2B26", body="4A3A35", muted="76655F", accent="B4532F", accent_soft="F8E9E3",
        connector="6E5A53", title_bg="5A2E22", title_body="F0DCD3",
        palette=("B4532F", "3D7A8C", "E2A03F", "6B8F71", "8C5E8A", "D9826B", "4E5D6C", "B9A44C"),
    ),
    "plum": Theme(
        ink="2E2140", body="3B3247", muted="6C6478", accent="74489A", accent_soft="EFE8F6",
        connector="625A6E", title_bg="2E2140", title_body="DDD3EA",
        palette=("74489A", "E07A5F", "3D9970", "F2C14E", "5C7AEA", "B5838D", "4A5568", "9BC1BC"),
    ),
    "ocean": Theme(
        ink="0F3352", body="26394D", muted="5D6B7A", accent="1F6FA8", accent_soft="E3EFF8",
        connector="52606E", title_bg="0F3352", title_body="CFE0EE",
        palette=("1F6FA8", "F28C28", "2E9E6A", "D64545", "7D5BA6", "E0B33A", "5A6B7B", "69B3C9"),
    ),
    "graphite": Theme(
        ink="222428", body="33363B", muted="65696F", accent="C23B4E", accent_soft="F8E6E9",
        connector="5A5E64", title_bg="2B2D31", title_body="DADCE0",
        palette=("C23B4E", "2B7A78", "EDAE49", "3D5A80", "8D6A9F", "66A182", "5F6B73", "E07A5F"),
    ),
}
DEFAULT_THEME = "navy"
THEME_NAMES = tuple(THEMES)


def pick_theme(name: str = "auto", key: str = "") -> Theme:
    """Палитра по имени; ``auto`` — по тексту запроса ``key`` (стабильно между запусками)."""
    if name == "auto":
        names = sorted(THEMES)
        return THEMES[names[zlib.crc32(key.encode("utf-8")) % len(names)]]
    if name not in THEMES:
        raise ValueError(f"неизвестная палитра: {name}; есть: {', '.join(THEME_NAMES)}")
    return THEMES[name]
