# Здесь объединяем сборщик PPTX и подготовку визуальных элементов; одинаковые для всех моделей и агентов.
from .builder import ElementReport, RenderReport, SlideStats, build_from_text, build_presentation
from .themes import THEME_NAMES, THEMES, Theme, pick_theme

__all__ = [
    "ElementReport", "RenderReport", "SlideStats", "THEME_NAMES", "THEMES", "Theme",
    "build_from_text", "build_presentation", "pick_theme",
]
