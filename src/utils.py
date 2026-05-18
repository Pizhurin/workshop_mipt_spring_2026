# src/utils.py
"""
Общие утилиты для Entity Resolution пайплайна.
"""

import math
import pandas as pd
from typing import Set, Any, Union


try:
    profile
except NameError:
    def profile(func):
        return func

@profile
def frozenset_to_str(val: Any) -> str:
    """
    Надёжно преобразует frozenset (или другие типы) в строку.
    Используется в blocking и feature engineering.
    """
    if isinstance(val, frozenset):
        # Убираем None/NaN и сортируем для стабильности
        cleaned = sorted(
            str(x) for x in val
            if x is not None
            and not (isinstance(x, float) and math.isnan(x))
            and x != ''
        )
        return ", ".join(cleaned)

    if isinstance(val, str):
        return val.strip()

    if val is not None and not (isinstance(val, float) and math.isnan(val)):
        return str(val).strip()

    return ""


@profile
def safe_set(val: Any) -> Set:
    """
    Преобразует frozenset, строку или None в set.
    """
    if isinstance(val, frozenset):
        return {
            x for x in val
            if x is not None
            and not (isinstance(x, float) and math.isnan(x))
            and x != ''
        }

    if isinstance(val, str):
        s = val.strip()
        if not s:
            return set()
        return {part for part in (x.strip() for x in s.split(",")) if part}

    if isinstance(val, (list, tuple, set)):
        return {
            x for x in val
            if x is not None
            and not (isinstance(x, float) and math.isnan(x))
            and x != ''
        }

    return set()


@profile
def normalize_bool(val: Any) -> bool:
    """Надёжное преобразование различных представлений True/False."""
    if val is None or pd.isna(val):
        return False

    if isinstance(val, bool):
        return val

    return str(val).strip().lower() in {"true", "1", "yes", "y", "t", "да"}


@profile
def jaccard(set_a: Set, set_b: Set) -> float:
    """
    Коэффициент Жаккара для двух множеств.
    |A ∪ B| = |A| + |B| – |A ∩ B|
    """
    if not set_a and not set_b:
        return 1.

    if not set_a or not set_b:
        return 0.

    intersection = set_a & set_b  # создаём пересечение только один раз
    return (
        len(intersection)
        / (len(set_a) + len(set_b) - len(intersection))
    )  # вычисление объединения без создания еще одно пересекающегося множества


@profile
def overlap_size(set_a: Set, set_b: Set) -> int:
    """Количество общих элементов."""
    return len(set_a & set_b)
