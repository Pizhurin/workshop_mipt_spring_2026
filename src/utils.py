# src/utils.py
"""
Общие утилиты для Entity Resolution пайплайна.
"""

import pandas as pd
from typing import Set, Any, Union


def frozenset_to_str(val: Any) -> str:
    """
    Надёжно преобразует frozenset (или другие типы) в строку.
    Используется в blocking и feature engineering.
    """
    if isinstance(val, frozenset):
        # Убираем None/NaN и сортируем для стабильности
        cleaned = sorted(str(x) for x in val if pd.notna(x) and x != '')
        return ", ".join(cleaned)

    if isinstance(val, str):
        return val.strip()

    if pd.notna(val):
        return str(val).strip()

    return ""


def safe_set(val: Any) -> Set:
    """
    Преобразует frozenset, строку или None в set.
    """
    if isinstance(val, frozenset):
        return {x for x in val if pd.notna(x) and x != ''}

    if isinstance(val, str):
        if not val.strip():
            return set()
        return {x.strip() for x in val.split(",") if x.strip()}

    if isinstance(val, (list, tuple, set)):
        return {x for x in val if pd.notna(x) and x != ''}

    return set()


def normalize_bool(val: Any) -> bool:
    """Надёжное преобразование различных представлений True/False."""
    if val is None or pd.isna(val):
        return False

    if isinstance(val, bool):
        return val

    return str(val).strip().lower() in {"true", "1", "yes", "y", "t", "да"}


def jaccard(set_a: Set, set_b: Set) -> float:
    """Коэффициент Жаккара для двух множеств."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def overlap_size(set_a: Set, set_b: Set) -> int:
    """Количество общих элементов."""
    return len(set_a & set_b)
