# site_activity_processor.py

import pandas as pd

import logging

logger = logging.getLogger(__name__)


def _safe_int(v):
    """Безопасное преобразование site_id в int."""
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def build_site_index(df_clean):
    """
    Строит индекс цифрового следа профилей.

    Вход: df_clean после full_preprocessing()
    Выход: dict[profile_id] -> dict[category] -> set[int]
    """
    logger.info("Сборка site_index...")

    site_prefixes = [
        "has_accept_", "has_account", "has_click_", "has_order_",
        "has_view_", "source_site_", "visited_",
    ]

    site_categories = [
        col for col in df_clean.columns
        if any(col.startswith(prefix) for prefix in site_prefixes)
    ]

    logger.info(f"  Найдено категорий: {len(site_categories)}")
    for cat in site_categories:
        logger.debug(f"    {cat}")

    site_index = {}

    # Инициализируем словарь для каждого profile_id
    for pid in df_clean["profile_id"].unique():
        site_index[pid] = {cat: set() for cat in site_categories}
        site_index[pid]["_all"] = set()

    # Заполняем по категориям
    for cat in site_categories:
        if cat not in df_clean.columns:
            logger.warning(f"  Колонка {cat} не найдена, пропускаем")
            continue

        grouped = df_clean.groupby("profile_id")[cat].apply(
            lambda x: {_safe_int(v) for v in x.dropna() if v and pd.notna(v)} - {None}
        )

        for pid, sites in grouped.items():
            site_index[pid][cat] = sites
            site_index[pid]["_all"].update(sites)

    logger.info(f"  Профилей в индексе: {len(site_index)}")
    logger.info(f"  Категорий: {len(site_categories)}")

    return site_index
