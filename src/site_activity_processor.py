# site_activity_processor.py
"""
Собирает индекс по цифровому следу пользователя
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

try:
    profile
except NameError:
    def profile(func):
        return func


@profile
def _safe_int(v):
    """Безопасное преобразование site_id в int."""
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


@profile
def build_site_index(df_clean):
    """
    Строит индекс цифрового следа профилей.

    Вход: df_clean после full_preprocessing()
    Выход: dict[profile_id] -> dict[category] -> set[int]
    """
    logger.info("Сборка site_index...")

    site_prefixes = ["has_accept_", "has_account", "has_click_", "has_order_",
                     "has_view_", "source_site_", "visited_",]

    site_categories = [
        col for col in df_clean.columns
        if any(col.startswith(prefix) for prefix in site_prefixes)
    ]

    logger.info(f"  Найдено категорий: {len(site_categories)}")
    for cat in site_categories:
        logger.debug(f"    {cat}")

    # Снова единый объект GroupBy для всех категорий
    gb = df_clean.groupby("profile_id")

    # site_index только для профилей, у которых есть хоть какие-то данные
    all_pids = df_clean["profile_id"].unique()

    # Словарь надо наполнять по мере обработки категорий
    # Ключи категорий добавляем позже, чтобы не создавать пустые словари заранее
    site_index = {pid: {"_all": set()} for pid in all_pids}

    # Заполняем по категориям
    for cat in site_categories:
        if cat not in df_clean.columns:
            logger.warning(f"  Колонка {cat} не найдена, пропускаем")
            continue
        # Векторизованная обработка группы
        # Быстрая лямбда без dropna и pd.notna
        grouped = gb[cat].apply(  # Вот и общий GroupBy
            lambda x: {
                val for v in x
                if pd.notna(v) and v
                and (val := _safe_int(v)) is not None  # := (walrus operator)
            }
        )

        for pid, sites in grouped.items():
            if pid not in site_index:
                site_index[pid] = {"_all": set()}
            site_index[pid][cat] = sites  # множество без None (фильтр _safe_int)
            site_index[pid]["_all"].update(sites)

    # Добавляем отсутствующие ключи категорий для всех профилей (пустые множества)
    for pid in site_index:
        for cat in site_categories:
            if cat not in site_index[pid]:
                site_index[pid][cat] = set()

    logger.info(f"  Профилей в индексе: {len(site_index)}")
    logger.info(f"  Категорий: {len(site_categories)}")

    return site_index
