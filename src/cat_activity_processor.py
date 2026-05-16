# cat_activity_processor.py
"""
Собирает индекс по почтовой активности пользователя
"""
import pandas as pd

import logging

logger = logging.getLogger(__name__)


def build_cat_index(df_clean):
    """
    Строит индекс почтовой активности профилей.

    Категории определяются динамически — все колонки, 
    соответствующие паттернам почтовых признаков.

    Вход: df_clean после full_preprocessing()
    Выход: dict[profile_id] -> dict[category] -> set[str]
    """
    logger.info("Сборка cat_index...")

    # Динамически определяем категории почтовой активности
    cat_prefixes = [
        "postman_response_", "postman_action_",
        "postman_campaign_", "postman_scenario_",
        "mm_event_",
    ]

    cat_categories = [
        col for col in df_clean.columns
        if any(col.startswith(prefix) for prefix in cat_prefixes)
    ]

    logger.info(f"  Найдено категорий: {len(cat_categories)}")
    for cat in cat_categories:
        logger.debug(f"    {cat}")

    cat_index = {}

    # Инициализируем словарь для каждого profile_id
    for pid in df_clean["profile_id"].unique():
        cat_index[pid] = {cat: set() for cat in cat_categories}
        cat_index[pid]["_all"] = set()

    # Заполняем по категориям
    for cat in cat_categories:
        grouped = df_clean.groupby("profile_id")[cat].apply(
            lambda x: {str(v).strip().lower() for v in x.dropna() 
                      if v and pd.notna(v) and str(v).strip()}
        )

        for pid, values in grouped.items():
            cat_index[pid][cat] = values
            cat_index[pid]["_all"].update(values)

    logger.info(f"  Профилей в индексе: {len(cat_index)}")

    return cat_index
