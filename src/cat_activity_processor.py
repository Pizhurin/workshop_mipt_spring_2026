# cat_activity_processor.py
"""
Собирает индекс по почтовой активности пользователя
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


    gb = df_clean.groupby("profile_id")
    # Только с _all, категории потом
    all_pids = df_clean["profile_id"].unique()
    cat_index = {pid: {"_all": set()} for pid in all_pids}

    # Заполняем по категориям
    for cat in cat_categories:
        grouped = gb[cat].apply(
            lambda x: {
                s.lower()
                for v in x
                if pd.notna(v) and v
                and (s := str(v).strip())
            }
        )

        for pid, values in grouped.items():
            if pid not in cat_index:  # защита: если pid не в all_pids
                cat_index[pid] = {"_all": set()}
            cat_index[pid][cat] = values
            cat_index[pid]["_all"].update(values)

    # Добавляем отсутствующие ключи категорий для всех профилей
    for pid in cat_index:
        for cat in cat_categories:
            if cat not in cat_index[pid]:
                cat_index[pid][cat] = set()

    logger.info(f"  Профилей в индексе: {len(cat_index)}")

    return cat_index
