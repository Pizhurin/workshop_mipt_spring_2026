# profiles_processor.py

"""
Единая точка входа для обработки профилей.
Объединяет preprocessing, агрегацию паспорта, сайтовый и категориальный индексы.
"""

import logging
from .preprocessing import full_preprocessing
from .aggregation_profiles import build_profiles_df
from .site_activity_processor import build_site_index
from .cat_activity_processor import build_cat_index

logger = logging.getLogger(__name__)


def process_profiles(df_raw):
    """
    Полный цикл обработки: сырые события -> профили -> индексы.

    Вход: df_raw — сырой датафрейм с 13 колонками
    Выход: (profiles_df, site_index, cat_index)
        - profiles_df: DataFrame с индексом profile_id (33 колонки)
        - site_index: dict[profile_id] -> dict[category] -> set[site_id]
        - cat_index: dict[profile_id] -> dict[category] -> set[str]
    """
    logger.info("=" * 50)
    logger.info("ЗАПУСК process_profiles()")
    logger.info("=" * 50)

    # Этап 1: Предобработка
    logger.info("Этап 1/4: Предобработка...")
    df_clean = full_preprocessing(df_raw)

    # Этап 2: Паспорт профиля
    logger.info("Этап 2/4: Построение паспорта профилей...")
    profiles_df = build_profiles_df(df_clean)

    # Этап 3: Индекс сайтов (цифровой след)
    logger.info("Этап 3/4: Построение индекса сайтов...")
    site_index = build_site_index(df_clean)

    # Этап 4: Индекс почтовой активности
    logger.info("Этап 4/4: Построение индекса почтовой активности...")
    cat_index = build_cat_index(df_clean)

    logger.info("=" * 50)
    logger.info(
        f"ГОТОВО: {len(profiles_df):,} профилей, "
        f"{len(site_index):,} сайтовых индексов, "
        f"{len(cat_index):,} категориальных индексов"
    )
    logger.info("=" * 50)

    return profiles_df, site_index, cat_index
