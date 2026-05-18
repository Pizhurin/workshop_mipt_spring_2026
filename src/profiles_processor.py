# profiles_processor.py

"""
Единая точка входа для обработки профилей.
Объединяет preprocessing, агрегацию паспорта, сайтовый и категориальный индексы,
разбивку на train/val/test (только для разработки).
"""

import logging
from sklearn.model_selection import train_test_split

from src.config_loader import CONFIG

from .preprocessing import full_preprocessing
from .aggregation_profiles import build_profiles_df
from .site_activity_processor import build_site_index
from .cat_activity_processor import build_cat_index

logger = logging.getLogger(__name__)

try:
    profile
except NameError:
    def profile(func):
        return func


@profile
def add_train_val_test_split(profiles_df):
    """
    Добавляет колонку 'split' с разбиением 60/20/20 на уровне entity_id.
    Тип entity_type вычисляется автоматически (multi_profile, если >1 профиля).
    """
    # Вычисляем entity_type по количеству профилей в entity_id
    entity_counts = profiles_df.groupby("entity_id").size()
    entity_type_computed = entity_counts.apply(
        lambda x: "multi_profile" if x > 1 else "single_profile"
    )

    entity_info = entity_type_computed.reset_index()
    entity_info.columns = ["entity_id", "entity_type"]

    # Преобразуем в списки
    entity_ids = entity_info["entity_id"].astype(str).tolist()
    entity_types = entity_info["entity_type"].astype(str).tolist()

    # Отделяем test (20%)
    train_val_ids, test_ids, train_val_types, test_types = train_test_split(
        entity_ids,
        entity_types,
        test_size=0.2,
        stratify=entity_types,
        random_state=CONFIG['seed'],
    )

    # Отделяем val (20% от исходного = 25% от train_val)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=0.25,
        stratify=train_val_types,
        random_state=CONFIG['seed'],
    )

    # Маппинг entity_id -> split
    entity_to_split = {}
    for eid in train_ids:
        entity_to_split[eid] = "train"
    for eid in val_ids:
        entity_to_split[eid] = "val"
    for eid in test_ids:
        entity_to_split[eid] = "test"

    profiles_df["split"] = profiles_df["entity_id"].map(entity_to_split)

    logger.info("Разбиение добавлено:")
    for split_name in ["train", "val", "test"]:
        n = profiles_df["split"].eq(split_name).sum()
        n_mp = profiles_df[
            (profiles_df["split"] == split_name)
            & (profiles_df["entity_type"] == "multi_profile")
        ].shape[0]
        logger.info(f"  {split_name}: {n:,} профилей (multi: {n_mp:,})")

    return profiles_df

@profile
def add_temporal_features(profiles_df, df_clean):
    """Добавляет временные признаки в profiles_df из сырых событий."""
    # first_event_time, last_event_time
    gb = df_clean.groupby("profile_id")

    # Одна агрегация вместо раздельных вызовов agg и apply
    time_agg = gb.agg(
        first_event_time=("created_at", "min"),
        last_event_time=("created_at", "max"),
        active_days=("day", lambda x: set(x[x.notna()])),  # dropna() оч тяжелая
        active_hours=("local_hour", lambda x: set(x[x.notna()]))
    )

    # Присоединяем к profiles_df
    profiles_df = profiles_df.join(time_agg)

    # Для features.py нужна last_event_date
    profiles_df["last_event_date"] = profiles_df["last_event_time"]

    return profiles_df

@profile
def process_profiles(df_raw, split_data=False):
    """
    Полный цикл обработки.

    Вход:
        df_raw — сырой датафрейм
        split_data — если True, добавляет колонку 'split' (train/val/test)
                     если False, работает без разбиения (для production)

    Выход:
        profiles_df, site_index, cat_index
    """
    logger.info("=" * 50)
    logger.info("Запуск process_profiles()")
    logger.info("=" * 50)

    # Определяем, есть ли ground truth
    has_entity = 'entity_id' in df_raw.columns
    if not has_entity:
        # Добавляем фиктивные колонки для прохождения preprocessing
        df_raw = df_raw.assign(  # единый assign вместо .copy() + 2 присваивания
            entity_id=df_raw['profile_id'],
            entity_type='single_profile'
        )
        split_data = False  # отключаем разбиение, так как нет истинных сущностей
        logger.info("Входные данные не содержат entity_id. Добавлены фиктивные колонки для обработки.")

    # Этап 1: Предобработка
    logger.info("Этап 1/5: Предобработка...")
    df_clean = full_preprocessing(df_raw)

    # Этап 2: Паспорт профиля
    logger.info("Этап 2/5: Построение паспорта профилей...")
    profiles_df = build_profiles_df(df_clean)

    # Этап 3: Индекс сайтов (цифровой след)
    logger.info("Этап 3/5: Построение индекса сайтов...")
    site_index = build_site_index(df_clean)

    # Этап 4: Индекс почтовой активности
    logger.info("Этап 4/5: Построение индекса почтовой активности...")
    cat_index = build_cat_index(df_clean)

    # Этап 5: Добавление временных признаков
    logger.info("Этап 5/5 Добавление временных признаков...")
    profiles_df = add_temporal_features(profiles_df, df_clean)

    # Опционально: добавляем разбиение
    if split_data:
        logger.info("Добавление разбиения train/val/test...")
        profiles_df = add_train_val_test_split(profiles_df)

    logger.info("=" * 50)
    logger.info(
        f"Готово: {len(profiles_df):,} профилей, "
        f"{len(site_index):,} сайтовых индексов, "
        f"{len(cat_index):,} категориальных индексов"
    )
    logger.info("=" * 50)

    # После построения profiles_df удаляем фиктивные колонки, если их не было
    if not has_entity:
        profiles_df = profiles_df.drop(columns=['entity_id', 'entity_type'], errors='ignore')
        logger.info("Фиктивные колонки entity_id/entity_type удалены из результата.")

    return profiles_df, site_index, cat_index
