# profiles_processor.py

import pandas as pd

import logging

logger = logging.getLogger(__name__)


# ─── Вспомогательные функции ───────────────────────────────────────────────────

def last_non_null(series):
    """Возвращает последнее непустое значение в серии (с учётом порядка событий)."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    return non_null.iloc[-1]


# ─── Паспорт профиля ────────────────────────────────────────────────────────────

def build_profiles_df(df_clean):
    """
    Собирает плоскую таблицу профилей.

    Вход: df_clean после full_preprocessing()
    Выход: profiles_df с индексом profile_id
    """
    logger.info("Сборка profiles_df...")

    # 1. Сортировка по времени
    df_sorted = df_clean.sort_values("created_at")

    # 2. Словарь агрегации
    agg_dict = {
        # Группа А: "last" векторизованная операция для проверенных (потери = 0)
        "first_name_clean": "last",
        "last_name_clean": "last",
        "sex": "last",
        "geoname_id": "last",
        "geoname": "last",
        "country": "last",
        "is_not_russia": "last",
        "is_million": "last",
        "population": "last",
        "subdivision_1_iso_code": "last",
        "tz_offset": "last",

        # Группа А: last_non_null только для редких
        "birthday": last_non_null,
        "local_hour": last_non_null,
        "day": last_non_null,

        # Группа Б: (frozenset)
        "email_normalized": lambda x: frozenset(x.dropna()),
        "email_domain": lambda x: frozenset(x.dropna()),
        "phone_normalized": lambda x: frozenset(x.dropna()),
        "phone_prefix": lambda x: frozenset(x.dropna()),
        "device": lambda x: frozenset(x.dropna()),
        "osfamily": lambda x: frozenset(x.dropna()),
        "browser": lambda x: frozenset(x.dropna()),

        # Группа В:
        "is_gmail": "max",
        "is_yandex": "max",
        "is_man": "max",
        "is_woman": "max",
        "is_phone": "max",
        "was_phone_lead": "max",

        # Группа Г: 
        "entity_id": "first",
        "entity_type": "first"
    }

    # 3. Группировка
    profiles_df = df_sorted.groupby("profile_id").agg(agg_dict)

    # 4. Счётчики
    profiles_df["n_events"] = df_sorted.groupby("profile_id").size()
    profiles_df["n_emails"] = profiles_df["email_normalized"].apply(len)
    profiles_df["n_phones"] = profiles_df["phone_normalized"].apply(len)
    profiles_df["n_devices"] = profiles_df["device"].apply(len)

    # 5. Приведение типов (булевы флаги из JSON -> bool)
    bool_true = {True, "True", "true", "1", 1}
    profiles_df["is_not_russia"] = profiles_df["is_not_russia"].apply(
        lambda x: x in bool_true
    )
    profiles_df["is_million"] = profiles_df["is_million"].apply(
        lambda x: x in bool_true
    )

    logger.info(f"  Профилей: {len(profiles_df)}")
    logger.info(f"  Колонок: {len(profiles_df.columns)}")

    return profiles_df
