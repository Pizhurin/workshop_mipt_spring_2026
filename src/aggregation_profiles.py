# aggregation_profiles.py
"""
Собирает паспорт профилей
"""
import pandas as pd
import logging
from .utils import frozenset_to_str, normalize_bool

logger = logging.getLogger(__name__)


def last_non_null(series):
    non_null = series.dropna()
    return non_null.iloc[-1] if len(non_null) > 0 else None


def build_profiles_df(df_clean):
    logger.info("Сборка profiles_df...")

    df_sorted = df_clean.sort_values("created_at")

    agg_dict = {
        # Последнее известное значение
        "first_name_clean": "last",
        "last_name_clean": "last",
        "sex": "last",
        "geoname_id": "last",
        "geoname": "last",
        "country": "last",
        "subdivision_1_iso_code": "last",
        "tz_offset": "last",

        # Редкие поля: последнее заполненное
        "birthday_clean": last_non_null,

        # Множества
        "email_normalized": lambda x: frozenset(x.dropna()),
        "email_domain": lambda x: frozenset(x.dropna()),
        "phone_normalized": lambda x: frozenset(x.dropna()),
        "phone_prefix": lambda x: frozenset(x.dropna()),
        "device": lambda x: frozenset(x.dropna()),
        "osfamily": lambda x: frozenset(x.dropna()),
        "browser": lambda x: frozenset(x.dropna()),

        # Флаги
        "is_gmail": "max",
        "is_yandex": "max",
        "is_man": "max",
        "is_woman": "max",
        "is_phone": "max",
        "was_phone_lead": "max",
        "created_at": "max",

        "entity_id": "first",
        "entity_type": "first",
    }

    profiles_df = df_sorted.groupby("profile_id").agg(agg_dict)
    profiles_df.rename(columns={"created_at": "last_event_date"}, inplace=True)

    # Счётчики
    profiles_df["n_events"] = df_sorted.groupby("profile_id").size()
    profiles_df["n_emails"] = profiles_df["email_normalized"].apply(len)
    profiles_df["n_phones"] = profiles_df["phone_normalized"].apply(len)
    profiles_df["n_devices"] = profiles_df["device"].apply(len)

    # Конвертация в булевые значения
    for col in ["is_not_russia", "is_million"]:
        if col in profiles_df.columns:
            profiles_df[col] = profiles_df[col].apply(normalize_bool)

    logger.info(f"  Профилей: {len(profiles_df):,}")
    return profiles_df
