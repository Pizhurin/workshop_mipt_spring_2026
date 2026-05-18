# aggregation_profiles.py
"""
Собирает паспорт профилей
"""
import pandas as pd
import logging
from .utils import frozenset_to_str, normalize_bool

logger = logging.getLogger(__name__)

try:
    profile
except NameError:
    def profile(func):
        return func

@profile
def last_non_null(series):
    # .dropna() создаёт "тяжелую" копию. МОжно воспользоваться маской. Быстрее
    mask = series.notna().to_numpy()  # булев массив, без копирования данных
    if not mask.any():
        return None
    idx = mask.nonzero()[0][-1]
    return series.iat[idx]

@profile
def build_profiles_df(df_clean):
    logger.info("Сборка profiles_df...")

    df_sorted = df_clean.sort_values("created_at")
    gb = df_sorted.groupby("profile_id")  # один GroupBy, чтобы не пересоздавать

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

        # Множества. .dropna() всегда МЕГАТЯЖЕЛЫЙ метод. Лучше numpy маска
        # без лишнего копирования
        "email_normalized": lambda x: frozenset(x[x.notna()]),
        "email_domain": lambda x: frozenset(x[x.notna()]),
        "phone_normalized": lambda x: frozenset(x[x.notna()]),
        "phone_prefix": lambda x: frozenset(x[x.notna()]),
        "device": lambda x: frozenset(x[x.notna()]),
        "osfamily": lambda x: frozenset(x[x.notna()]),
        "browser": lambda x: frozenset(x[x.notna()]),

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

    profiles_df = gb.agg(agg_dict)  # тот же GroupBy
    profiles_df.rename(columns={"created_at": "last_event_date"}, inplace=True)

    # Счётчики
    profiles_df["n_events"] = gb.size()  # И хдесь тот же GroupBy
    profiles_df["n_emails"] = profiles_df["email_normalized"].map(len)  # map быстрее, если нужно обработать всё и сразу
    profiles_df["n_phones"] = profiles_df["phone_normalized"].map(len)
    profiles_df["n_devices"] = profiles_df["device"].map(len)

    # Конвертация в булевые значения
    for col in ["is_not_russia", "is_million"]:
        if col in profiles_df.columns:
            profiles_df[col] = profiles_df[col].apply(normalize_bool)

    logger.info(f"  Профилей: {len(profiles_df):,}")
    return profiles_df
