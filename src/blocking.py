# src/blocking.py
"""
Многопроходный блокинг.
"""

import logging
import itertools
from collections import defaultdict
from typing import List, Tuple

import pandas as pd

from src.config_loader import CONFIG
from .utils import frozenset_to_str

logger = logging.getLogger(__name__)


def make_pairs(indices: List, max_group_size: int = CONFIG['blocking']['max_group_size']) -> List[Tuple]:
    """Генерирует пары с защитой от слишком больших групп."""
    if len(indices) < 2:
        return []
    if len(indices) > max_group_size:
        logger.warning(f"Группа слишком большая ({len(indices):,}) -> ограничиваем до {max_group_size}")
        indices = indices[:max_group_size]
    return list(itertools.combinations(indices, 2))


def blocking_pipeline(profiles_df: pd.DataFrame, 
                     site_index: dict, 
                     max_group_size: int = CONFIG['blocking']['max_group_size']) -> List[Tuple]:
    """Основной пайплайн блокинга."""
    logger.info("=" * 70)
    logger.info("Запуск многопроходного блокинга")
    logger.info("=" * 70)

    all_pairs = []

    passes = [
        ("Sites", lambda: blocking_by_sites(profiles_df, site_index, max_group_size)),
        ("Geo + Device", lambda: blocking_by_geo_device(profiles_df, max_group_size)),
        ("Name + Geo", lambda: blocking_by_name_geo(profiles_df, max_group_size)),
        ("Sex + Geo", lambda: blocking_by_sex_geo(profiles_df, max_group_size)),
        ("TZ + Geo", lambda: blocking_by_tz_geo(profiles_df, max_group_size)),
        ("Device + TZ + Geo", lambda: blocking_by_device_tz_geo(profiles_df, max_group_size)),
    ]

    for name, func in passes:
        logger.info(f"Pass: {name}...")
        pairs = func()
        logger.info(f"  -> {len(pairs):,} пар")
        all_pairs.extend(pairs)

    candidate_pairs = list(set(tuple(sorted(p)) for p in all_pairs))

    logger.info("=" * 70)
    logger.info(f"Блокинг завершен: {len(candidate_pairs):,} уникальных пар-кандидатов")
    logger.info("=" * 70)

    return candidate_pairs


# Блоки

def blocking_by_sites(profiles_df, site_index, max_group_size=CONFIG['blocking']['max_group_size']):
    site_to_profiles = defaultdict(list)
    for pid, categories in site_index.items():
        for site_id in categories.get("_all", set()):
            site_to_profiles[site_id].append(pid)

    pairs = []
    for plist in site_to_profiles.values():
        if len(plist) >= 2:
            pairs.extend(make_pairs(plist, max_group_size))
    return list(set(tuple(sorted(p)) for p in pairs))


def blocking_by_geo_device(profiles_df, max_group_size=CONFIG['blocking']['max_group_size']):
    mask = (
        profiles_df["geoname_id"].notna() &
        profiles_df["osfamily"].notna() &
        profiles_df["device"].notna()
    )
    df = profiles_df[mask].copy()

    df["_key"] = (
        df["geoname_id"].astype(str) + "|" +
        df["osfamily"].apply(frozenset_to_str) + "|" +
        df["device"].apply(frozenset_to_str)
    )

    grouped = df.groupby("_key").apply(lambda x: list(x.index))

    pairs = []
    for plist in grouped:
        if len(plist) >= 2:
            pairs.extend(make_pairs(plist, max_group_size))

    logger.info(f"  Ключей: {len(grouped):,}, групп ≥2: {sum(1 for x in grouped if len(x) >= 2):,}")
    return list(set(tuple(sorted(p)) for p in pairs))


def blocking_by_name_geo(profiles_df, max_group_size=CONFIG['blocking']['max_group_size']):
    mask = profiles_df["first_name_clean"].notna() & profiles_df["geoname_id"].notna()
    df = profiles_df[mask].copy()

    df["_key"] = df["first_name_clean"].astype(str) + "|" + df["geoname_id"].astype(str)

    has_last = df["last_name_clean"].notna()
    df.loc[has_last, "_key"] = (
        df.loc[has_last, "first_name_clean"].astype(str) + "|" +
        df.loc[has_last, "last_name_clean"].astype(str) + "|" +
        df.loc[has_last, "geoname_id"].astype(str)
    )

    grouped = df.groupby("_key").apply(lambda x: list(x.index))

    pairs = []
    for plist in grouped:
        if len(plist) >= 2:
            pairs.extend(make_pairs(plist, max_group_size))

    logger.info(f"  Ключей: {len(grouped):,}, групп ≥2: {sum(1 for x in grouped if len(x) >= 2):,}")
    return list(set(tuple(sorted(p)) for p in pairs))


def blocking_by_sex_geo(profiles_df, max_group_size=CONFIG['blocking']['max_group_size']):
    key_to_profiles = defaultdict(list)
    for pid, row in profiles_df.iterrows():
        sex = row.get('sex')
        geo = row.get('geoname_id')
        if pd.isna(sex) or pd.isna(geo) or sex == 'unknown':
            continue
        key_to_profiles[f"{sex}|{geo}"].append(pid)

    pairs = []
    for plist in key_to_profiles.values():
        if len(plist) >= 2:
            pairs.extend(make_pairs(plist, max_group_size))

    logger.info(f"  Ключей: {len(key_to_profiles):,}, групп ≥2: {sum(1 for v in key_to_profiles.values() if len(v) >= 2):,}")
    return list(set(tuple(sorted(p)) for p in pairs))


def blocking_by_tz_geo(profiles_df, max_group_size=CONFIG['blocking']['max_group_size']):
    key_to_profiles = defaultdict(list)
    for pid, row in profiles_df.iterrows():
        tz = row.get('tz_offset')
        geo = row.get('geoname_id')
        if pd.isna(tz) or pd.isna(geo):
            continue
        key_to_profiles[f"{tz}|{geo}"].append(pid)

    pairs = []
    for plist in key_to_profiles.values():
        if len(plist) >= 2:
            pairs.extend(make_pairs(plist, max_group_size))

    logger.info(f"  Ключей: {len(key_to_profiles):,}, групп ≥2: {sum(1 for v in key_to_profiles.values() if len(v) >= 2):,}")
    return list(set(tuple(sorted(p)) for p in pairs))


def blocking_by_device_tz_geo(profiles_df, max_group_size=CONFIG['blocking']['max_group_size']):
    key_to_profiles = defaultdict(list)
    for pid, row in profiles_df.iterrows():
        device = row.get('device')
        tz = row.get('tz_offset')
        geo = row.get('geoname_id')
        if pd.isna(device) or pd.isna(tz) or pd.isna(geo):
            continue
        key = f"{frozenset_to_str(device)}|{tz}|{geo}"
        key_to_profiles[key].append(pid)

    pairs = []
    for plist in key_to_profiles.values():
        if len(plist) >= 2:
            pairs.extend(make_pairs(plist, max_group_size))

    logger.info(f"  Ключей: {len(key_to_profiles):,}, групп ≥2: {sum(1 for v in key_to_profiles.values() if len(v) >= 2):,}")
    return list(set(tuple(sorted(p)) for p in pairs))
