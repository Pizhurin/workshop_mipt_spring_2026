# src/features.py
"""
Feature Engineering
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from Levenshtein import distance as lev_distance


from .utils import (
    frozenset_to_str,
    safe_set,
    jaccard,
    overlap_size
)

logger = logging.getLogger(__name__)

try:
    profile
except NameError:
    def profile(func):
        return func

# IDF
@profile
def compute_idf_sites(site_index: Dict, num_profiles: int) -> Dict:
    logger.info("Вычисление IDF для site_id...")
    doc_count = defaultdict(int)
    for categories in site_index.values():
        for site_id in categories.get("_all", set()):
            if pd.notna(site_id):
                doc_count[site_id] += 1
    idf = {k: np.log(num_profiles / v) if v > 0 else 0.0 for k, v in doc_count.items()}
    logger.info(f"IDF site_id: {len(idf):,}")
    return idf


@profile
def compute_idf_email_domains(profiles_df: pd.DataFrame) -> Dict:
    logger.info("Вычисление IDF для email доменов...")
    num_profiles = len(profiles_df)
    doc_count = defaultdict(int)
    for domains in profiles_df["email_domain"]:
        if isinstance(domains, frozenset):
            for d in domains:
                if d and pd.notna(d):
                    doc_count[d] += 1
    idf = {k: np.log(num_profiles / v) if v > 0 else 0.0 for k, v in doc_count.items()}
    logger.info(f"IDF domains: {len(idf):,}")
    return idf


# Вспомогательные функции
@profile
def _normalized_levenshtein(str_a: str, str_b: str) -> float:
    if not str_a and not str_b: return 0.0
    if not str_a or not str_b: return 1.0
    return lev_distance(str_a, str_b) / max(len(str_a), len(str_b))

@profile
def _char_jaccard(str_a: str, str_b: str, ngram: int = 2) -> float:
    if not str_a and not str_b: return 1.0
    if not str_a or not str_b: return 0.0
    def ngrams(s, n):
        return {s[i:i+n] for i in range(len(s) - n + 1)}
    return jaccard(ngrams(str_a.lower(), ngram), ngrams(str_b.lower(), ngram))

@profile
def time_overlap_ratio(set1, set2):
    if not set1 or not set2:
        return 0.
    overlap = overlap_size(set1, set2)  # вычислять надо 1 раз,
    # а не 2 раза одно и то же в выражении ниже
    return overlap / (len(set1) + len(set2) - overlap)

@profile
def add_conflict_features(features, rows1, rows2):
    features['country_conflict'] = (
          (rows1['country'].fillna('') != '')
        & (rows2['country'].fillna('') != '')
        & (rows1['country'] != rows2['country'])
    ).astype(int)

    features['tz_far_apart'] = (
        np.abs(rows1['tz_offset'].fillna(0) - rows2['tz_offset'].fillna(0)) > 4
    ).astype(int)

    # Векторизованная проверка непустоты device / замена apply(frozenset_to_str)
    dev1_nonempty, dev2_nonempty = (
        rows1['device'].astype(bool),
        rows2['device'].astype(bool)
    )

    features['device_os_conflict'] = (
          dev1_nonempty
        & dev2_nonempty
        & (rows1['osfamily'] != rows2['osfamily'])  # прямое сравнение
    ).astype(int)
    return features


# Главная функция
@profile
def build_features(
    pairs: List[Tuple],
    profiles_df: pd.DataFrame,
    site_index: Dict,
    cat_index: Dict,
    idf_sites: Dict,
    idf_domains: Dict,
    batch_size: int = 400_000,
    training: bool = True,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:

    logger.info(f"Генерация признаков для {len(pairs):,} пар...")
    if training and 'entity_id' in profiles_df.columns:
        entity_id_map = profiles_df["entity_id"].to_dict()
    else:
        entity_id_map = None

    phone_prefix_set = profiles_df['phone_prefix'].map(safe_set)
    device_set = profiles_df['device'].map(safe_set)
    osfamily_set = profiles_df['osfamily'].map(safe_set)
    if 'active_days' in profiles_df.columns:
        active_days_set = profiles_df['active_days'].map(safe_set)
    else:
        active_days_set = None

    X_batches, y_batches = [], []

    for start in range(0, len(pairs), batch_size):
        end = min(start + batch_size, len(pairs))
        batch_pairs = pairs[start:end]

        idx1 = [p[0] for p in batch_pairs]
        idx2 = [p[1] for p in batch_pairs]

        rows1 = profiles_df.loc[idx1].reset_index(drop=True)
        rows2 = profiles_df.loc[idx2].reset_index(drop=True)

        features = pd.DataFrame(index=range(len(batch_pairs)))

        # Имена
        fname1 = rows1["first_name_clean"].fillna("").astype(str)
        fname2 = rows2["first_name_clean"].fillna("").astype(str)
        lname1 = rows1["last_name_clean"].fillna("").astype(str)
        lname2 = rows2["last_name_clean"].fillna("").astype(str)

        features["fname_lev"] = [
            _normalized_levenshtein(a, b) for a, b in zip(fname1, fname2)
        ]
        features["fname_jaccard"] = [
            _char_jaccard(a, b) for a, b in zip(fname1, fname2)
        ]
        features["full_name_jaccard"] = [
            _char_jaccard(a + " " + b, c + " " + d)
            for a, b, c, d in zip(fname1, lname1, fname2, lname2)
        ]

        # Phone
        pref1 = phone_prefix_set[idx1].tolist()
        pref2 = phone_prefix_set[idx2].tolist()
        features["phone_prefix_jaccard"] = [
            jaccard(a, b) for a, b in zip(pref1, pref2)
        ]
        features["phone_prefix_overlap"] = [
            overlap_size(a, b) for a, b in zip(pref1, pref2)
        ]

        # Geo & TZ
        features["geo_match"] = (
            rows1["geoname_id"] == rows2["geoname_id"]
        ).astype(int)
        features["tz_diff"] = np.abs(
            rows1["tz_offset"].fillna(0) - rows2["tz_offset"].fillna(0)
        )

        # Устройства (готовые множества)
        dev1 = device_set[idx1].tolist()
        dev2 = device_set[idx2].tolist()
        os1  = osfamily_set[idx1].tolist()
        os2  = osfamily_set[idx2].tolist()

        features["device_jaccard"] = [jaccard(a, b) for a, b in zip(dev1, dev2)]
        features["osfamily_jaccard"] = [jaccard(a, b) for a, b in zip(os1, os2)]

        # Сайты
        site_all1 = [site_index.get(pid, {}).get("_all", set()) for pid in idx1]
        site_all2 = [site_index.get(pid, {}).get("_all", set()) for pid in idx2]
        features["sites_jaccard"] = [
            jaccard(a, b) for a, b in zip(site_all1, site_all2)
        ]
        features["sites_overlap"] = [
            overlap_size(a, b) for a, b in zip(site_all1, site_all2)
        ]

        site_cats = [
            "has_order_365", "has_account", "has_accept_365",
            "visited_365", "has_click_365"
        ]
        for cat in site_cats:
            s1 = [site_index.get(pid, {}).get(cat, set()) for pid in idx1]
            s2 = [site_index.get(pid, {}).get(cat, set()) for pid in idx2]
            features[f"{cat}_jaccard"] = [jaccard(a, b) for a, b in zip(s1, s2)]
            features[f"{cat}_overlap"] = [
                overlap_size(a, b) for a, b in zip(s1, s2)
            ]

        # Временные
        features["last_event_diff_days"] = np.abs(
            pd.to_datetime(rows1["last_event_date"]) -
            pd.to_datetime(rows2["last_event_date"])
        ).dt.total_seconds() / 86400

        if active_days_set is not None:
            days1 = active_days_set[idx1].tolist()
            days2 = active_days_set[idx2].tolist()
        else:
            days1 = [set() for _ in idx1]
            days2 = [set() for _ in idx2]
        features["day_overlap_ratio"] = [
            time_overlap_ratio(a, b) for a, b in zip(days1, days2)
        ]

        # Флаги и конфликты
        for flag in ["is_gmail", "is_yandex", "is_man", "is_woman", "is_phone"]:
            features[f"both_{flag}"] = (
                  rows1[flag].fillna(False).astype(bool)
                & rows2[flag].fillna(False).astype(bool)
            ).astype(int)

        features = add_conflict_features(features, rows1, rows2)

        X_batches.append(features)

        if training and entity_id_map is not None:
            y_batch = np.array([entity_id_map.get(p1) == entity_id_map.get(p2)
                               for p1, p2 in batch_pairs], dtype=int)
            y_batches.append(y_batch)

    X = pd.concat(X_batches, ignore_index=True)
    y = np.concatenate(y_batches) if y_batches else None
    logger.info(f"Feature generation завершена. Новое shape: {X.shape}")
    return X, y
