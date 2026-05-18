# src/clustering.py
import logging
import networkx as nx
from collections import Counter, defaultdict

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    profile
except NameError:
    def profile(func):
        return func

@profile
def evaluate_clustering(clusters, profiles_df, split="test"):
    """
    Оценка качества кластеризации. Только для обучения
    """
    test_df = profiles_df[profiles_df["split"] == split]
    true_entity_map = test_df["entity_id"].to_dict()
    true_multi = test_df.groupby("entity_id").size()
    true_multi = true_multi[true_multi > 1]
    n_true_multi = len(true_multi)
    perfect = 0
    partial = 0
    for cluster in clusters:
        eids = {true_entity_map.get(pid) for pid in cluster if pid in true_entity_map}
        eids.discard(None)
        if len(eids) == 1:
            perfect += 1
        elif len(eids) > 1:
            partial += 1
    recovery_rate = perfect / n_true_multi if n_true_multi > 0 else 0
    print(f"  Полностью восстановлено: {perfect}/{n_true_multi} ({recovery_rate:.1%})")
    print(f"  Частично (смешано): {partial}")
    return {
        'recovery_rate': recovery_rate,
        'perfect_recovered': perfect,
        'n_true_multi': n_true_multi,
        'partial': partial
    }

@profile
def hierarchical_average_clustering(pairs, probabilities, all_profiles, threshold=0.7):
    """
    Агломеративная кластеризация с методом средней связи (average linkage).

    Аргументы:
        pairs: список пар (profile_id1, profile_id2)
        probabilities: массив вероятностей для каждой пары
        all_profiles: список всех profile_id (можно взять из profiles_df.index)
        threshold: порог сходства (0..1) для отсечения дендрограммы.
                   Рекомендуемый диапазон: 0.6-0.8.
    """
    logger.info(f"Hierarchical average clustering | threshold={threshold}")

    profile_list = list(all_profiles)
    n = len(profile_list)
    idx_map = {p: i for i, p in enumerate(profile_list)}

    # Инициализируем матрицу расстояний единицами (диагональ 0)
    dist_matrix = np.ones((n, n))
    np.fill_diagonal(dist_matrix, 0)

    # Заполняем известные расстояния (1 - prob)
    for (a, b), p in zip(pairs, probabilities):
        i, j = idx_map[a], idx_map[b]
        dist = 1 - p
        if dist < dist_matrix[i, j]:   # берём минимальное расстояние (максимальную вероятность)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Сжатый вид (треугольник)
    condensed = squareform(dist_matrix)

    # Иерархическая кластеризация методом средней связи
    Z = linkage(condensed, method='average')

    # Отсекаем по порогу расстояния (1 - threshold)
    labels = fcluster(Z, 1 - threshold, criterion='distance')

    # Группируем
    clusters = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(lbl, []).append(profile_list[i])

    result = [c for c in clusters.values() if len(c) >= 2]
    logger.info(f"  Создано {len(result)} кластеров")
    return result

@profile
def is_confident_cluster(cluster, prob_map, site_index, profiles_df,
                         min_avg_prob=0.9,
                         max_pairs_without_sites_ratio=0.5,
                         max_days_gap=730):
    """
    Проверяет, является ли кластер достаточно уверенным для автоматического принятия.
    Возвращает (is_confident, details), где details – словарь с метриками.
    """
    n = len(cluster)
    if n < 2:
        return False, {}

    # 1. Средняя и минимальная вероятности
    probs = []
    for i in range(n):
        for j in range(i+1, n):
            key = tuple(sorted((cluster[i], cluster[j])))
            p = prob_map.get(key, 0.0)
            probs.append(p)
    avg_prob = np.mean(probs) if probs else 0.0
    min_prob = np.min(probs) if probs else 1.0

    # 2. Доля пар без общих сайтов
    total_pairs = 0
    no_sites_pairs = 0
    for i in range(n):
        sites_i = site_index.get(cluster[i], {}).get("_all", set())
        for j in range(i+1, n):
            sites_j = site_index.get(cluster[j], {}).get("_all", set())
            total_pairs += 1
            if not (sites_i & sites_j):
                no_sites_pairs += 1
    no_sites_ratio = no_sites_pairs / total_pairs if total_pairs else 0.0

    # 3. Разброс дат последней активности
    dates = []
    for pid in cluster:
        dt = profiles_df.loc[pid].get('last_event_date')
        if pd.notna(dt):
            dates.append(pd.to_datetime(dt))
    days_gap = (max(dates) - min(dates)).days if len(dates) > 1 else 0

    # Решение
    is_confident = (avg_prob >= min_avg_prob and
                    no_sites_ratio <= max_pairs_without_sites_ratio and
                    days_gap <= max_days_gap)

    details = {
        'avg_prob': avg_prob,
        'min_prob': min_prob,
        'no_sites_ratio': no_sites_ratio,
        'days_gap': days_gap,
        'size': n
    }
    return is_confident, details
