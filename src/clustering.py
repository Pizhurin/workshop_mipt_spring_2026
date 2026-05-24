# src/clustering.py
import logging
from collections import Counter, defaultdict

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from src.config_loader import CONFIG

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    profile
except NameError:
    def profile(func):
        return func

@profile
def evaluate_clustering(clusters, profiles_df, pairs=None, split="test"):
    """
    Оценка кластеризации с полной картиной: сущности, прошедшие блокинг,
    делятся на оценённые кластеризацией и потерянные ею.
    """
    # Собираем профили из кластеров
    profiles_in_clusters = set()
    for cl in clusters:
        profiles_in_clusters.update(cl)
    
    if split:
        eval_df = profiles_df[profiles_df["split"] == split].copy()
    else:
        eval_df = profiles_df.copy()
    
    # Собираем профили, прошедшие блокинг
    if pairs is not None:
        profiles_in_pairs = set()
        for a, b in pairs:
            profiles_in_pairs.add(a)
            profiles_in_pairs.add(b)
        
        # Все сущности, у которых более 2 профилей в блокинге
        blocking_df = eval_df[eval_df.index.isin(profiles_in_pairs)]
        blocking_entity_counts = blocking_df.groupby('entity_id').size()
        entities_in_blocking = set(blocking_entity_counts[blocking_entity_counts >= 2].index)
        
        # Из них: те, у которых более 2 профилей в кластерах (оценённые)
        clustered_df = blocking_df[blocking_df.index.isin(profiles_in_clusters)]
        clustered_entity_counts = clustered_df.groupby('entity_id').size()
        entities_evaluated = set(clustered_entity_counts[clustered_entity_counts >= 2].index)
        
        # Потерянные кластеризацией: прошли блокинг, но <2 профилей в кластерах
        entities_lost_by_clustering = entities_in_blocking - entities_evaluated
        

        eval_df = clustered_df[clustered_df['entity_id'].isin(entities_evaluated)]
    else:
        entities_in_blocking = set()
        entities_lost_by_clustering = set()
        eval_df = eval_df[eval_df.index.isin(profiles_in_clusters)]
        entity_counts = eval_df.groupby('entity_id').size()
        multi_entities = entity_counts[entity_counts >= 2].index
        eval_df = eval_df[eval_df['entity_id'].isin(multi_entities)]

    if "profile_id" not in eval_df.columns:
        eval_df = eval_df.reset_index()

    entity_sizes = eval_df.groupby("entity_id").size()
    n_true_multi = len(entity_sizes)
    
    entity_profiles = eval_df.groupby("entity_id")["profile_id"].apply(set).to_dict()

    perfect_multi = 0
    fully_broken = 0
    partially_recovered = 0
    merged_with_others = 0

    detailed_stats = []

    for eid, true_pids in entity_profiles.items():
        true_size = len(true_pids)
        
        related_clusters = []
        for cluster in clusters:
            cluster_set = set(cluster)
            intersection = cluster_set & true_pids
            if intersection:
                related_clusters.append((cluster_set, len(intersection)))

        num_related = len(related_clusters)

        if num_related == 1 and len(related_clusters[0][0]) == true_size:
            perfect_multi += 1
            status = "perfect"
        elif num_related == 1:
            partially_recovered += 1
            merged_with_others += 1
            status = "merged"
        elif num_related > 1:
            partially_recovered += 1
            if any(len(cl[0]) > len(true_pids) for cl in related_clusters):
                merged_with_others += 1
            status = "split"
        else:
            fully_broken += 1
            status = "broken"

        detailed_stats.append({
            'entity_id': eid,
            'status': status
        })


    # Общая статистика
    n_passed_blocking = len(entities_in_blocking) if pairs is not None else n_true_multi
    n_lost_by_clustering = len(entities_lost_by_clustering) if pairs is not None else 0

    # Итоговые метрики
    recovery_multi = perfect_multi / n_true_multi if n_true_multi > 0 else 0.0

    print("=" * 80)
    print(f"{'Метрика':<50} {'Значение':<10} {'Процент'}")
    print("=" * 80)
    if pairs is not None:
        print(f"Мульти-сущностей, прошедших блокинг          {n_passed_blocking:<10} (100%)")
        print(f"  ├─ Оценено кластеризацией                  {n_true_multi:<10} ({n_true_multi/n_passed_blocking:.1%})")
        print(f"  └─ Потеряно кластеризацией                 {n_lost_by_clustering:<10} ({n_lost_by_clustering/n_passed_blocking:.1%})")
        print("-" * 80)
    print(f"Оценённых мульти-сущностей                   {n_true_multi:<10} (100%)")
    print(f"  ├─ Perfect (идеально)                      {perfect_multi:<10} ({recovery_multi:.1%})")
    print(f"  ├─ Partially recovered                     {partially_recovered:<10} ({partially_recovered/n_true_multi:.1%})")
    print(f"  │   ├─ Fully broken                        {fully_broken:<10} ({fully_broken/n_true_multi:.1%})")
    print(f"  │   └─ Merged with others                  {merged_with_others:<10}")
    print(f"Total clusters                               {len(clusters):<10}")
    print("=" * 80)

    # Статистика по статусам
    status_counts = pd.DataFrame(detailed_stats)['status'].value_counts().to_dict()

    return {
        "n_passed_blocking": n_passed_blocking,
        "n_lost_by_clustering": n_lost_by_clustering,
        "recovery_multi": recovery_multi,
        "n_true_multi": n_true_multi,
        "perfect_multi": perfect_multi,
        "partially_recovered": partially_recovered,
        "fully_broken": fully_broken,
        "merged_with_others": merged_with_others,
        "total_clusters": len(clusters),
        "status_counts": status_counts
    }
    
@profile
def hierarchical_clustering(
    pairs, probabilities, all_profiles,
    threshold=CONFIG['clustering']['threshold'],
    method=CONFIG['clustering']['method']
):
    """
    Агломеративная кластеризация с выбором метода.

    Аргументы:
        pairs: список пар (profile_id1, profile_id2)
        method: метод (average, ward, или complete)
        probabilities: массив вероятностей для каждой пары
        all_profiles: список всех profile_id (можно взять из profiles_df.index)
        threshold: порог сходства (0..1) для отсечения дендрограммы.
    """
    logger.info(f"Hierarchical {method} clustering | threshold={threshold}")

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

    # Иерархическая кластеризация c выбранным методом
    Z = linkage(condensed, method=method)

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
def is_confident_cluster(
    cluster, prob_map, site_index, profiles_df,
    min_avg_prob=CONFIG['confidence']['min_avg_prob']
):
    """
    Проверяет, является ли кластер достаточно уверенным для автоматического принятия.
    Возвращает (is_confident, details), где details – словарь с метриками.
    """
    n = len(cluster)
    if n < 2:
        return False, {}

    # Средняя вероятность
    probs = []
    for i in range(n):
        for j in range(i+1, n):
            key = tuple(sorted((cluster[i], cluster[j])))
            p = prob_map.get(key, 0.0)
            probs.append(p)
    avg_prob = np.mean(probs) if probs else 0.0
    min_prob = np.min(probs) if probs else 1.0

    # Решение
    is_confident = avg_prob >= min_avg_prob
    
    details = {
        'avg_prob': avg_prob,
        'min_prob': min_prob,
        'size': n
    }
    return is_confident, details
