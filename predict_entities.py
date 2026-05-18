# predict_entities.py
import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from catboost import CatBoostClassifier

from src.config_loader import get_active_experiment
from src.profiles_processor import process_profiles
from src.blocking import blocking_pipeline
from src.features import build_features
from src.clustering import hierarchical_average_clustering, is_confident_cluster

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    profile
except NameError:
    def profile(func):
        return func

@profile
def load_model_from_experiment():
    config, model_path = get_active_experiment()
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}. Сначала запустите train_model.py")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return config, model

@profile
def predict_entities(df_raw, model, config, threshold=None,
                     output_dir=None,
                     base_name=None):
    """
    Параметры:
        output_dir: папка для сохранения результатов
        base_name: базовое имя файлов (без расширения). Если None, генерируется из времени.
    """
    if output_dir is None:
        output_dir = config['paths'].get('inference_results', 'data/inference-results')

    if threshold is None:
        threshold = config['clustering']['threshold']
    confidence_cfg = config.get('confidence', {})
    min_avg_prob = confidence_cfg.get('min_avg_prob', 0.9)
    max_no_sites_ratio = confidence_cfg.get('max_pairs_without_common_sites_ratio', 0.5)
    max_days_gap = confidence_cfg.get('max_days_between_events', 730)

    # Создаём папку для результатов
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Определяем базовое имя файлов
    if base_name is None:
        base_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    auto_csv = out_path / f"{base_name}_auto.csv"
    review_csv = out_path / f"{base_name}_review.csv"

    profiles_df, site_index, cat_index = process_profiles(df_raw, split_data=False)
    logger.info(f"Обработано профилей: {len(profiles_df)}")

    pairs = blocking_pipeline(profiles_df, site_index)
    logger.info(f"Сгенерировано пар: {len(pairs)}")

    if not pairs:
        logger.warning("Нет пар для кластеризации. Возвращаем одиночные профили.")
        result = pd.DataFrame({
            "profile_id": profiles_df.index,
            "predicted_entity_id": profiles_df.index
        })
        result.to_csv(auto_csv, index=False)
        logger.info(f"Результат сохранён в {auto_csv}")
        return result

    X, _ = build_features(pairs, profiles_df, site_index, cat_index,
                          idf_sites=None, idf_domains=None, training=False)
    probabilities = model.predict_proba(X)[:, 1]

    all_profiles = profiles_df.index.tolist()
    clusters = hierarchical_average_clustering(pairs, probabilities, all_profiles, threshold=threshold)
    logger.info(f"Получено кластеров: {len(clusters)}")

    prob_map = {tuple(sorted((a, b))): p for (a, b), p in zip(pairs, probabilities)}

    auto_clusters = []
    review_clusters_with_meta = []
    for cl in clusters:
        is_conf, details = is_confident_cluster(
            cl, prob_map, site_index, profiles_df,
            min_avg_prob=min_avg_prob,
            max_pairs_without_sites_ratio=max_no_sites_ratio,
            max_days_gap=max_days_gap
        )
        if is_conf:
            auto_clusters.append(cl)
        else:
            review_clusters_with_meta.append((cl, details))

    review_records = []
    for idx, (cl, details) in enumerate(review_clusters_with_meta):
        cluster_review_id = f"review_{idx}"
        for pid in cl:
            row = profiles_df.loc[pid]
            review_records.append({
                'cluster_id': cluster_review_id,
                'profile_id': pid,
                'first_name_clean': row.get('first_name_clean', ''),
                'last_name_clean': row.get('last_name_clean', ''),
                'email_domain': ', '.join(row.get('email_domain', [])),
                'phone_prefix': ', '.join(row.get('phone_prefix', [])),
                'geoname_id': row.get('geoname_id', ''),
                'device': ', '.join(row.get('device', [])),
                'osfamily': ', '.join(row.get('osfamily', [])),
                'country': row.get('country', ''),
                'avg_prob_in_cluster': details['avg_prob'],
                'min_prob_in_cluster': details['min_prob'],
                'no_sites_ratio': details['no_sites_ratio'],
                'days_gap': details['days_gap']
            })
        logger.debug(f"Кластер размера {len(cl)} отправлен на проверку как {cluster_review_id}")

    if review_records:
        review_df = pd.DataFrame(review_records)
        review_df.to_csv(review_csv, index=False)
        logger.info(f"Неуверенные кластеры сохранены в {review_csv}")

    # Присвоение entity_id
    cluster_id_map = {}
    entity_counter = 0
    for cl in auto_clusters:
        eid = f"entity_{entity_counter}"
        for pid in cl:
            cluster_id_map[pid] = eid
        entity_counter += 1
    for pid in all_profiles:
        if pid not in cluster_id_map:
            cluster_id_map[pid] = f"entity_{entity_counter}"
            entity_counter += 1

    auto_df = pd.DataFrame({
        "profile_id": list(cluster_id_map.keys()),
        "predicted_entity_id": list(cluster_id_map.values())
    })
    auto_df.to_csv(auto_csv, index=False)
    logger.info(f"Результат для уверенных сущностей сохранён в {auto_csv}")

    return auto_df

@profile
def get_latest_parquet(directory="data/raw/inference"):
    path = Path(directory)
    parquet_files = list(path.glob("*.parquet"))
    if not parquet_files:
        return None
    latest = max(parquet_files, key=lambda p: p.stat().st_mtime)
    return latest

if __name__ == "__main__":
    if len(sys.argv) == 2:
        input_path = sys.argv[1]
    else:
        input_path = get_latest_parquet()
        if input_path is None:
            print("Укажите путь к данным: python predict_entities.py <input_parquet>")
            print("   или положите файл .parquet в data/raw/inference и запустите без аргументов")
            sys.exit(1)
        print(f"Аргумент не указан, используем последний файл: {input_path}")

    data = pd.read_parquet(input_path)
    # Удаляем колонки разметки, если они есть (имитация production)
    if 'entity_id' in data.columns:
        data = data.drop(columns=['entity_id', 'entity_type'], errors='ignore')
        print("Колонки entity_id/entity_type удалены (режим production).")

    config, model = load_model_from_experiment()
    # Используем имя входного файла (без расширения) как базовое
    base_name = Path(input_path).stem
    predict_entities(data, model, config, threshold=config['clustering']['threshold'],
                     output_dir="data/inference-results", base_name=base_name)
    print("Готово.")
