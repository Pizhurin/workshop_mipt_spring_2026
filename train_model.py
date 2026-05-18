# train_model.py – полный пайплайн обучения с сохранением модели и отчёта
import pickle
import json
import datetime
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

from src.config_loader import CONFIG
from src.profiles_processor import process_profiles
from src.blocking import blocking_pipeline
from src.features import build_features, compute_idf_sites, compute_idf_email_domains
from src.clustering import hierarchical_average_clustering, evaluate_clustering

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    profile
except NameError:
    def profile(func):
        return func

@profile
def get_latest_parquet(directory="data/raw/training"):
    path = Path(directory)
    parquet_files = list(path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Нет .parquet файлов в {directory}")
    latest = max(parquet_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Выбран файл: {latest}")
    return latest

@profile
def main():
    # 1. Загрузка и предобработка (с разбиением на train/val/test 60/20/20)
    DATA_PATH = get_latest_parquet()
    logger.info("Загрузка данных и предобработка...")
    df_raw = pd.read_parquet(DATA_PATH)
    profiles_df, site_index, cat_index = process_profiles(df_raw, split_data=True)

    # 2. Блокинг
    logger.info("Генерация пар через блокинг...")
    train_profiles = profiles_df[profiles_df["split"] == "train"]
    val_profiles = profiles_df[profiles_df["split"] == "val"]
    test_profiles = profiles_df[profiles_df["split"] == "test"]

    train_site_index = {pid: site_index[pid] for pid in train_profiles.index}
    val_site_index = {pid: site_index[pid] for pid in val_profiles.index}
    test_site_index = {pid: site_index[pid] for pid in test_profiles.index}

    train_pairs = blocking_pipeline(
        train_profiles, train_site_index,
        max_group_size=CONFIG['blocking']['max_group_size']
    )
    val_pairs = blocking_pipeline(
        val_profiles, val_site_index,
        max_group_size=CONFIG['blocking']['max_group_size']
    )
    test_pairs = blocking_pipeline(
        test_profiles, test_site_index,
        max_group_size=CONFIG['blocking']['max_group_size']
    )

    # 3. Вычисление IDF
    idf_sites = compute_idf_sites(
        site_index, (profiles_df["split"] == "train").sum()
    )
    idf_domains = compute_idf_email_domains(
        profiles_df[profiles_df["split"] == "train"]
    )

    # 4. Балансировка пар
    entity_map = profiles_df["entity_id"].to_dict()
    pos_pairs = [(a,b) for a,b in train_pairs if entity_map[a]==entity_map[b]]
    neg_pairs = [(a,b) for a,b in train_pairs if entity_map[a]!=entity_map[b]]
    ratio = CONFIG['training']['balance_ratio']
    n_neg = min(len(pos_pairs) * ratio, len(neg_pairs))
    rng = np.random.default_rng(CONFIG['seed'])
    sampled_neg = [
        neg_pairs[i] for i in rng.choice(len(neg_pairs), n_neg, replace=False)
    ]
    train_pairs_balanced = pos_pairs + sampled_neg
    rng.shuffle(train_pairs_balanced)
    logger.info(
        f"Train: {len(pos_pairs)} pos, {len(sampled_neg)} neg (ratio {ratio})"
    )

    # 5. Построение признаков
    X_train, y_train = build_features(
        train_pairs_balanced, profiles_df, site_index, cat_index,
        idf_sites, idf_domains, training=True
    )
    X_val, y_val = build_features(
        val_pairs, profiles_df, site_index, cat_index,
        idf_sites, idf_domains, training=True
    )
    X_test, y_test = build_features(
        test_pairs, profiles_df, site_index, cat_index,
        idf_sites, idf_domains, training=True
    )

    # 6. Обучение модели
    model = CatBoostClassifier(
        iterations=CONFIG['model']['iterations'],
        learning_rate=CONFIG['model']['learning_rate'],
        depth=CONFIG['model']['depth'],
        l2_leaf_reg=CONFIG['model']['l2_leaf_reg'],
        random_seed=CONFIG['seed'],
        verbose=100,
        early_stopping_rounds=CONFIG['model']['early_stopping_rounds'],
        scale_pos_weight=CONFIG['model']['scale_pos_weight'],
        eval_metric=CONFIG['model']['eval_metric'],
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # 7. Оценка на тесте (парная)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    thr = CONFIG['clustering']['threshold']
    y_pred = (y_test_proba >= thr).astype(int)
    print("\n--- Оценка пар на тесте ---")
    print(classification_report(y_test, y_pred, digits=4))

    # 8. Кластеризация на тесте
    all_test_profiles = set()
    for a,b in test_pairs:
        all_test_profiles.add(a)
        all_test_profiles.add(b)
    all_test_profiles = list(all_test_profiles)
    clusters = hierarchical_average_clustering(test_pairs, y_test_proba, all_test_profiles, threshold=thr)
    metrics = evaluate_clustering(clusters, profiles_df, split="test")

    # 9. Сохранение модели и отчёта
    # Генерируем уникальный идентификатор эксперимента
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_id = timestamp

    # Папка для моделей
    model_dir = Path(CONFIG['paths']['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_filename = f"catboost_model_{experiment_id}.cbm"
    model_path = model_dir / model_filename
    model.save_model(model_path)
    logger.info(f"Модель сохранена как {model_path}")

    # Папка для результатов обучения
    results_dir = Path(CONFIG['paths'].get('training_results', 'data/training-results'))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Подготовка отчёта
    sizes_true = profiles_df[profiles_df["split"] == "test"].groupby("entity_id").size().value_counts().sort_index().to_dict()
    sizes_pred = {size: sum(1 for cl in clusters if len(cl) == size) for size in set(len(cl) for cl in clusters)}

    report = {
        "experiment_id": experiment_id,
        "model_filename": model_filename,
        "recovery_rate": metrics['recovery_rate'],
        "perfect_recovered": metrics['perfect_recovered'],
        "n_true_multi": metrics['n_true_multi'],
        "partial_clusters": metrics['partial'],
        "total_clusters": len(clusters),
        "true_multi_sizes": sizes_true,
        "predicted_cluster_sizes": sizes_pred,
        "threshold_used": thr,
        "num_pairs_generated": len(test_pairs),
        "num_profiles_test": len(profiles_df[profiles_df["split"] == "test"]),
        "pair_classification_report": classification_report(y_test, y_pred, output_dict=True),
        "config": {
            "seed": CONFIG['seed'],
            "blocking": CONFIG['blocking'],
            "model": CONFIG['model'],
            "training": CONFIG['training'],
            "clustering": CONFIG['clustering'],
            "confidence": CONFIG.get('confidence', {}),
            "paths": CONFIG['paths']
        }
    }

    report_path = results_dir / f"training_report_{experiment_id}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Отчёт сохранён в {report_path}")

    # Обновляем config.yaml: устанавливаем active_experiment_report на этот отчёт
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg['active_experiment_report'] = str(report_path.resolve())
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Конфиг обновлён: active_experiment_report = {report_path}")

    print(f"Recovery на тесте: {metrics['recovery_rate']:.1%} (perfect: {metrics['perfect_recovered']}/{metrics['n_true_multi']})")

if __name__ == "__main__":
    main()
