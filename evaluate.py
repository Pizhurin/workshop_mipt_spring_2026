# evaluate.py – оценка качества на размеченных данных
import sys
import json
import argparse
from pathlib import Path
import logging
import pandas as pd
from catboost import CatBoostClassifier

from src.config_loader import get_active_experiment
from src.profiles_processor import process_profiles
from src.blocking import blocking_pipeline
from src.features import build_features
from src.clustering import hierarchical_average_clustering, evaluate_clustering

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    profile
except NameError:
    def profile(func):
        return func

def evaluate_on_labeled_data(df_raw, config, model, threshold=None,
                             output_report=None, output_clusters=None):
    if threshold is None:
        threshold = config['clustering']['threshold']

    if 'entity_id' not in df_raw.columns:
        raise ValueError("Входные данные должны содержать колонку 'entity_id' для оценки.")

    profiles_df, site_index, cat_index = process_profiles(df_raw, split_data=False)
    profiles_df['split'] = 'test'

    pairs = blocking_pipeline(profiles_df, site_index)
    logger.info(f"Сгенерировано пар: {len(pairs)}")
    if not pairs:
        logger.warning("Нет пар для кластеризации.")
        return None

    X, _ = build_features(pairs, profiles_df, site_index, cat_index,
                          idf_sites=None, idf_domains=None, training=False)
    probabilities = model.predict_proba(X)[:, 1]

    all_profiles = profiles_df.index.tolist()
    clusters = hierarchical_average_clustering(pairs, probabilities, all_profiles, threshold=threshold)
    metrics = evaluate_clustering(clusters, profiles_df, split="test")

    sizes_true = profiles_df.groupby("entity_id").size().value_counts().sort_index().to_dict()
    sizes_pred = {size: sum(1 for cl in clusters if len(cl) == size) for size in set(len(cl) for cl in clusters)}

    result = {
        "recovery_rate": metrics['recovery_rate'],
        "perfect_recovered": metrics['perfect_recovered'],
        "n_true_multi": metrics['n_true_multi'],
        "partial_clusters": metrics['partial'],
        "total_clusters": len(clusters),
        "true_multi_sizes": sizes_true,
        "predicted_cluster_sizes": sizes_pred,
        "threshold_used": threshold,
        "num_pairs_generated": len(pairs),
        "num_profiles": len(profiles_df)
    }

    print("\n" + "="*60)
    print("Результат оценки")
    print("="*60)
    print(f"Всего мульти-сущностей (размер >1): {metrics['n_true_multi']}")
    print(f"Полностью восстановлено: {metrics['perfect_recovered']} ({metrics['recovery_rate']:.1%})")
    print(f"Частично смешанных кластеров: {metrics['partial']}")

    # Сохранение отчёта
    eval_dir = Path(config['paths'].get('evaluation_results', 'data/evaluation-results'))
    eval_dir.mkdir(parents=True, exist_ok=True)

    if output_report is None:
        output_report = eval_dir / "evaluation_report.json"
    else:
        output_report = Path(output_report)
        if not output_report.is_absolute():
            output_report = eval_dir / output_report

    with open(output_report, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Отчёт сохранён в {output_report}")

    if output_clusters:
        cluster_path = Path(output_clusters)
        if not cluster_path.is_absolute():
            cluster_path = eval_dir / cluster_path
        cluster_path.parent.mkdir(parents=True, exist_ok=True)
        cluster_df = pd.DataFrame([(idx, pid) for idx, cl in enumerate(clusters) for pid in cl],
                                  columns=["cluster_id", "profile_id"])
        cluster_df.to_csv(cluster_path, index=False)
        logger.info(f"Кластеры сохранены в {cluster_path}")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", help="Путь к размеченным данным (parquet)")
    parser.add_argument("--output_report", default=None, help="Имя файла отчёта (JSON)")
    parser.add_argument("--output_clusters", default=None, help="Сохранить кластеры в CSV")
    parser.add_argument("--threshold", type=float, default=None, help="Порог кластеризации")
    args = parser.parse_args()

    if args.input:
        data_path = Path(args.input)
    else:
        raw_dir = Path("data/raw/evaluation")
        parquet_files = list(raw_dir.glob("*.parquet"))
        if not parquet_files:
            print("Ошибка: не указан входной файл и нет .parquet в data/raw/evaluation")
            sys.exit(1)
        data_path = max(parquet_files, key=lambda p: p.stat().st_mtime)
        print(f"Используем последний файл: {data_path}")

    df = pd.read_parquet(data_path)
    config, model_path = get_active_experiment()
    if model_path is None or not model_path.exists():
        print("Ошибка: не удалось загрузить модель. Сначала запустите train_model.py")
        sys.exit(1)
    # Загружаем модель
    model = CatBoostClassifier()
    model.load_model(model_path)
    evaluate_on_labeled_data(df, config, model, threshold=args.threshold,
                             output_report=args.output_report, output_clusters=args.output_clusters)
