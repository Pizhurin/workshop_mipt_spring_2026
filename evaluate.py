# evaluate.py – оценка качества на размеченных данных
import sys
import json
import argparse
from pathlib import Path
import logging
import pandas as pd
from catboost import CatBoostClassifier

from sklearn.metrics import classification_report

from src.config_loader import get_active_experiment
from src.profiles_processor import process_profiles
from src.blocking import blocking_pipeline, compute_blocking_recall
from src.features import build_features
from src.clustering import hierarchical_clustering, evaluate_clustering
from src.utils import make_json_serializable

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    profile
except NameError:
    def profile(func):
        return func

def evaluate_on_labeled_data(df_raw, config, model,
                             thr_clustering=None, thr_matching=None,
                             output_report=None, output_clusters=None):
    if thr_clustering is None:
        thr_clustering = config['clustering']['threshold']
    if thr_matching is None:
        thr_matching = config['model']['threshold']

    if 'entity_id' not in df_raw.columns:
        raise ValueError("Входные данные должны содержать колонку 'entity_id' для оценки.")

    profiles_df, site_index, cat_index = process_profiles(df_raw, split_data=False)
    profiles_df['split'] = 'test'

    pairs = blocking_pipeline(profiles_df, site_index)
    logger.info(f"Сгенерировано пар: {len(pairs)}")
    if not pairs:
        logger.warning("Нет пар для кластеризации.")
        return None

    blocking_recall = compute_blocking_recall(pairs, profiles_df, split_name='test')

    X, _ = build_features(pairs, profiles_df, site_index, cat_index,
                          idf_sites=None, idf_domains=None, training=False)
    
    probabilities = model.predict_proba(X)[:, 1]
    
    logger.info(f"Threshold для метчинга: {thr_matching}")


    print("\n--- Оценка пар на тесте ---")
    entity_map = profiles_df['entity_id'].to_dict()
    true = [1 if entity_map[a] == entity_map[b] else 0 for a, b in pairs]

    pred = (probabilities >= thr_matching).astype(int)
    
    print(classification_report(true, pred, digits=4))

    all_profiles = profiles_df.index.tolist()
    logger.info(f"Threshold для кластеризации: {thr_clustering}")
    clusters = hierarchical_clustering(pairs, probabilities, all_profiles, threshold=thr_clustering)
    metrics = evaluate_clustering(clusters, profiles_df, pairs=pairs, split="test")

    sizes_true = profiles_df.groupby("entity_id").size().value_counts().sort_index().to_dict()
    sizes_pred = {size: sum(1 for cl in clusters if len(cl) == size) for size in set(len(cl) for cl in clusters)}

    result = {
        "blocking_recall": blocking_recall,
        "pair_classification_report": classification_report(true, pred, output_dict=True),
        "n_lost_by_clustering": metrics['n_lost_by_clustering'],
        "recovery_multi": metrics['recovery_multi'],
        "n_true_multi": metrics['n_true_multi'],
        "perfect_multi": metrics['perfect_multi'],
        "partially_recovered": metrics['partially_recovered'],
        "fully_broken": metrics['fully_broken'],
        "merged_with_others": metrics['merged_with_others'],
        "total_clusters": metrics['total_clusters'],
        "status_counts": metrics['status_counts'],
        "true_multi_sizes": sizes_true,
        "predicted_cluster_sizes": sizes_pred,
        "threshold_matching": thr_matching,
        "thershold_clustering": thr_clustering,
        "num_pairs_generated": len(pairs),
        "num_profiles": len(profiles_df)
    }

    result = make_json_serializable(result)

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
    evaluate_on_labeled_data(df, config, model, 
                             thr_clustering=args.threshold,
                             thr_matching=args.threshold,
                             output_report=args.output_report, output_clusters=args.output_clusters)
