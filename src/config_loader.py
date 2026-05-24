# src/config_loader.py
import yaml
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

try:
    profile
except NameError:
    def profile(func):
        return func

@profile
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()

@profile
def get_active_experiment():
    """
    Возвращает кортеж (config, model_path), где config – словарь параметров эксперимента,
    а model_path – путь к файлу модели.
    Если active_experiment_report не указан или файл не найден, возвращает (CONFIG, None)
    """
    report_path = CONFIG.get('active_experiment_report')
    if report_path and Path(report_path).exists():
        with open(report_path, 'r') as f:
            experiment = json.load(f)
        exp_config = experiment.get('config')
        model_filename = experiment.get('model_filename')
        if exp_config and model_filename:
            model_dir = Path(exp_config['paths']['model_dir'])
            model_path = model_dir / model_filename
            return exp_config, model_path
    # fallback
    model_dir = Path(CONFIG['paths']['model_dir'])
    default_model = model_dir / CONFIG.get('active_model', 'catboost_model_latest.cbm')
    return CONFIG, default_model if default_model.exists() else None
