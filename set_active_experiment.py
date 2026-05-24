# set_active_experiment.py
import sys
import yaml
from pathlib import Path

if len(sys.argv) != 2:
    print("Используется эксперимент: python set_active_experiment.py <path_to_report.json>")
    sys.exit(1)

report_path = Path(sys.argv[1])
if not report_path.exists():
    print(f"Файл не найден: {report_path}")
    sys.exit(1)

config_path = Path("config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config['active_experiment_report'] = str(report_path.resolve())
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f"Активный эксперимент установлен: {report_path}")
