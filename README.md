# Entity Resolution Pipeline

Система для связывания профилей (entity resolution) на основе событийной активности.  
Пайплайн включает предобработку, многопроходный блокинг, генерацию признаков, обучение CatBoost, иерархичную кластеризацию и оценку качества.

Отчет о проделанной работае представлен [здесь](https://docs.google.com/document/d/1jLyp1D6w2UUIogO51iZ4IHcztMuKpNYq4QcyobH6NVw/edit?usp=sharing)
---

## Быстрый старт

### 1. Клонирование и установка

```bash
git clone <url-репозитория>
cd workshop_mipt_spring_2026
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Подготовка данных

Поместите исходные .parquet файлы в следующие папки (создаются автоматически):

| Назначение | Путь |
|------------|------|
| Обучающие данные (с колонкой entity_id) | data/raw/training/ |
| Данные для инференса (без entity_id) | data/raw/inference/ |
| Размеченные данные для оценки | data/raw/evaluation/ |

### 3. Обучение модели

```bash
python train_model.py
```

Скрипт:

- берёт самый свежий `.parquet` из `data/raw/training/`
- выполняет предобработку, блокинг, балансировку
- обучает CatBoost и сохраняет модель в `models/`
- создаёт JSON-отчёт в `data/training-results/`
- автоматически обновляет `config.yaml`, указывая активный эксперимент

После обучения в консоли вы увидите:

- парные метрики (precision/recall/f1 для порога 0.99)
- recovery rate кластеризации (доля полностью восстановленных мульти-сущностей)

Дополнительно в `data/training-results` сохраняется детальный отчет с метриками эксперимента.

### 4. Использование конкретной модели (переключение эксперимента)

Все скрипты всегда используют активный эксперимент, указанный в `config.yaml` (ключ `active_experiment_report`).


Структура  `config.yaml`
```yaml
active_experiment_report: data/training-results/training_report_*.json # Название эксперимента
blocking: # Параметры блокинга
  max_group_size: 800
  use_geo_device: true
  use_sites: true
clustering: # Параметры кластеризации
  eval_split: test
  method: complete
  threshold: 0.85
confidence: # Параметры для автоматического объединения
  check_conflicts: true
  min_avg_prob: 0.99
model: # Параметры CatBoost
  depth: 3
  early_stopping_rounds: 150
  eval_metric: F1
  iterations: 2500
  l2_leaf_reg: 30
  learning_rate: 0.03
  scale_pos_weight: 10
  threshold: 0.99
paths: # Пути
  evaluation_results: data/evaluation-results
  inference_results: data/inference-results
  model_dir: models
  training_results: data/training-results
seed: 42
training: # Коэффициент балансировки обучающей выборки (чем 
  balance_ratio: 8

```
Чтобы переключиться на другую ранее обученную модель:

```bash
python set_active_experiment.py data/training-results/training_report_*.json
```

После этого `predict_entities.py` и `evaluate.py` будут использовать ту модель и параметры, которые зафиксированы в этом отчёте.

### 5. Инференс на новых данных (без разметки)

Запустите:

```bash
python predict_entities.py data/raw/inference/my_data.parquet
```

Если файл один, можно без аргументов – скрипт возьмёт последний файл из `data/raw/inference/`.

Выходные файлы (создаются в `data/inference-results/`):

- `<basename>_auto.csv` – профили, объединённые в уверенные кластеры (каждому профилю присвоен entity_N)
- `<basename>_review.csv` – кластеры, которые модель не смогла уверенно классифицировать (с метриками для ручной проверки)


### 6. Оценка качества на размеченных данных

Если у вас есть тестовые данные с истинными entity_id, выполните:

```bash
python evaluate.py data/raw/evaluation/labeled.parquet
```

Скрипт:

- загружает активную модель
- выполняет полный пайплайн (предобработка, блокинг, кластеризация)
- сравнивает полученные кластеры с истинными сущностями
- выводит recovery rate и число смешанных кластеров
- сохраняет JSON-отчёт в `data/evaluation-results/evaluation_report.json`

Чтобы сохранить кластеры (для визуального анализа), добавьте флаг:

```bash
python evaluate.py --output_clusters my_clusters.csv
```
### 7. Где смотреть метрики

| Что | Где |
|-----|-----|
| Парные метрики (precision/recall) | терминал при запуске train_model.py |
| Recovery rate кластеризации | терминал при train_model.py или evaluate.py |
| Полный отчёт эксперимента (конфиг, метрики, размеры кластеров) | data/training-results/training_report_*.json |
| Отчёт оценки | data/evaluation-results/evaluation_report.json |
| Логи | терминал (logging уровня INFO) |

### 8. Структура проекта
```text
.
├── config.yaml                    # Текущий конфиг + указатель активного эксперимента
├── requirements.txt
├── train_model.py
├── predict_entities.py
├── evaluate.py
├── set_active_experiment.py
├── data/
│   ├── raw/                       # исходные .parquet (не в git)
│   │   ├── training/
│   │   ├── inference/
│   │   └── evaluation/
│   ├── processed/                 # кэш индексов (не в git)
│   ├── training-results/          # JSON‑отчёты экспериментов (в git)
│   ├── evaluation-results/        # JSON‑отчёты оценки (не в git)
│   └── inference-results/         # CSV с предсказаниями (не в git)
├── models/                        # .cbm модели (в git)
└── src/                           # исходные модули
    ├── config_loader.py
    ├── utils.py
    ├── preprocessing.py
    ├── aggregation_profiles.py
    ├── site_activity_processor.py
    ├── cat_activity_processor.py
    ├── profiles_processor.py
    ├── blocking.py
    ├── features.py
    └── clustering.py
```

### 9. Настройка параметров

Все параметры (размер блоков, гиперпараметры модели, пороги) задаются в `config.yaml`.

После обучения конфиг автоматически обновляется, но вы можете отредактировать его вручную, чтобы изменить параметры для следующего эксперимента.

Основные параметры:

| Параметр | Описание |
|----------|----------|
| seed | глобальный seed для воспроизводимости |
| blocking.max_group_size | максимальный размер группы для генерации пар |
| model.* | параметры CatBoost |
| training.balance_ratio | соотношение отрицательных/положительных пар |
| clustering.threshold | порог для иерархической кластеризации |
| confidence.* | пороги для определения уверенных кластеров (используются в predict_entities.py) |

### 10. FAQ

**Q: Почему при инференсе удаляются колонки entity_id / entity_type?**

A: Это имитация production‑среды – на реальных данных разметки нет. Если они есть, система их игнорирует.

**Q: Как интерпретировать _review.csv?**

A: Это кластеры, которые модель сочла недостаточно уверенными. В каждой строке указаны средняя вероятность, доля пар без общих сайтов, разброс дат. Вы можете вручную объединить или разделить их.

**Q: Как добавить новые признаки или проходы блокинга?**

A: Измените features.py или blocking.py, затем переобучите модель – старый эксперимент останется, а новый будет сохранён отдельно.

## Профилирование
`kernprof -l -v train_model.py` - запуск профайлера
`kernprof -l -v -o "reports/profile_$(date +%Y%m%d_%H%M%S).lprof" train_model.py` - с соранением timestamp метки в отчёте
`python -m line_profiler reports/profiling/train_model.py.lprof > profile_report.txt` - сохранить в .txt для анализа
``

## Покрытие
`coverage run --data-file=reports/.coverage train_model.py` - покрытие с сохранением в директорию (если параметр `--data-file` не указан, то `.coverage` будет сохранен в текущей директории)
`coverage html --data-file=reports/coverage/.coverage` - оценка покрытия


##  Требования к окружению

| Требование | Значение |
|------------|----------|
| Python | 3.10+ |
| RAM | 8+ GB (для обработки 4M пар) |
| Установка зависимостей | `pip install -r requirements.txt` |

## Воспроизведение результатов

1. Клонировать репозиторий.
2. Установить зависимости.
3. Поместить размеченный файл в `data/raw/training/`.
4. Запустить `python train_model.py`.
5. После обучения использовать `predict_entities.py` или `evaluate.py` как описано выше.

**Можно сразу использовать готовую модель и config по умолчанию**, которые были загружены в репозиторий.

**Лицензия**: Студенческая

**Автор**: Смирнова Анастасия
