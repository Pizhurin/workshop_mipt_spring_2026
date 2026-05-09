# Entity Resolution для маркетплейса скидок (Version 1 Data Processing Pipeline)

Структура репозитория:

```text
entity-resolution/
├── README.md
├── requirements.txt
├── .gitignore
├── test_mini_pipeline.py
│
├── data/
│   ├── raw/ # !Сюда кладем данные для обработки!
│   │   └── .gitkeep
│   └── processed/ # Тут можно посмотреть результаты всего пайплайна после запуска
│       └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── aggregation_profiles.py
│   ├── site_activity_processor.py
│   ├── cat_activity_processor.py
│   └── profiles_processor.py
│
├── notebooks/ # Здесь лежит ноутбук с EDA и дополнительными тестами
├── docs/ # Описание стратегии и логики работы модулей
├── app/ # На будущее для Streamlit
└── logs/ # На будущее для сохранение логов в файл
```

Система автоматического поиска и объединения дубликатов профилей клиентов по неполным и зашумлённым данным.

## Задача

Определить, какие профили на маркетплейсе скидок относятся к одному реальному человеку, и предложить их автоматическое или полуавтоматическое объединение.

## Данные

- **472 106** уникальных профилей
- **972 186** событий
- **5%** профилей — дубликаты
- Данные анонимизированы (детерминированно)

### Источник

Синтетический анонимизированный датасет профилей клиентов маркетплейса скидок. Предоставлен компанией Flocktory в рамках учебного практикума.

### Состав

| Поле | Тип | Описание |
|---|---|---|
| `created_at` | timestamp | Время события |
| `profile_id` | string | Идентификатор профиля |
| `entity_id` | string | Идентификатор реального пользователя (целевая переменная) |
| `entity_type` | string | `single_profile` или `multi_profile` |
| `email` | string | Анонимизированный email |
| `phone` | string | Анонимизированный телефон |
| `first_name` | string | Анонимизированное имя |
| `last_name` | string | Анонимизированная фамилия |
| `birthday` | date | Анонимизированная дата рождения |
| `sex` | string | Пол |
| `fs_features` | array | Исторические признаки профиля |
| `non_processing_features` | array | Технические признаки события |
| `realtime_features` | string | JSON с гео и временными признаками |

### Анонимизация

- Все персональные данные заменены детерминированно.
- Одно и то же исходное значение -> одно и то же синтетическое.
- Домен email и префикс телефона сохранены.
- Род имени и фамилии сохранён.

## Ограничения

Данные используются только в рамках практикума. Не подлежат распространению за пределами команды.

## Архитектура решения

```text
Сырые события -> Preprocessing -> Агрегация -> Blocking -> 
-> Feature Engineering -> Модель -> Кластеризация
```

## Структура проекта

```text
src/
├── preprocessing.py               # Парсинг, очистка, нормализация
├── aggregation_profiles.py        # Паспорт профиля
├── site_activity_processor.py     # Индекс цифрового следа
├── cat_activity_processor.py      # Индекс почтовой активности
└── profiles_processor.py          # Единая точка входа
```

## Быстрый старт

# 1. Клонируем репозиторий

```bash
git clone https://github.com/Pizhurin/workshop_mipt_spring_2026
cd workshop_mipt_spring_2026
git checkout data-pipeline-version-1
```

# 2. Загружаем сырые данные по нужному пути

Требования к данным:

Формат: `Parquet`

Обязательные колонки: `created_at`, `profile_id`, `entity_id`, `entity_type`, `email`, `phone`, `first_name`, `last_name`, `birthday`, `sex`, `fs_features`, `non_processing_features`, `realtime_features`

```bash
/data/raw
```

# 3. Устанавливаем зависимости
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# 4. Запускаем пайплайн

```bash
python test_mini_pipeline.py
```

# 5. Результат

На выходе формируются три структуры, которые **сохраняются в `data/processed/`**:

| Файл | Тип | Описание |
|---|---|---|
| `profiles_df.csv` | CSV | Паспорт профиля: демография, контакты, флаги (можно открыть в Excel) |
| `profiles_df.parquet` | Parquet | Та же таблица в бинарном формате (быстрая загрузка в Pandas) |
| `site_index.pkl` | Pickle (dict) | Цифровой след: посещённые сайты по 12 категориям |
| `cat_index.pkl` | Pickle (dict) | Почтовая активность: реакции на рассылки |
| `summary.txt` | TXT | Краткая сводка: количество профилей, время обработки, дата |

## Как загрузить сохранённые результаты

```python
import pandas as pd
import pickle

# Паспорт профиля
profiles_df = pd.read_parquet("data/processed/profiles_df.parquet")

# Индексы
with open("data/processed/site_index.pkl", "rb") as f:
    site_index = pickle.load(f)

with open("data/processed/cat_index.pkl", "rb") as f:
    cat_index = pickle.load(f)
```

## Текущий статус

- [x] Preprocessing (парсинг, нормализация)
- [x] Агрегация профилей (3 источника)
- [ ] Blocking (многопроходный)
- [ ] Feature Engineering
- [ ] Модель матчинга
- [ ] Кластеризация
- [ ] Демо-интерфейс

## Документация

- [Предлагаемая стратегия решения](docs/strategy.md)
- [Описание модулей](docs/modules_description.md)


## Лицензия

Проект выполнен в рамках учебного практикума. Данные используются под NDA.