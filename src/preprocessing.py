# preprocessing.py
"""
Предобработка и нормализация
"""

import re
import pandas as pd
import json
import numpy as np

import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── non_processing_feature, fs_features ────────────────
def parse_array_like_features(features_list):
    """
    Парсит массив признаков с словарь.
    Обрабатывает пары ключ/значение и флаги.
    """
    if not isinstance(features_list, (list, np.ndarray)) or features_list is None:
        return {}
    parsed_features = {}
    for feature in features_list:
        if isinstance(feature, str):
            parts = feature.split(':', 1)
            if len(parts) == 2:
                parsed_features[parts[0]] = parts[1]
            else:
                parsed_features[feature] = True  # Отдельные строки интерпертируем как флаги
    return parsed_features

# ─── realtime_features ───────────────────────────────────
def json_unpack(df_col):
    """
    Распаковывает JSON столбец.

    Args:
        df_col: Значение ячейки столбца, содержащее данные в формате JSON

    Returns:
        dict: Словарь с извлеченными данными или пустой словарь в случае ошибки
    """
    try:
        if pd.isna(df_col):
            return {}
        return json.loads(df_col)
    except (TypeError, json.JSONDecodeError):
        return {}

# ─── Entity ──────────────────────────────────────────────
def unify_entity_ids(df):
    """
    Для profile_id с несколькими entity_id берёт актуальный entity_id
    из последнего по времени события.
    """
    df_sorted = df.sort_values('created_at')
    last_event = df_sorted.groupby('profile_id').tail(1)

    # Собираем только те колонки, которые есть в данных
    cols_available = ['profile_id', 'entity_id']
    if 'entity_type' in df.columns:
        cols_available.append('entity_type')

    entity_mapping = last_event[cols_available]

    # Удаляем старые колонки, которые будем заменять, если они есть
    cols_to_drop = [c for c in ['entity_id', 'entity_type'] if c in df.columns]
    df = df.drop(columns=cols_to_drop).merge(
        entity_mapping,
        on='profile_id',
        how='left'
    )
    return df

# ─── Email ──────────────────────────────────────────────

def normalize_email(email_str):
    """Приводит email к нижнему регистру, удаляет пробелы, валидирует."""
    if pd.isna(email_str):
        return None
    s = str(email_str).strip().lower()
    s = s.replace(" ", "")
    if s.count("@") != 1:
        return None
    return s

def extract_email_domain(email_str):
    """Извлекает доменную часть email."""
    if pd.isna(email_str):
        return None
    try:
        return email_str.split("@")[1].lower()
    except (IndexError, AttributeError):
        return None


# ─── Phone ──────────────────────────────────────────────

def normalize_phone(phone_str):
    """Оставляет только цифры, приводит к формату 79XXXXXXXXX."""
    if pd.isna(phone_str):
        return None

    digits = re.sub(r"\D", "", str(phone_str))

    if len(digits) == 10:
        digits = "7" + digits
    elif len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    elif len(digits) == 11 and digits.startswith("7"):
        pass
    else:
        return None  # невалидная длина или структура

    return digits

def extract_phone_prefix(phone_str):
    """Извлекает код страны + первые 3 цифры оператора."""
    if pd.isna(phone_str):
        return None
    digits = normalize_phone(phone_str)
    if digits and len(digits) >= 4:
        return digits[:4]
    return None

# ─── Names ──────────────────────────────────────────────

def normalize_name(name_str):
    """Стрип, lower case, удаление спецсимволов кроме дефиса и апострофа."""
    if pd.isna(name_str):
        return None
    s = str(name_str).strip()
    s = s.lower()
    s = " ".join(s.split())
    s = re.sub(r"[^a-zа-яё\-']", "", s)

    # Если после очистки пустая строка - None
    if not s:
        return None
    return s

# ─── Birthday ─────────────────────────────────────────────

def normalize_birthday(birthday_val):
    """
    Приводит дату рождения к строке YYYY-MM-DD или None.
    """
    if pd.isna(birthday_val) or birthday_val is None:
        return None
    try:
        # Если это datetime или Timestamp
        if hasattr(birthday_val, "strftime"):
            return birthday_val.strftime("%Y-%m-%d")
        # Если это строка
        s = str(birthday_val).strip()
        if s in ("", "NaT", "None", "nan", "NaN"):
            return None
        return pd.Timestamp(s).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None

# ─── Извлекаем данные из json and array-like полей ─────────────────────────────────
def extract_array_features(df, col_name):
    """Для fs_features, non_processing_features"""
    parsed_array = df[col_name].apply(parse_array_like_features)
    parsed_array_df = pd.json_normalize(parsed_array)
    df = pd.concat([df, parsed_array_df], axis=1)
    df = df.drop(columns=[col_name])
    return df

def extract_json_features(df, col_name):
    """Для realtime_features"""
    parsed_json = df[col_name].apply(json_unpack)
    json_data = pd.json_normalize(parsed_json)
    df = pd.concat([df, json_data], axis=1)
    df = df.drop(columns=[col_name])
    return df

# ─── Применяем нормализацию и извлекаем доп.данные из контактов ────────────────────

def normalize_contacts(df):
    """Применяет нормализацию email, phone, имен ко всем строкам."""
    df = df.copy()
    df["email_normalized"] = df["email"].apply(normalize_email)
    df["email_domain"] = df["email"].apply(extract_email_domain)
    df["phone_normalized"] = df["phone"].apply(normalize_phone)
    df["phone_prefix"] = df["phone"].apply(extract_phone_prefix)
    df["first_name_clean"] = df["first_name"].apply(normalize_name)
    df["last_name_clean"] = df["last_name"].apply(normalize_name)
    df["birthday_clean"] = df["birthday"].apply(normalize_birthday)
    return df

# ─── Весь пайплайн ─────────────────────────────────────────────────────────────────
def full_preprocessing(df):
    """
    Полный пайплайн предобработки.
    """
    initial_rows = len(df)
    logger.info("=" * 50)
    logger.info("Запущен пайплайн предобработки")
    logger.info(f"Исходное количество строк: {initial_rows}")
    logger.info("-" * 50)

    # 1. Парсинг fs_features
    logger.info("Шаг 1/6: парсинг fs_features...")
    df = extract_array_features(df, "fs_features")
    logger.info(f"  Добавлено колонок из fs_features: {df.shape[1]}")

    # 2. Парсинг non_processing_features
    logger.info("Шаг 2/6: парсинг non_processing_features...")
    df = extract_array_features(df, "non_processing_features")
    logger.info(f"  Всего колонок после парсинга: {df.shape[1]}")

    # 3. Парсинг realtime_features
    logger.info("Шаг 3/6: парсинг realtime_features...")
    df = extract_json_features(df, "realtime_features")
    logger.info(f"  Всего колонок после парсинга: {df.shape[1]}")

    # 4. Удаление полных дубликатов
    logger.info("Шаг 4/6: удаление полных дубликатов...")
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    if removed > 0:
        logger.info(f"  Удалено полных дубликатов: {removed} строк")
    else:
        logger.info("  Полных дубликатов не найдено")

    # 5. Унификация entity_id
    logger.info("Шаг 5/6: унификация entity_id...")
    problem_count = df.groupby('profile_id')['entity_id'].nunique().max()
    if problem_count > 1:
        logger.warning(
            f"  Обнаружено profile_id с несколькими entity_id. "
            f"Будет выполнена унификация."
        )
    df = unify_entity_ids(df)
    # Вычисляем entity_type, если колонка отсутствует
    if 'entity_type' not in df.columns:
        # Определяем количество профилей в каждой entity
        entity_counts = df.groupby('entity_id')['profile_id'].transform('nunique')
        df['entity_type'] = entity_counts.apply(
            lambda x: 'multi_profile' if x > 1 else 'single_profile'
        )
        logger.info("Колонка 'entity_type' вычислена автоматически.")

    after_fix = df.groupby('profile_id')['entity_id'].nunique().max()
    logger.info(f"  После унификации: max entity_id на profile_id = {after_fix}")

    # 6. Нормализация контактов
    logger.info("Шаг 6/6: нормализация контактов...")
    df = normalize_contacts(df)
    logger.info(f"  Добавлены колонки: email_normalized, email_domain, "
                f"phone_normalized, phone_prefix, first_name_clean, last_name_clean, birthday_clean")
    logger.info("-" * 50)
    logger.info(f"Предобработка завершена. Строк: {len(df)}, колонок: {df.shape[1]}")
    logger.info("=" * 50)

    return df
