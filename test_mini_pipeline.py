# test_mini_pipeline.py — проверка полного пайплайна

import time
from pathlib import Path

import pickle
import pandas as pd

from src import process_profiles

# Пути
DATA_PATH = Path("data/raw/split_label_train_V2.snappy.parquet")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Чтение данных
data = pd.read_parquet(DATA_PATH)

print("=" * 60)
print("ЗАПУСК МИНИ-ПАЙПЛАЙНА")
print("=" * 60)

df_raw = data.copy()

start = time.time()
profiles_df, site_index, cat_index = process_profiles(df_raw)
elapsed = time.time() - start

# Проверки
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ МИНИ-ПАЙПЛАЙНА")
print("=" * 60)

print(f"\n1. ПАСПОРТ ПРОФИЛЯ:")
print(f"   Профилей: {len(profiles_df):,}")
print(f"   Колонок: {len(profiles_df.columns)}")
print(f"   Пример колонок: {list(profiles_df.columns[:5])}...")

print(f"\n2. ИНДЕКС САЙТОВ:")
print(f"   Профилей в индексе: {len(site_index):,}")
sample_pid = list(site_index.keys())[0]
print(f"   Категорий: {len(site_index[sample_pid])}")
non_empty = sum(1 for v in site_index[sample_pid].values() if v)
print(f"   Непустых категорий у первого профиля: {non_empty}")
profiles_with_sites = sum(1 for v in site_index.values() if v.get("_all"))
print(f"   Профилей с хотя бы одним site_id: {profiles_with_sites:,}")

print(f"\n3. ИНДЕКС ПОЧТОВОЙ АКТИВНОСТИ:")
print(f"   Профилей в индексе: {len(cat_index):,}")
sample_cats = [k for k in cat_index[sample_pid].keys() if k != "_all"]
print(f"   Категорий: {len(sample_cats)}")
profiles_with_cats = sum(1 for v in cat_index.values() if v.get("_all"))
print(f"   Профилей с хотя бы одной реакцией: {profiles_with_cats:,}")

print(f"\n4. ОБЩЕЕ ВРЕМЯ:")
print(f"    {elapsed:.0f} сек (~{elapsed/60:.1f} мин)")

# 4. Проверяем согласованность
print(f"\n5. СОГЛАСОВАННОСТЬ:")
ids_df = set(profiles_df.index)
ids_site = set(site_index.keys())
ids_cat = set(cat_index.keys())
print(f"   profile_id в profiles_df: {len(ids_df):,}")
print(f"   profile_id в site_index:  {len(ids_site):,}")
print(f"   profile_id в cat_index:   {len(ids_cat):,}")
print(f"   Все совпадают: {ids_df == ids_site == ids_cat}")

# Сохранение результатов
print(f"\n6. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ:")
print(f"   Папка: {OUTPUT_DIR}")

# Создаём копию для сохранения (чтобы не менять исходный profiles_df)
profiles_save = profiles_df.copy()

# Преобразуем frozenset -> строка для сохранения в CSV/Parquet
set_cols = ["email_normalized", "email_domain", "phone_normalized",
            "phone_prefix", "device", "osfamily", "browser"]

for col in set_cols:
    profiles_save[col] = profiles_save[col].apply(
        lambda x: ", ".join(sorted(x)) if isinstance(x, frozenset) and x else ""
    )

# Паспорт профиля -> CSV
profiles_path = OUTPUT_DIR / "profiles_df.csv"
profiles_save.to_csv(profiles_path)
print(f"   profiles_df.csv ({len(profiles_df):,} строк)")

# Паспорт профиля -> Parquet
profiles_parquet_path = OUTPUT_DIR / "profiles_df.parquet"
profiles_save.to_parquet(profiles_parquet_path)
print(f"   profiles_df.parquet")

# site_index -> Pickle (словарь с множествами — сохраняется как есть)
import pickle
site_path = OUTPUT_DIR / "site_index.pkl"
with open(site_path, "wb") as f:
    pickle.dump(site_index, f)
print(f"   site_index.pkl ({len(site_index):,} записей)")

# cat_index -> Pickle
cat_path = OUTPUT_DIR / "cat_index.pkl"
with open(cat_path, "wb") as f:
    pickle.dump(cat_index, f)
print(f"   cat_index.pkl ({len(cat_index):,} записей)")

# Сводка -> TXT
summary_path = OUTPUT_DIR / "summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"Датасет: {DATA_PATH.name}\n")
    f.write(f"Профилей: {len(profiles_df):,}\n")
    f.write(f"Колонок: {len(profiles_df.columns)}\n")
    f.write(f"Профилей с цифровым следом: {profiles_with_sites:,}\n")
    f.write(f"Профилей с почтовой активностью: {profiles_with_cats:,}\n")
    f.write(f"Время обработки: {elapsed:.0f} сек\n")
    f.write(f"Дата: {pd.Timestamp.now()}\n")
print(f"   summary.txt")
