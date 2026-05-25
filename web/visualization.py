import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime


def plot_missing_values(df):
    missing = (
        (df.isna().sum() / df.shape[0]) * 100
    ).round(2)
    missing = missing.sort_values(ascending=False)    
    fig, ax = plt.subplots(figsize=(12, 5))
    missing.plot(kind="barh", ax=ax)
    ax.set_title("Процент пропущенных значений")

    return fig

def visits_day_of_week(df):
    days_map = {
        0: 'Понедельник',
        1: 'Вторник',
        2: 'Среда',
        3: 'Четверг',
        4: 'Пятница',
        5: 'Суббота',
        6: 'Воскресенье'
    }
    created_at = df['created_at']
    temp_res = (
        created_at.dt.day_of_week.value_counts()
        .sort_values(ascending=False)
        .head(10)
        )
    temp_res = temp_res.rename(index=days_map)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=temp_res.index, y=temp_res.values, ax=ax)
    ax.set_xlabel("День недели")
    ax.set_ylabel("Количество посещений")
    ax.set_title("Распределение посещений по дням недели")
    return fig

def plot_destribute_by_age(df):
    birthday = df['birthday']
    birthday = pd.to_datetime(birthday)
    temp_res = df.copy()
    temp_res['age'] = datetime.now().year - birthday.dt.year
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(temp_res['age'], bins=30, kde=True, ax=ax)    
    ax.set_title("Распределение по возрасту")
    ax.set_xlabel("Возраст")
    ax.set_ylabel("Количество пользователей")
    return fig

def plot_by_sex(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.countplot(x='sex', data=df, ax=ax)
    ax.set_title("Распределение по полу")
    ax.set_xlabel("Пол")
    ax.set_ylabel("Количество пользователей")   
    return fig

def plot_non_procecing_features(df):
    non_processing_features = df['non_processing_features']
    valid_idxs = non_processing_features.dropna().index
    df_valid = pd.DataFrame(
        non_processing_features.dropna().apply(
            lambda arr: dict(
                item.split(': ', 1) if ': ' in item 
                else (item.split(':', 1)[0], item.split(':', 1)[1].strip()) if ':' in item 
                else (item, True) 
                for item in arr
            )
        ).tolist(),
        index=valid_idxs
    )
    non_processing_features_df = df_valid.reindex(non_processing_features.index)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    plt.subplots_adjust(hspace=0.4)
    plt.title(
        "Распределение по необрабатываемым признакам (device, browser, osfamily)"
        , fontsize=16
        )
    sns.histplot(non_processing_features_df['device'], bins=30, ax=axes[0])
    axes[0].set_title("Распределение по устройствам")
    axes[0].set_xlabel("Устройство")
    axes[0].set_ylabel("Количество пользователей")
    
    sns.histplot(non_processing_features_df['browser'], bins=30, ax=axes[1])
    axes[1].set_title("Распределение по браузерам")
    axes[1].set_xlabel("Браузер")
    axes[1].set_ylabel("Количество пользователей")
   
    sns.histplot(non_processing_features_df['osfamily'], bins=30, ax=axes[2])
    axes[2].set_title("Распределение по операционным системам")
    axes[2].set_xlabel("Операционная система")
    axes[2].set_ylabel("Количество пользователей")
    
    plt.tight_layout()
    
    return fig
