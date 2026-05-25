import json

import gradio as gr
import pandas as pd
import requests

from converter import convert_parquet_to_json, convert_json_to_parquet
from PIL import Image
from visualization import (
    plot_by_sex, 
    plot_destribute_by_age,
    plot_missing_values, 
    plot_non_procecing_features, 
    visits_day_of_week
    )


def process_parquet_file(file):
    """
    Загружает Parquet файл и возвращает инфу и первые 5 строк.
    """
    if file is None:
        return "Пожалуйста, загрузите .parquet файл."
    try:
        df = pd.read_parquet(file)
        
        # информация о файле и первые 5 строк df
        preview = df.head().to_string()
        info = f"**Файл успешно загружен!**\n\n"
        info += f"**Форма данных:** {df.shape[0]} строк, {df.shape[1]} колонок\n"
        info += f"**Колонки:** {', '.join(df.columns)}\n\n"
        info += "**Предпросмотр (первые 5 строк):**\n```\n"
        info += preview
        info += "\n```"
        return info
    except Exception as e:
        return f"Произошла ошибка при чтении файла: {str(e)}"


def generate_eda(file):
    """
    Получаем путь к .parquet и возвращаем графики
    """
    if file is None:
        return None, gr.update(visible=False)

    df = pd.read_parquet(file.name)

    fig_missing = plot_missing_values(df)
    fig_visits = visits_day_of_week(df)
    fig_age = plot_destribute_by_age(df)
    fig_sex = plot_by_sex(df)
    fig_non_procecing_features = plot_non_procecing_features(df)
    
    return (
        fig_missing,
        fig_visits,
        fig_age,
        fig_sex,
        fig_non_procecing_features,
        gr.update(visible=True)
        )
          

def get_dublicated(parquet_file):
    """
    Получает .parquet файл и отправляет .json на проверку дубликатов.
    """
    if parquet_file is None:
        return "Пожалуйста, загрузите .parquet файл."
    try:
        data_json = convert_parquet_to_json(parquet_file)

        info = f"**Отправка .json на проверку**\n\n"
        
        temp_file = "request.json" 
        with open(temp_file, "r") as f:
            data = json.load(f) 
             
        response = requests.post("http://localhost:8000/send-batch", json=data) 
        key = response.json().get("batch_id")
        
        # Возвращаю info, что данные отправлены на проверку дубликатов
        info += f"**Файл успешно отправлен на проверку дубликатов!**\n\n"        
        info += f"**Ключ для получения результатов:** {key}\n"        
        return info
    except Exception as e:
        return f"Произошла ошибка при чтении файла: {str(e)}"


def get_merged(uuid):
    """
    Получает .parquet файл и отправляет .json на получение результатов объединения.
    """
    if uuid is None:
        return "Пожалуйста, введите UUID."
    try:
        # # Отправляю по Fast API запрос на получение результатов объединения по ключу
        response = requests.get(
            f"http://localhost:8000/batch/{uuid}",
            verify=False
            )
        info = f'Status code: {str(response.status_code)}' 
        # info += f'{response.json()}' 
        df = pd.DataFrame(response.json())  
        preview = df.head().to_string()
        info += "**TEST:**\n```\n"
        info += preview
        info += "\n```"
        return info
    except Exception as e:
        return f"Произошла ошибка при получении результатов объединения: {str(e)}"


with gr.Blocks() as web_app:
    # Заголовок и описание проекта
    gr.Markdown("# **Разработка системы матчинга и дедупликации профилей клиентов на маркетплейсе скидок**")
    gr.Markdown("""
                ## **🎯 Цель:** <br> 
                Создать алгоритм и прототип системы автоматического 
                поиска и объединения дубликатов профилей клиентов по неполным 
                и зашумленным данным (ФИО, email, телефон, дата рождения,
                город и др.), чтобы повысить качество клиентской базы, точность
                персонализации и корректность аналитики на маркетплейсе.
                """)

    # Описание команды и ролей участников
    gr.Markdown("## 👥 Команда 22")
    with gr.Row():
        try:
            img = Image.open("Images/Smirnova_A.png")
            gr.Image(img, label="Смирнова Анастасия", height=200)
        except:            
            gr.Markdown("*Изображение не найдено*")                                  
        gr.Markdown('''
                    **ML Pipeline**
                    Построение полного пайплайна решения для Entity Resolution. 
                    Взяла на себя все этапы обработки данных: от предварительной 
                    нормализации, блокинга и алгоритмов мэтчинга, до 
                    кластеризации сущностей. Дополнительно обучила финальную 
                    модель и рассчитала метрики качества.                    
                    ''')

        try:
            img = Image.open("Images/Pizhurin_Y.png")
            gr.Image(img, label="Пижурин Юрий", height=200)
        except:            
            gr.Markdown("*Изображение не найдено*")                                  
        gr.Markdown('''
                    **Frontend**
                    Разработал интерактивный интерфейс на Gradio. Реализовал 
                    визуализацию результатов разведочного анализа данных 
                    (EDA-графики), настроил HTTP-взаимодействие с бэкендом для 
                    отправки данных в модель и полученния результатов в реальном 
                    времени.
                    ''')

        try:
            img = Image.open("Images/Vorobkin_A.png")
            gr.Image(img, label="Воробкин Артём", height=200)
        except:            
            gr.Markdown("*Изображение не найдено*")                                  
        gr.Markdown('''
                    **Backend**
                    Спроектировал и развернул бэкенд-инфраструктуру. 
                    Внедрил Kafka для асинхронной обработки потоков данных, 
                    поднял Redis для быстрого кэширования промежуточных 
                    состояний и завернул всё в контейнеры Docker, обеспечив 
                    воспроизводимость окружения.
                    ''')

        try:
            img = Image.open("Images/Solovyov_E.png")
            gr.Image(img, label="Соловьев Егор", height=200)
        except:            
            gr.Markdown("*Изображение не найдено*")                                  
        gr.Markdown('''
                    **EDA**
                    Провел углубленный разведочный анализ данных (EDA), выявив 
                    скрытые закономерности и аномалии. Выполнил рефакторинг 
                    кодовой базы, улучшив её читаемость, модульность и 
                    поддерживаемость для всей команды.
                    ''')

    with gr.Tabs():
        # Tab с EDA
        with gr.TabItem("📊 EDA"): 
            file_upload_eda = gr.File(
                label="Выберите .parquet файл",
                file_types=[".parquet"]
                )

            with gr.Column(visible=False) as eda_section:
                gr.Markdown("## Результаты EDA")
                
                eda_plot_missing = gr.Plot(
                    label="Пропущенные значения"
                    )

                eda_plot_visits = gr.Plot(
                    label="Посещения по дням недели"
                    )
                
                eda_plot_age = gr.Plot(
                    label="Распределение по возрасту"
                    )

                eda_plot_sex = gr.Plot(
                    label="Распределение по полу"
                    )

                eda_plot_non_processing_features = gr.Plot(
                    label="Распределение по необрабатываемым признакам"
                    )
                
                gr.Markdown("""
                ### Выводы
                
                **Совокупность неизменного технического окружения**, строго 
                            последовательного роста общего счётчика визитов и 
                            временной непрерывности сопровождающиеся хаотичными 
                            изменениями признаков пользователя (`email`, 
                            `gender` и т.п.) позволяет с высокой достоверностью 
                            заключить, что все действия совершены одним человеком 
                            с одного компьютера, который намеренно генерирует 
                            множество учётных записей

                **«Чистые признаки»** представляют собой комбинации «`device` + 
                            `geoname_id` + `osfamily`+ постоянные поведенческие 
                            флаги + стабильные «признаки геоданных» образуют 
                            стабильный цифровой отпечаток и позволяют 
                            предположить, что сессии принадлежат одному лицу. 
                            Медленно изменяющиеся счётчики служат дополнительным 
                            подтверждением - они не должны «скакать» вниз или 
                            хаотично изменяться/обнуляться в рамках одной сессии
                """)

            file_upload_eda.change(
                fn=generate_eda,
                inputs=file_upload_eda,
                outputs=[
                    eda_plot_missing,
                    eda_plot_visits,
                    eda_plot_age,
                    eda_plot_sex,
                    eda_plot_non_processing_features,
                    eda_section
                    ]
                )          
        
        # Tab с загрузкой данных как .parquet
        with gr.TabItem("📂 Загрузка данных"):
            gr.update(visible=False)
            gr.Markdown("## Загрузите ваш .parquet файл")
            file_upload = gr.File(label="Выберите .parquet файл", file_types=[".parquet"])
            output_info = gr.Markdown("UUID для получения результата:")
            
            file_upload.change(fn=process_parquet_file, inputs=file_upload, outputs=output_info)
            
            button_check = gr.Button(
                "Проверить наличие дубликатов",
                variant="primary",
                scale=1,
                min_width=100
                )
            
            button_check.click(fn=get_dublicated, inputs=file_upload, outputs=output_info)
        
        # Tab с получением объединеных данных
        with gr.TabItem("🔗 Объединеные данные"):
            gr.update(visible=False)
            gr.Markdown("## Данные для объединения. Разделить " \
            "отдельно автоматически объединенные и для ручной проверки")
            
            # Добавить поле для ввода UUID, по которому можно будет получить результаты
            uuid_input = gr.Textbox(label="Введите UUID для получения результатов объединения", placeholder="Введите UUID", max_lines=1)
            output_info = gr.Markdown("Здесь появится информация о результатах объединения...")

            button_result = gr.Button(
                "Получить результаты объединения",
                variant="primary",
                scale=1,
                min_width=100
                )
            
            button_result.click(fn=get_merged, inputs=uuid_input, outputs=output_info)
            


if __name__ == "__main__":
    web_app.launch()
