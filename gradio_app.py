import gradio as gr
import pandas as pd
import requests

from converter import convert_parquet_to_json, convert_json_to_parquet
from PIL import Image


RESPONSE_KEY = '12345-67890-09876-54321'  # Заглушка для ключа.

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
    

def get_dublicated(parquet_file):
    """
    Получает Parquet файл и отправляет json на проверку дубликатов.
    """
    if parquet_file is None:
        return "Пожалуйста, загрузите .parquet файл."
    try:
        data_json = convert_parquet_to_json(parquet_file)

        info = f"**Отправка .json на проверку**\n"
        # # Отправляю по Fast API на проверку дубликатов (здесь нужно будет реализовать реальный запрос к API)
        # # В проде реализовать отправку батчами (файл может быть большим)
        # response = requests.post("http://......", json=data_json) 
        # # Получаю ключ для получения результатов проверки дубликатов
        # key = response.json()......
        # RESPONSE_KEY = key 

        # Возвращаю info, что данные отправлены на проверку дубликатов
        info += f"**Файл успешно отправлен на проверку дубликатов!**\n"        
        info += f"**Ключ для получения результатов:** {RESPONSE_KEY}\n"        
        return info
    except Exception as e:
        return f"Произошла ошибка при чтении файла: {str(e)}"


def get_merged(uuid):
    """
    Получает Parquet файл и отправляет json на получение результатов объединения.
    """
    if uuid is None:
        return "Пожалуйста, введите UUID."
    try:
        # # Отправляю по Fast API запрос на получение результатов объединения по ключу
        response = requests.get(
            f"https://habr.com/ru/articles/142019/",
            verify=False
            )
        # # Получаю json с результатами объединения
        # Возвращаю info, что данные успешно получены  
        info = f'Status code: {str(response.status_code)}'        
        return info
        # info = f"**Результаты объединения успешно получены!**\n\n"        
        # return info
    except Exception as e:
        return f"Произошла ошибка при получении результатов объединения: {str(e)}"


with gr.Blocks() as web_app:
    # Заголовок и описание проекта
    gr.Markdown("# **Разработка системы матчинга и дедупликации профилей клиентов на маркетплейсе скидок**")
    gr.Markdown("""
                ## **🎯 Цель:** <br> 
                создать алгоритм и прототип системы автоматического 
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
        gr.Markdown("Реализация ML части проекта")

        try:
            img = Image.open("Images/Pizhurin_Y.png")
            gr.Image(img, label="Пижурин Юрий", height=200)
        except:            
            gr.Markdown("*Изображение не найдено*")                                  
        gr.Markdown("Реализация Frontend части проекта")

        try:
            img = Image.open("Images/Vorobkin_A.png")
            gr.Image(img, label="Воробкин Артём", height=200)
        except:            
            gr.Markdown("*Изображение не найдено*")                                  
        gr.Markdown("Реализация Backend части проекта")

        try:
            img = Image.open("Images/Solovyov_E.png")
            gr.Image(img, label="Соловьев Егор", height=200)
        except:            
            gr.Markdown("*Изображение не найдено*")                                  
        gr.Markdown("Реализация EDA части проекта")

    with gr.Tabs():
        with gr.TabItem("📊 EDA"):
            gr.Markdown("## ЗАГЛУШКА: Результаты EDA будут представлены здесь, если успею")
            # Добавить графики с EDA и выводы

        with gr.TabItem("📂 Загрузка данных"):
            gr.Markdown("## Загрузите ваш .parquet файл")
            # Компонент для загрузки файла
            file_upload = gr.File(label="Выберите .parquet файл", file_types=[".parquet"])
            # Компонент для вывода информации о файле
            output_info = gr.Markdown("UUID для получения результата:")
            
            # Загружаем файл с применением функции process_parquet_file
            file_upload.change(fn=process_parquet_file, inputs=file_upload, outputs=output_info)
            
            button_check = gr.Button(
                "Проверить наличие дубликатов",
                variant="primary",
                scale=1,
                min_width=100
                )
            
            button_check.click(fn=get_dublicated, inputs=file_upload, outputs=output_info)
            
        
        with gr.TabItem("🔗 Данные для объединения"):
            gr.Markdown("## ЗАГЛУШКА: Данные для объединения. Разделить " \
            "отдельно автоматически объединенные и для ручной проверки")
            
            # Добавить поле для ввода UUID, по которому можно будет получить результаты объединения по ключу, который возвращается при отправке данных на проверку дубликатов
            uuid_input = gr.Textbox(label="Введите UUID для получения результатов объединения", placeholder="Введите UUID", max_lines=1)
            # Компонент для вывода информации о файле
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
