import pandas as pd
import numpy as np
import io

def convert_json_to_parquet(file_json):
    """Конвертирует JSON данные в bytes parquet"""
    df = pd.DataFrame(file_json)
    
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    
    return buffer.getvalue()


def convert_parquet_to_json(file_parquet):
    """Конвертирует bytes parquet в JSON данные"""
    if isinstance(file_parquet, bytes):
        buffer = io.BytesIO(file_parquet)
        df = pd.read_parquet(buffer)
    else:
        df = pd.read_parquet(file_parquet)
    
    # Конвертируем в список словарей
    data = df.to_dict(orient='records')
            
    return convert_to_native(data)


# Рекурсивно конвертируем все numpy типы в Python типы
def convert_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    else:
        return obj