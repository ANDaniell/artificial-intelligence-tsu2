import pandas as pd
import json


def generate_city_thresholds_json(csv_path: str, json_path: str):
    # Загружаем данные
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin-1')

    # Список столбцов для агрегации
    agg_columns = {
        'Median': 'median',
        'Mean': 'mean',
        'ALand': 'median',
        'AWater': 'median'
    }

    # Добавляем Stdev и Households, если они есть в данных
    if 'Stdev' in df.columns:
        agg_columns['Stdev'] = 'median'
    if 'Households' in df.columns:
        agg_columns['Households'] = 'median'

    # Рассчитываем агрегаты по городам
    city_stats = df.groupby('City').agg(agg_columns)

    # Преобразуем в словарь с округлением до 2 знаков
    city_thresholds = city_stats.round(2).to_dict(orient='index')

    # Сохраняем в JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(city_thresholds, f, ensure_ascii=False, indent=4)

    print(f"JSON с порогами по городам сохранен в {json_path}")


if __name__ == "__main__":
    csv_file = 'kaggle_income.csv'  # путь к твоему csv
    json_file = 'city_thresholds.json'  # куда сохранить пороги

    generate_city_thresholds_json(csv_file, json_file)
