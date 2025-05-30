import pandas as pd
from main2 import FamilyClassifier

# Инициализация с путями к JSON
classifier = FamilyClassifier(
    city_thresholds_path='city_thresholds.json',
    rules_path='classification_rules.json'
)

# Список тестов: каждый кортеж содержит (данные семьи, ожидаемая категория)
test_cases = [
    ({'City': 'New York', 'Median': 200000, 'Mean': 220000, 'ALand': 3000, 'AWater': 100, 'Stdev': 0.4}, 'Обеспеченная'),
    ({'City': 'New York', 'Median': 100000, 'Mean': 105000, 'ALand': 1200, 'AWater': 50, 'Stdev': 0.9}, 'Средний класс'),
    ({'City': 'New York', 'Median': 60000, 'Mean': 70000, 'ALand': 700, 'AWater': 30, 'Stdev': 1.2}, 'Бедная'),
    ({'City': 'New York', 'Median': 25000, 'Mean': 30000, 'ALand': 200, 'AWater': 20, 'Stdev': 2.5}, 'За гранью бедности'),
    ({'City': 'New York', 'Median': 140000, 'Mean': 160000, 'ALand': 2500, 'AWater': 80, 'Stdev': 0.3}, 'Обеспеченная'),
    ({'City': 'New York', 'Median': 95000, 'Mean': 97000, 'ALand': 1000, 'AWater': 60, 'Stdev': 0.8}, 'Средний класс'),
    ({'City': 'New York', 'Median': 50000, 'Mean': 55000, 'ALand': 500, 'AWater': 40, 'Stdev': 1.5}, 'Бедная'),
    ({'City': 'New York', 'Median': 15000, 'Mean': 20000, 'ALand': 100, 'AWater': 10, 'Stdev': 3.0}, 'За гранью бедности'),
    ({'City': 'New York', 'Median': 180000, 'Mean': 195000, 'ALand': 3200, 'AWater': 150, 'Stdev': 0.2}, 'Обеспеченная'),
    ({'City': 'New York', 'Median': 80000, 'Mean': 90000, 'ALand': 1100, 'AWater': 70, 'Stdev': 0.7}, 'Средний класс')
]

print("Результаты тестирования:\n")
for i, (family_data, expected) in enumerate(test_cases, 1):
    family_series = pd.Series(family_data)
    actual = classifier.classify_family(family_series)
    print(f"Тест {i}: Ожидалось: {expected} | Получено: {actual}")
