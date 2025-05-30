import pandas as pd
from typing import Dict, Optional


class FamilyClassifier:
    def __init__(self, data_path: str, **read_csv_kwargs):
        """Инициализация с поддержкой параметров чтения CSV"""
        try:
            # Пробуем загрузить с указанной кодировкой
            self.df = pd.read_csv(data_path, **read_csv_kwargs)
        except UnicodeDecodeError as e:
            # Если ошибка кодировки, пробуем latin-1
            print(f"Ошибка кодировки: {e}. Пробуем latin-1...")
            self.df = pd.read_csv(data_path, encoding='latin-1',
                                  **{k: v for k, v in read_csv_kwargs.items() if k != 'encoding'})

        self.city_stats = self._calculate_city_stats()

    def _calculate_city_stats(self) -> Dict[str, Dict]:
        """Рассчитываем статистики по городам"""
        return self.df.groupby('City').agg({
            'Median': 'median',
            'Mean': 'mean',
            'ALand': 'median',
            'AWater': 'median'
        }).to_dict('index')

    def _get_city_thresholds(self, city: str) -> Optional[Dict]:
        """Получаем пороговые значения для города"""
        return self.city_stats.get(city)

    def _check_condition(self, family: pd.Series, thresholds: Dict, rules: list) -> bool:
        """Рекурсивная проверка условий"""
        for condition in rules:
            if isinstance(condition, list):
                if all(self._check_condition(family, thresholds, condition)):
                    return True
            else:
                field, op, value = condition
                family_val = family[field]
                threshold_val = thresholds[value] if isinstance(value, str) else value

                if op == '>' and family_val > threshold_val:
                    continue
                elif op == '<' and family_val < threshold_val:
                    continue
                elif op == '>=' and family_val >= threshold_val:
                    continue
                elif op == '<=' and family_val <= threshold_val:
                    continue
                else:
                    return False
        return True

    def classify_family(self, family: pd.Series) -> str:
        """Основной метод классификации с обратным выводом"""
        thresholds = self._get_city_thresholds(family['City'])
        if not thresholds:
            return "Недостаточно данных по городу"

        # Правила классификации (можно расширять)
        rules = {
            'Обеспеченная': [
                [('Median', '>', 2.0 * thresholds['Median']),
                 ('ALand', '>', 1.5 * thresholds['ALand'])],
                [('Mean', '>', 2.5 * thresholds['Mean'])]
            ],
            'Средний класс': [
                [('Median', '>=', thresholds['Median']),
                 ('Median', '<=', 2.0 * thresholds['Median'])],
                [('ALand', '>=', 0.8 * thresholds['ALand']),
                 ('ALand', '<=', 1.5 * thresholds['ALand'])]
            ],
            'Бедная': [
                [('Median', '<', thresholds['Median']),
                 ('Median', '>=', 0.5 * thresholds['Median'])]
            ],
            'За гранью бедности': [
                [('Median', '<', 0.5 * thresholds['Median'])],
                [('ALand', '<', 0.3 * thresholds['ALand'])]
            ]
        }

        # Обратный вывод (проверка от высшего класса к низшему)
        for category in ['Обеспеченная', 'Средний класс', 'Бедная', 'За гранью бедности']:
            for rule_set in rules[category]:
                if self._check_condition(family, thresholds, rule_set):
                    return category

        return "Не определено"


if __name__ == "__main__":
    classifier = FamilyClassifier(
        'kaggle_income.csv',
        encoding='latin-1',
        sep=',',
        usecols=['City', 'Median', 'Mean', 'ALand', 'AWater']
    )

    test_family = pd.Series({
        'City': 'New York',
        'Median': 85000,
        'Mean': 95000,
        'ALand': 1500,
        'AWater': 200
    })

    result = classifier.classify_family(test_family)
    print(f"Категория семьи: {result}")
